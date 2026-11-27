use crate::support::{GEOMETRIES, Rng, cumulative};
use half::f16;
use kodamaembed_cuda::{
    Context, DeviceBuffer, Stream,
    kernels::{CORRECTED_BELOW, Kernels, Packed, RelativeTable},
};
use kodamaembed_model::{
    Geometry,
    relative::{BUCKETS, RelativeBias},
};

/// Per token: Q and K in FP32, V as stored (FP16), each `width` wide.
struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f16>,
}

impl Inputs {
    /// Queries `query_scale` times wider than keys and values.
    fn new(tokens: usize, width: usize, query_scale: f32, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            q: rng.fill(tokens * width, query_scale),
            k: rng.fill(tokens * width, 1.0),
            v: rng
                .fill(tokens * width, 1.0)
                .into_iter()
                .map(f16::from_f32)
                .collect(),
        }
    }

    /// As the QKV GEMM writes them: [Q | K | V] in FP16, [Q | K]'s residuals by 2^11.
    fn packed(&self, width: usize) -> (Vec<f16>, Vec<f16>) {
        let tokens = self.v.len() / width;
        let (mut qkv, mut low) = (Vec::new(), Vec::new());
        for t in 0..tokens {
            let row = t * width..(t + 1) * width;
            for &value in self.q[row.clone()].iter().chain(&self.k[row.clone()]) {
                let rounded = f16::from_f32(value);
                qkv.push(rounded);
                low.push(f16::from_f32((value - rounded.to_f32()) * 2048.0));
            }
            qkv.extend_from_slice(&self.v[row]);
        }
        (qkv, low)
    }
}

/// The attention a layer runs: over each whole sequence, or over the keys within `reach` rows
/// of the query.
#[derive(Clone, Copy)]
enum Span<'a> {
    Whole(Option<&'a RelativeBias>),
    Local { reach: usize },
}

/// Softmax(q . k / sqrt(D) + bias) v per sequence and head in f64 over the keys of `span`, from
/// FP32 Q and K or, over whole sequences of CORRECTED_BELOW tokens or more, from their FP16 parts.
fn reference(geometry: Geometry, inputs: &Inputs, lengths: &[usize], span: Span) -> Vec<f64> {
    let (width, heads, d_head) = (geometry.hidden(), geometry.heads(), geometry.head_dim());
    let tokens: usize = lengths.iter().sum();
    let mut out = vec![0.0; tokens * width];
    let mut start = 0;
    for &len in lengths {
        let corrected = matches!(span, Span::Local { .. }) || len < CORRECTED_BELOW;
        let read = |value: f32| match corrected {
            true => f64::from(value),
            false => f64::from(f16::from_f32(value).to_f32()),
        };
        for head in 0..heads {
            let at = |token: usize, d: usize| (start + token) * width + head * d_head + d;
            for q in 0..len {
                let (keys, bias) = match span {
                    Span::Whole(bias) => (0..len, bias),
                    Span::Local { reach } => (
                        q.saturating_sub(reach)..len.min(q.saturating_add(reach) + 1),
                        None,
                    ),
                };
                let scores: Vec<f64> = keys
                    .clone()
                    .map(|k| {
                        let score = (0..d_head)
                            .map(|d| read(inputs.q[at(q, d)]) * read(inputs.k[at(k, d)]))
                            .sum::<f64>()
                            / (d_head as f64).sqrt();
                        match bias {
                            Some(bias) => score + f64::from(bias.keys(head, q, len)[k]),
                            None => score,
                        }
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..d_head {
                    let value: f64 = weights
                        .iter()
                        .zip(keys.clone())
                        .map(|(w, k)| w * f64::from(inputs.v[at(k, d)].to_f32()))
                        .sum();
                    out[at(q, d)] = value / total;
                }
            }
        }
        start += len;
    }
    out
}

/// A batch of `lengths` as the kernels take it, over the lengths' cumulative sums on the device.
fn batch<'a>(cu_seqlens: &'a DeviceBuffer<i32>, lengths: &[usize]) -> Packed<'a> {
    Packed {
        cu_seqlens,
        sequences: lengths.len(),
        tokens: lengths.iter().sum(),
        max_len: lengths.iter().copied().max().unwrap(),
        min_len: lengths.iter().copied().min().unwrap(),
    }
}

/// The attention kernels' output over `span` for `qkv` and `low` packed as `lengths`.
fn attend(geometry: Geometry, qkv: &[f16], low: &[f16], lengths: &[usize], span: Span) -> Vec<f16> {
    let width = geometry.hidden();
    let tokens: usize = lengths.iter().sum();
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    let kernels = Kernels::load(&context, geometry).unwrap();
    let qkv_gpu = DeviceBuffer::from_slice(&context, qkv).unwrap();
    let low_gpu = DeviceBuffer::from_slice(&context, low).unwrap();
    let cu = DeviceBuffer::from_slice(&context, &cumulative(lengths)).unwrap();
    let out = DeviceBuffer::<f16>::zeroed(&context, tokens * width).unwrap();
    let packed = batch(&cu, lengths);
    // SAFETY: every buffer lives until the stream is synchronized below.
    unsafe {
        match span {
            Span::Whole(bias) => {
                let table = bias.map(|bias| RelativeTable::upload(&context, bias).unwrap());
                kernels.attention(&stream, &qkv_gpu, &low_gpu, &packed, &out, table.as_ref())
            }
            Span::Local { reach } => {
                kernels.local_attention(&stream, &qkv_gpu, &low_gpu, &packed, &out, reach)
            }
        }
        .unwrap()
    };
    stream.synchronize().unwrap();
    out.read().unwrap()
}

fn worst_error(geometry: Geometry, query_scale: f32, lengths: &[usize], span: Span) -> f64 {
    let width = geometry.hidden();
    let inputs = Inputs::new(lengths.iter().sum(), width, query_scale, 11);
    let expected = reference(geometry, &inputs, lengths, span);
    let (qkv, low) = inputs.packed(width);
    attend(geometry, &qkv, &low, lengths, span)
        .iter()
        .zip(&expected)
        .map(|(got, &want)| (f64::from(got.to_f32()) - want).abs())
        .fold(0.0, f64::max)
}

#[test]
fn attention_matches_the_reference_across_block_edges() {
    for geometry in GEOMETRIES {
        // Queries four times wider than keys: scaled scores with a spread near 1.3. The longest
        // sequence sets the query block, so 96 tokens take blocks of 32 rows where they exist.
        for lengths in [
            &[1, 17, 64, 65, 130, 255, 256, 300][..],
            &[1, 17, 49, 64, 65, 96],
        ] {
            let error = worst_error(geometry, 4.0, lengths, Span::Whole(None));
            assert!(error <= 2e-3, "{geometry:?} {lengths:?}: {error}");
        }
    }
}

/// Scores near +-20, which FP16 queries and keys alone would move by about 1e-2.
#[test]
fn large_scores_are_exact_to_fp32() {
    for geometry in GEOMETRIES {
        let error = worst_error(geometry, 64.0, &[40, 129], Span::Whole(None));
        assert!(error <= 2e-3, "{geometry:?}: {error}");
    }
}

/// ModernBERT's 64 rows each side; 8, where a block's later rows see no key of its first key
/// block; the query alone; and 1,000, past every sequence.
#[test]
fn local_attention_matches_the_reference_inside_its_window() {
    let lengths = [1, 17, 64, 65, 130, 255, 256, 300];
    for geometry in GEOMETRIES {
        for reach in [64, 8, 0, 1000] {
            for query_scale in [4.0, 64.0] {
                let error = worst_error(geometry, query_scale, &lengths, Span::Local { reach });
                assert!(
                    error <= 2e-3,
                    "{geometry:?}, reach {reach}, queries {query_scale} wide: {error}"
                );
            }
        }
    }
}

/// With every key in reach a window changes nothing: short sequences, which the kernel over
/// whole sequences corrects too, keep its bits.
#[test]
fn a_window_past_the_sequence_gives_the_bits_of_attention_over_all_of_it() {
    let lengths = [1, 17, 64, 65, 130, CORRECTED_BELOW - 1];
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let (qkv, low) = Inputs::new(lengths.iter().sum(), width, 4.0, 5).packed(width);
        let bits = |span| -> Vec<u16> {
            attend(geometry, &qkv, &low, &lengths, span)
                .iter()
                .map(|value| value.to_bits())
                .collect()
        };
        assert_eq!(
            bits(Span::Local {
                reach: CORRECTED_BELOW
            }),
            bits(Span::Whole(None)),
            "{geometry:?}"
        );
    }
}

#[test]
fn local_attention_reads_the_low_parts_at_every_length() {
    let lengths = [CORRECTED_BELOW - 1, CORRECTED_BELOW + 44];
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let (qkv, low) = Inputs::new(lengths.iter().sum(), width, 4.0, 3).packed(width);
        let zeros = vec![f16::ZERO; low.len()];
        let span = Span::Local { reach: 64 };
        let (read, ignored) = (
            attend(geometry, &qkv, &low, &lengths, span),
            attend(geometry, &qkv, &zeros, &lengths, span),
        );
        let short = lengths[0] * width;
        for (sequence, rows) in [(0, 0..short), (1, short..read.len())] {
            assert!(
                read[rows.clone()]
                    .iter()
                    .zip(&ignored[rows])
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "{geometry:?}, sequence {sequence}"
            );
        }
    }
}

#[test]
fn only_sequences_below_the_limit_read_the_low_parts() {
    let lengths = [CORRECTED_BELOW - 1, CORRECTED_BELOW];
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let heads = geometry.heads();
        let inputs = Inputs::new(lengths.iter().sum(), width, 4.0, 3);
        let (qkv, low) = inputs.packed(width);
        let zeros = vec![f16::ZERO; low.len()];
        let table = RelativeBias::new(&Rng::new(4).fill(BUCKETS * heads, 2.0), heads, 300);
        for bias in [None, Some(&table)] {
            let bits = |low: &[f16]| -> Vec<u16> {
                attend(geometry, &qkv, low, &lengths, Span::Whole(bias))
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            };
            let (read, ignored) = (bits(&low), bits(&zeros));
            let short = lengths[0] * width;
            assert_ne!(
                read[..short],
                ignored[..short],
                "{geometry:?}, bias {}",
                bias.is_some()
            );
            assert_eq!(
                read[short..],
                ignored[short..],
                "{geometry:?}, bias {}",
                bias.is_some()
            );
        }
    }
}

/// MPNet's relative bias joins every scaled score; rows past a sequence end read inside the table.
#[test]
fn relative_bias_attention_matches_the_reference_across_block_edges() {
    for geometry in GEOMETRIES {
        let heads = geometry.heads();
        let bias = RelativeBias::new(&Rng::new(4).fill(BUCKETS * heads, 2.0), heads, 300);
        let span = Span::Whole(Some(&bias));
        let error = worst_error(geometry, 4.0, &[1, 17, 64, 65, 130, 300], span);
        assert!(error <= 2e-3, "{geometry:?}: {error}");
    }
}

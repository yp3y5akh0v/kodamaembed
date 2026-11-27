use crate::support::{GEOMETRIES, Rng, cumulative, devices, dispatch, matrix_devices};
use half::f16;
use kodamaembed_model::{
    Geometry,
    relative::{BUCKETS, RelativeBias},
};
use kodamaembed_vulkan::{Buffer, Device, KernelSpec, Push, shaders};

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

    /// As the QKV GEMM writes them: [Q | K | V] in FP16, [Q | K]'s residuals in FP32.
    fn packed(&self, width: usize) -> (Vec<f16>, Vec<f32>) {
        let tokens = self.v.len() / width;
        let (mut qkv, mut low) = (Vec::new(), Vec::new());
        for t in 0..tokens {
            let row = t * width..(t + 1) * width;
            for &value in self.q[row.clone()].iter().chain(&self.k[row.clone()]) {
                let rounded = f16::from_f32(value);
                qkv.push(rounded);
                low.push(value - rounded.to_f32());
            }
            qkv.extend_from_slice(&self.v[row]);
        }
        (qkv, low)
    }
}

/// What a query attends over: its whole sequence, with MPNet's bias or without, or the keys
/// within `reach` rows of it.
#[derive(Clone, Copy)]
enum Span {
    Whole,
    Biased,
    Local { reach: usize },
}

/// Softmax(q . k / sqrt(D) + bias) v per sequence and head in f64, from FP32 Q and K, over the
/// keys within `reach` rows of each query when one is given.
fn reference(
    geometry: Geometry,
    inputs: &Inputs,
    lengths: &[usize],
    bias: Option<&RelativeBias>,
    reach: Option<usize>,
) -> Vec<f64> {
    let (width, heads, d_head) = (geometry.hidden(), geometry.heads(), geometry.head_dim());
    let tokens: usize = lengths.iter().sum();
    let mut out = vec![0.0; tokens * width];
    let mut start = 0;
    for &len in lengths {
        for head in 0..heads {
            let at = |token: usize, d: usize| (start + token) * width + head * d_head + d;
            for q in 0..len {
                let keys = match reach {
                    Some(reach) => q.saturating_sub(reach)..len.min(q.saturating_add(reach) + 1),
                    None => 0..len,
                };
                let scores: Vec<f64> = keys
                    .clone()
                    .map(|k| {
                        let score = (0..d_head)
                            .map(|d| f64::from(inputs.q[at(q, d)]) * f64::from(inputs.k[at(k, d)]))
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

/// FP16 output rounding (2^-12 near 1) plus the matrix path's FP16 weights (2^-11 relative).
const TOLERANCE: f64 = 1e-3;

struct Shader {
    spirv: &'static [u8],
    subgroup: Option<u32>,
}

const PORTABLE: Shader = Shader {
    spirv: shaders::ATTENTION,
    subgroup: None,
};

const MATRIX: Shader = Shader {
    spirv: shaders::ATTENTION_MATRIX,
    subgroup: Some(32),
};

const PORTABLE_BIAS: Shader = Shader {
    spirv: shaders::ATTENTION_BIAS,
    subgroup: None,
};

const MATRIX_BIAS: Shader = Shader {
    spirv: shaders::ATTENTION_MATRIX_BIAS,
    subgroup: Some(32),
};

const PORTABLE_LOCAL: Shader = Shader {
    spirv: shaders::ATTENTION_LOCAL,
    subgroup: None,
};

const MATRIX_LOCAL: Shader = Shader {
    spirv: shaders::ATTENTION_MATRIX_LOCAL,
    subgroup: Some(32),
};

/// A random bias of up to +-2 for every geometry's heads, over sequences of up to 300 tokens.
fn relative(geometry: Geometry) -> RelativeBias {
    let heads = geometry.heads();
    RelativeBias::new(&Rng::new(4).fill(BUCKETS * heads, 2.0), heads, 300)
}

fn check(shader: &Shader, devices: &[Device], query_scale: f32, lengths: &[usize], span: Span) {
    let tokens: usize = lengths.iter().sum();
    let max_len = lengths.iter().copied().max().unwrap() as u32;
    let (biased, reach) = match span {
        Span::Whole => (false, None),
        Span::Biased => (true, None),
        Span::Local { reach } => (false, Some(reach)),
    };
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let bias = biased.then(|| relative(geometry));
        let inputs = Inputs::new(tokens, width, query_scale, 11);
        let expected = reference(geometry, &inputs, lengths, bias.as_ref(), reach);
        let (qkv, low) = inputs.packed(width);
        let constants = [geometry.head_dim() as u32, geometry.heads() as u32];
        let spec = KernelSpec {
            spirv: shader.spirv,
            bindings: 4 + u32::from(biased),
            push_bytes: 4 + 4 * u32::from(biased || reach.is_some()),
            constants: &constants,
            subgroup_size: shader.subgroup,
        };
        let scale = 1.0 / (geometry.head_dim() as f32).sqrt();
        for device in devices {
            let qkv_gpu = Buffer::from_slice(device, &qkv).unwrap();
            let low_gpu = Buffer::from_slice(device, &low).unwrap();
            let cu = Buffer::from_slice(device, &cumulative(lengths)).unwrap();
            let out = Buffer::<f16>::zeroed(device, tokens * width).unwrap();
            let table = bias
                .as_ref()
                .map(|bias| Buffer::from_slice(device, bias.values()).unwrap());
            let groups = [
                max_len.div_ceil(64),
                geometry.heads() as u32,
                lengths.len() as u32,
            ];
            let bindings: Vec<_> = [
                qkv_gpu.binding(),
                low_gpu.binding(),
                cu.binding(),
                out.binding(),
            ]
            .into_iter()
            .chain(table.as_ref().map(Buffer::binding))
            .collect();
            let push = match (&bias, reach) {
                (Some(bias), _) => Push::new().f32(scale).u32(bias.longest() as u32),
                // As the encoder pushes it: no key lies farther than the longest sequence.
                (None, Some(reach)) => Push::new().f32(scale).u32((reach as u32).min(max_len)),
                (None, None) => Push::new().f32(scale),
            };
            dispatch(device, &spec, &bindings, &push, groups);
            let (index, error) = out
                .read(device)
                .unwrap()
                .iter()
                .zip(&expected)
                .map(|(got, &want)| (f64::from(got.to_f32()) - want).abs())
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert!(
                error <= TOLERANCE,
                "{} {geometry:?}: error {error} at element {index}",
                device.name
            );
        }
    }
}

// Queries four times wider than keys: scaled scores with a spread near 1.3.
const BLOCK_EDGES: [usize; 6] = [1, 17, 64, 65, 130, 300];

// Scores near +-20, which FP16 queries and keys alone would move by about 1e-2.
const LARGE_SCORES: [usize; 2] = [40, 129];

#[test]
fn attention_matches_the_reference_across_block_edges_on_every_gpu() {
    check(&PORTABLE, &devices(), 4.0, &BLOCK_EDGES, Span::Whole);
}

#[test]
fn large_scores_are_exact_to_fp32_on_every_gpu() {
    check(&PORTABLE, &devices(), 64.0, &LARGE_SCORES, Span::Whole);
}

#[test]
fn the_matrix_attention_matches_the_reference_across_block_edges_where_supported() {
    check(&MATRIX, &matrix_devices(), 4.0, &BLOCK_EDGES, Span::Whole);
}

#[test]
fn the_matrix_attention_keeps_large_scores_exact_where_supported() {
    check(&MATRIX, &matrix_devices(), 64.0, &LARGE_SCORES, Span::Whole);
}

#[test]
fn relative_bias_attention_matches_the_reference_across_block_edges_on_every_gpu() {
    check(&PORTABLE_BIAS, &devices(), 4.0, &BLOCK_EDGES, Span::Biased);
}

#[test]
fn the_matrix_relative_bias_attention_matches_the_reference_where_supported() {
    check(
        &MATRIX_BIAS,
        &matrix_devices(),
        4.0,
        &BLOCK_EDGES,
        Span::Biased,
    );
}

/// ModernBERT's 64 rows each side; 8, where a workgroup's later rows see no key of the first
/// block it reads; the query alone; and 1,000, past every sequence.
const REACHES: [usize; 4] = [64, 8, 0, 1000];

#[test]
fn local_attention_matches_the_reference_inside_its_window_on_every_gpu() {
    for reach in REACHES {
        let span = Span::Local { reach };
        check(&PORTABLE_LOCAL, &devices(), 4.0, &BLOCK_EDGES, span);
        check(&PORTABLE_LOCAL, &devices(), 64.0, &LARGE_SCORES, span);
    }
}

#[test]
fn the_matrix_local_attention_matches_the_reference_where_supported() {
    for reach in REACHES {
        let span = Span::Local { reach };
        check(&MATRIX_LOCAL, &matrix_devices(), 4.0, &BLOCK_EDGES, span);
        check(&MATRIX_LOCAL, &matrix_devices(), 64.0, &LARGE_SCORES, span);
    }
}

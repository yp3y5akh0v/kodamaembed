use crate::support::{GEOMETRIES, Rng, cumulative};
use half::f16;
use kodamaembed_cuda::{
    Context, DeviceBuffer, Stream,
    kernels::{Kernels, Packed, PoolerHead, PredictionHead, PredictionKernels},
};
use kodamaembed_model::{
    Activation, Geometry, Head, Pooling, Positions, Rotary, Shape, Tensors,
    reference::{self, Synthetic},
};

const HEADS: [(Pooling, Activation); 4] = [
    (Pooling::Mean, Activation::Gelu),
    (Pooling::Mean, Activation::Silu),
    (Pooling::Cls, Activation::Gelu),
    (Pooling::Cls, Activation::Silu),
];

/// A one-layer ModernBERT as wide as `geometry`, whose synthetic checkpoint holds a prediction
/// head; its epsilon is large enough to count beside the features' variance.
fn head_shape(geometry: Geometry) -> Shape {
    Shape {
        layers: 1,
        hidden: geometry.hidden(),
        heads: geometry.heads(),
        intermediate: geometry.hidden(),
        vocab: 2,
        max_positions: 2,
        type_vocab: 1,
        eps: 0.05,
        positions: Positions::ModernBert(Rotary {
            global_theta: 160_000.0,
            local_theta: 10_000.0,
            local_layers: 0,
            window: 2,
        }),
    }
}

#[test]
fn pooling_is_the_normalized_mean_of_the_rows_each_pooling_names() {
    let lengths = [1, 5, 300];
    let tokens: usize = lengths.iter().sum();
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let hidden = Rng::new(21).fill(tokens * width, 4.0);
        let kernels = Kernels::load(&context, geometry).unwrap();
        let hidden_gpu = DeviceBuffer::from_slice(&context, &hidden).unwrap();
        let cu = DeviceBuffer::from_slice(&context, &cumulative(&lengths)).unwrap();
        let out = DeviceBuffer::<f32>::zeroed(&context, lengths.len() * width).unwrap();
        let scratch = DeviceBuffer::<f16>::zeroed(&context, lengths.len() * width * 4).unwrap();
        let packed = Packed {
            cu_seqlens: &cu,
            sequences: lengths.len(),
            tokens,
            max_len: 300,
            min_len: 1,
        };
        for pooling in [Pooling::Mean, Pooling::Cls] {
            // SAFETY: every buffer lives until the stream is synchronized below.
            unsafe {
                kernels
                    .pool_normalize(&stream, pooling, &hidden_gpu, &packed, &out, &scratch)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let got = out.read().unwrap();
            let mut start = 0;
            for (sequence, &len) in lengths.iter().enumerate() {
                let rows = pooling.rows(len);
                let mean: Vec<f64> = (0..width)
                    .map(|d| {
                        (start..start + rows)
                            .map(|t| f64::from(hidden[t * width + d]))
                            .sum::<f64>()
                            / rows as f64
                    })
                    .collect();
                let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
                for (d, &m) in mean.iter().enumerate() {
                    let want = m / norm;
                    let have = f64::from(got[sequence * width + d]);
                    assert!(
                        (have - want).abs() <= want.abs() * 1.2e-7 + 1e-12,
                        "{geometry:?} {pooling:?} {have} vs {want}"
                    );
                }
                start += len;
            }
        }
    }
}

#[test]
fn the_reranking_head_scores_each_sequence_from_its_first_row() {
    let lengths = [4, 1, 37];
    let tokens: usize = lengths.iter().sum();
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let mut rng = Rng::new(43);
        let hidden = rng.fill(tokens * width, 2.0);
        let pooler_w = rng.fill(width * width, 0.05);
        let pooler_b = rng.fill(width, 0.1);
        let classifier_w = rng.fill(width, 0.2);
        let classifier_b = rng.fill(1, 0.5);

        let kernels = Kernels::load(&context, geometry).unwrap();
        let upload = |values: &[f32]| DeviceBuffer::from_slice(&context, values).unwrap();
        let hidden_gpu = upload(&hidden);
        let (pooler_w_gpu, pooler_b_gpu) = (upload(&pooler_w), upload(&pooler_b));
        let (classifier_w_gpu, classifier_b_gpu) = (upload(&classifier_w), upload(&classifier_b));
        let head = PoolerHead {
            pooler_w: &pooler_w_gpu,
            pooler_b: &pooler_b_gpu,
            classifier_w: &classifier_w_gpu,
            classifier_b: &classifier_b_gpu,
        };
        let cu = DeviceBuffer::from_slice(&context, &cumulative(&lengths)).unwrap();
        let scores = DeviceBuffer::<f32>::zeroed(&context, lengths.len()).unwrap();
        let packed = Packed {
            cu_seqlens: &cu,
            sequences: lengths.len(),
            tokens,
            max_len: 37,
            min_len: 1,
        };
        // SAFETY: every buffer lives until the stream is synchronized below.
        unsafe {
            kernels
                .rerank_head(&stream, &hidden_gpu, &packed, &head, &scores)
                .unwrap();
        }
        stream.synchronize().unwrap();
        let got = scores.read().unwrap();
        let mut start = 0;
        for (sequence, &len) in lengths.iter().enumerate() {
            let first = &hidden[start * width..(start + 1) * width];
            let want: f64 = (0..width)
                .map(|f| {
                    let row = &pooler_w[f * width..(f + 1) * width];
                    let pre = f64::from(pooler_b[f])
                        + row
                            .iter()
                            .zip(first)
                            .map(|(&w, &x)| f64::from(w) * f64::from(x))
                            .sum::<f64>();
                    pre.tanh() * f64::from(classifier_w[f])
                })
                .sum::<f64>()
                + f64::from(classifier_b[0]);
            let have = f64::from(got[sequence]);
            assert!(
                (have - want).abs() <= 1e-5 * (1.0 + want.abs()),
                "{geometry:?} sequence {sequence}: {have} vs {want}"
            );
            start += len;
        }
    }
}

#[test]
fn reranking_does_not_hide_overflowing_pooler_activations_with_tanh() {
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let kernels = Kernels::load(&context, geometry).unwrap();
        let upload = |values: &[f32]| DeviceBuffer::from_slice(&context, values).unwrap();
        let mut hidden = vec![0.0; 2 * width];
        hidden[..2].fill(1e20);
        let hidden = upload(&hidden);
        let cu = DeviceBuffer::from_slice(&context, &[0, 1, 2]).unwrap();
        let packed = Packed {
            cu_seqlens: &cu,
            sequences: 2,
            tokens: 2,
            max_len: 1,
            min_len: 1,
        };
        let pooler_b = upload(&vec![0.0; width]);
        let mut classifier = vec![0.0; width];
        classifier[0] = 1.0;
        let classifier_w = upload(&classifier);
        let classifier_b = upload(&[0.25]);
        for sign in [1.0, -1.0] {
            let mut weights = vec![0.0; width * width];
            // The exact dot is zero, but the first FP32 FMA overflows and
            // cannot be undone by the cancelling second finite product.
            weights[0] = sign * 1e20;
            weights[1] = -sign * 1e20;
            let pooler_w = upload(&weights);
            let head = PoolerHead {
                pooler_w: &pooler_w,
                pooler_b: &pooler_b,
                classifier_w: &classifier_w,
                classifier_b: &classifier_b,
            };
            let scores = DeviceBuffer::<f32>::zeroed(&context, 2).unwrap();
            // SAFETY: all buffers outlive the synchronization below.
            unsafe {
                kernels
                    .rerank_head(&stream, &hidden, &packed, &head, &scores)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let got = scores.read().unwrap();
            assert!(got[0].is_nan(), "{geometry:?}, sign {sign}: {}", got[0]);
            assert_eq!(got[1], 0.25, "{geometry:?}: ordinary row");
        }
    }
}

#[test]
fn the_prediction_head_scores_each_sequence_from_its_pooled_rows() {
    let lengths = [4usize, 1, 37];
    let tokens: usize = lengths.iter().sum();
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let shape = head_shape(geometry);
        let model = Synthetic::new(&shape, 47);
        let hidden = Rng::new(53).fill(tokens * width, 2.0);
        let rows: Vec<Vec<f64>> = hidden
            .chunks(width)
            .map(|row| row.iter().map(|&x| f64::from(x)).collect())
            .collect();

        let kernels = Kernels::load(&context, geometry).unwrap();
        let prediction = PredictionKernels::load(&context, geometry).unwrap();
        let upload = |name: &str, len: usize| {
            DeviceBuffer::from_slice(&context, &model.get(name, len).unwrap()).unwrap()
        };
        let hidden_gpu = DeviceBuffer::from_slice(&context, &hidden).unwrap();
        let dense_w = upload("head.dense.weight", width * width);
        let norm_g = upload("head.norm.weight", width);
        let classifier_w = upload("classifier.weight", width);
        let classifier_b = upload("classifier.bias", 1);
        let cu = DeviceBuffer::from_slice(&context, &cumulative(&lengths)).unwrap();
        let scores = DeviceBuffer::<f32>::zeroed(&context, lengths.len()).unwrap();
        let scratch = DeviceBuffer::<f16>::zeroed(&context, lengths.len() * width * 4).unwrap();
        let packed = Packed {
            cu_seqlens: &cu,
            sequences: lengths.len(),
            tokens,
            max_len: 37,
            min_len: 1,
        };
        for (pooling, activation) in HEADS {
            let weights = PredictionHead {
                pooling,
                activation,
                dense_w: &dense_w,
                norm_g: &norm_g,
                classifier_w: &classifier_w,
                classifier_b: &classifier_b,
                eps: shape.eps,
            };
            // SAFETY: every buffer lives until the stream is synchronized below.
            unsafe {
                prediction
                    .prediction_head(
                        &kernels,
                        &stream,
                        &hidden_gpu,
                        &packed,
                        &weights,
                        &scores,
                        &scratch,
                    )
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let got = scores.read().unwrap();
            let head = Head::Prediction(pooling, activation);
            let expected = reference::scores(&shape, &model, head, &rows, &lengths).unwrap();
            for (sequence, (&have, &want)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (f64::from(have) - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "{geometry:?} {head:?} sequence {sequence}: {have} vs {want}"
                );
            }
        }
    }
}

#[test]
fn a_prediction_head_does_not_hide_overflow_behind_its_norm() {
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let kernels = Kernels::load(&context, geometry).unwrap();
        let prediction = PredictionKernels::load(&context, geometry).unwrap();
        let upload = |values: &[f32]| DeviceBuffer::from_slice(&context, values).unwrap();
        let cu = DeviceBuffer::from_slice(&context, &[0, 1, 2]).unwrap();
        let packed = Packed {
            cu_seqlens: &cu,
            sequences: 2,
            tokens: 2,
            max_len: 1,
            min_len: 1,
        };
        let ones = upload(&vec![1.0; width]);
        let classifier_b = upload(&[0.25]);
        let scratch = DeviceBuffer::<f16>::zeroed(&context, 2 * width * 4).unwrap();
        // The first sequence's row under the dense layer's first row. The first case's exact dot
        // is zero, but its first FP32 FMA overflows; the second's activations are finite and their
        // variance overflows.
        for (row, dense_row, overflowing) in [
            ([1e20, 1e20], [1e20, -1e20], "dense sum"),
            ([1e20, -1e20], [1.0, 0.0], "variance"),
        ] {
            let mut hidden = vec![0.0; 2 * width];
            hidden[..2].copy_from_slice(&row);
            let hidden = upload(&hidden);
            let mut dense = vec![0.0; width * width];
            dense[..2].copy_from_slice(&dense_row);
            let dense_w = upload(&dense);
            for (pooling, activation) in HEADS {
                let weights = PredictionHead {
                    pooling,
                    activation,
                    dense_w: &dense_w,
                    norm_g: &ones,
                    classifier_w: &ones,
                    classifier_b: &classifier_b,
                    eps: 1e-5,
                };
                let scores = DeviceBuffer::<f32>::zeroed(&context, 2).unwrap();
                // SAFETY: all buffers outlive the synchronization below.
                unsafe {
                    prediction
                        .prediction_head(
                            &kernels, &stream, &hidden, &packed, &weights, &scores, &scratch,
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
                let got = scores.read().unwrap();
                let case = format!("{geometry:?} {pooling:?} {activation:?}, {overflowing}");
                assert!(got[0].is_nan(), "{case}: {}", got[0]);
                assert_eq!(got[1], 0.25, "{case}: ordinary row");
            }
        }
    }
}

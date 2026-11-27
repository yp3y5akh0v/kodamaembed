use crate::support::{GEOMETRIES, Rng, cumulative, devices, dispatch, quarter};
use kodamaembed_model::{
    Activation, Geometry, Head, Pooling, Positions, Rotary, Shape, Tensors,
    reference::{self, Synthetic},
};
use kodamaembed_vulkan::{Buffer, KernelSpec, Push, shaders};

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
fn pooling_is_the_normalized_mean_of_the_rows_each_pooling_names_on_every_gpu() {
    let lengths = [1, 5, 300];
    let tokens: usize = lengths.iter().sum();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let hidden = Rng::new(21).fill(tokens * width, 4.0);
        for pooling in [Pooling::Mean, Pooling::Cls] {
            // The shader's workgroup width, then whether it pools the first row alone.
            let constants = [quarter(geometry)[0], u32::from(pooling == Pooling::Cls)];
            let spec = KernelSpec {
                spirv: shaders::POOL,
                bindings: 3,
                push_bytes: 0,
                constants: &constants,
                subgroup_size: None,
            };
            for device in devices() {
                let hidden_gpu = Buffer::from_slice(&device, &hidden).unwrap();
                let cu = Buffer::from_slice(&device, &cumulative(&lengths)).unwrap();
                let out = Buffer::<f32>::zeroed(&device, lengths.len() * width).unwrap();
                dispatch(
                    &device,
                    &spec,
                    &[hidden_gpu.binding(), cu.binding(), out.binding()],
                    &Push::new(),
                    [lengths.len() as u32, 1, 1],
                );
                let got = out.read(&device).unwrap();
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
                            (have - want).abs() <= want.abs() * 1e-6 + 1e-7,
                            "{} {geometry:?} {pooling:?}: {have} vs {want}",
                            device.name
                        );
                    }
                    start += len;
                }
            }
        }
    }
}

#[test]
fn pooling_rejects_finite_norm_overflow_and_preserves_zero_means() {
    let lengths = [1, 2, 1, 2];
    let mut failures = Vec::new();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let mut hidden = vec![0.0f32; lengths.iter().sum::<usize>() * width];
        hidden[..3 * width].fill(1e20);
        // The third sequence is zero. The fourth has exactly cancelling large
        // finite rows, so its mean and norm are zero without overflow.
        hidden[4 * width..5 * width].fill(1e20);
        hidden[5 * width..].fill(-1e20);
        assert!(hidden.iter().all(|value| value.is_finite()));
        let constants = quarter(geometry);
        let spec = KernelSpec {
            spirv: shaders::POOL,
            bindings: 3,
            push_bytes: 0,
            constants: &constants,
            subgroup_size: None,
        };
        for device in devices() {
            let hidden_gpu = Buffer::from_slice(&device, &hidden).unwrap();
            let cu = Buffer::from_slice(&device, &cumulative(&lengths)).unwrap();
            let out = Buffer::<f32>::zeroed(&device, lengths.len() * width).unwrap();
            dispatch(
                &device,
                &spec,
                &[hidden_gpu.binding(), cu.binding(), out.binding()],
                &Push::new(),
                [lengths.len() as u32, 1, 1],
            );
            let got = out.read(&device).unwrap();
            // An overflowing norm must give NaN, not a finite zero vector.
            if !got[..2 * width].iter().all(|value| value.is_nan()) {
                failures.push(format!(
                    "{} {geometry:?}: overflowing norm returned {:?}",
                    device.name,
                    &got[..4]
                ));
            }
            assert!(
                got[2 * width..].iter().all(|&value| value == 0.0),
                "{} {geometry:?}: zero means changed",
                device.name
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

#[test]
fn the_reranking_head_scores_each_sequence_from_its_first_row_on_every_gpu() {
    let lengths = [4, 1, 37];
    let tokens: usize = lengths.iter().sum();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let mut rng = Rng::new(43);
        let hidden = rng.fill(tokens * width, 2.0);
        let pooler_w = rng.fill(width * width, 0.05);
        let pooler_b = rng.fill(width, 0.1);
        let classifier_w = rng.fill(width, 0.2);
        let classifier_b = rng.fill(1, 0.5);
        let constants = [width as u32];
        let spec = KernelSpec {
            spirv: shaders::RERANK,
            bindings: 7,
            push_bytes: 0,
            constants: &constants,
            subgroup_size: None,
        };
        for device in devices() {
            let upload = |values: &[f32]| Buffer::from_slice(&device, values).unwrap();
            let cu = Buffer::from_slice(&device, &cumulative(&lengths)).unwrap();
            let buffers = [
                upload(&hidden),
                upload(&pooler_w),
                upload(&pooler_b),
                upload(&classifier_w),
                upload(&classifier_b),
            ];
            let scores = Buffer::<f32>::zeroed(&device, lengths.len()).unwrap();
            dispatch(
                &device,
                &spec,
                &[
                    buffers[0].binding(),
                    cu.binding(),
                    buffers[1].binding(),
                    buffers[2].binding(),
                    buffers[3].binding(),
                    buffers[4].binding(),
                    scores.binding(),
                ],
                &Push::new(),
                [lengths.len() as u32, 1, 1],
            );
            let got = scores.read(&device).unwrap();
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
                    "{} {geometry:?} sequence {sequence}: {have} vs {want}",
                    device.name
                );
                start += len;
            }
        }
    }
}

#[test]
fn the_prediction_head_scores_each_sequence_from_its_pooled_rows_on_every_gpu() {
    let lengths = [4usize, 1, 37];
    let tokens: usize = lengths.iter().sum();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let shape = head_shape(geometry);
        let model = Synthetic::new(&shape, 47);
        let hidden = Rng::new(53).fill(tokens * width, 2.0);
        let rows: Vec<Vec<f64>> = hidden
            .chunks(width)
            .map(|row| row.iter().map(|&x| f64::from(x)).collect())
            .collect();
        let tensor = |name: &str, len: usize| model.get(name, len).unwrap();
        let weights = [
            tensor("head.dense.weight", width * width),
            tensor("head.norm.weight", width),
            tensor("classifier.weight", width),
            tensor("classifier.bias", 1),
        ];
        for pooling in [Pooling::Mean, Pooling::Cls] {
            for activation in [Activation::Gelu, Activation::Silu] {
                let head = Head::Prediction(pooling, activation);
                let expected = reference::scores(&shape, &model, head, &rows, &lengths).unwrap();
                // The width, then whether the rows' mean is pooled and whether SiLU activates.
                let constants = [
                    width as u32,
                    u32::from(pooling == Pooling::Mean),
                    u32::from(activation == Activation::Silu),
                ];
                let spec = KernelSpec {
                    spirv: shaders::RERANK_PREDICTION,
                    bindings: 7,
                    push_bytes: 4,
                    constants: &constants,
                    subgroup_size: None,
                };
                for device in devices() {
                    let upload = |values: &[f32]| Buffer::from_slice(&device, values).unwrap();
                    let cu = Buffer::from_slice(&device, &cumulative(&lengths)).unwrap();
                    let buffers = [
                        upload(&hidden),
                        upload(&weights[0]),
                        upload(&weights[1]),
                        upload(&weights[2]),
                        upload(&weights[3]),
                    ];
                    let scores = Buffer::<f32>::zeroed(&device, lengths.len()).unwrap();
                    dispatch(
                        &device,
                        &spec,
                        &[
                            buffers[0].binding(),
                            cu.binding(),
                            buffers[1].binding(),
                            buffers[2].binding(),
                            buffers[3].binding(),
                            buffers[4].binding(),
                            scores.binding(),
                        ],
                        &Push::new().f32(shape.eps),
                        [lengths.len() as u32, 1, 1],
                    );
                    let got = scores.read(&device).unwrap();
                    for (sequence, (&have, &want)) in got.iter().zip(&expected).enumerate() {
                        assert!(
                            (f64::from(have) - want).abs() <= 1e-4 * (1.0 + want.abs()),
                            "{} {geometry:?} {head:?} sequence {sequence}: {have} vs {want}",
                            device.name
                        );
                    }
                }
            }
        }
    }
}

use crate::support::{GEOMETRIES, Rng, devices, dispatch, quarter};
use half::f16;
use kodamaembed_vulkan::{Buffer, KernelSpec, Push, shaders};

fn reference(x: &[f32], width: usize, gamma: &[f32], beta: &[f32], eps: f64) -> Vec<f64> {
    x.chunks(width)
        .flat_map(|row| {
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64;
            let variance = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / width as f64;
            let rstd = 1.0 / (variance + eps).sqrt();
            row.iter()
                .zip(gamma.iter().zip(beta))
                .map(move |(&v, (&g, &b))| {
                    (f64::from(v) - mean) * rstd * f64::from(g) + f64::from(b)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn embedding_sums_word_type_and_position_rows_before_normalizing_on_every_gpu() {
    let (vocab, max_positions, type_vocab) = (100, 512, 2);
    let lengths = [3usize, 70];
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let mut rng = Rng::new(31);
        let word = rng.fill(vocab * width, 0.1);
        let position = rng.fill(max_positions * width, 0.05);
        let token_type = rng.fill(type_vocab * width, 0.02);
        let gamma: Vec<f32> = (0..width).map(|_| 1.0 + rng.symmetric() * 0.5).collect();
        let beta = rng.fill(width, 0.2);
        let ids: Vec<i32> = (0..73).map(|_| rng.below(vocab as u64) as i32).collect();
        let types: Vec<i32> = (0..73)
            .map(|_| rng.below(type_vocab as u64) as i32)
            .collect();
        let positions: Vec<i32> = lengths.iter().flat_map(|&len| 0..len as i32).collect();
        let tokens = ids.len();
        let summed: Vec<f32> = ids
            .iter()
            .zip(&types)
            .zip(&positions)
            .flat_map(|((&id, &kind), &pos)| {
                let (w, t, p) = (
                    id as usize * width,
                    kind as usize * width,
                    pos as usize * width,
                );
                let (word, token_type, position) = (&word, &token_type, &position);
                (0..width).map(move |d| (word[w + d] + token_type[t + d]) + position[p + d])
            })
            .collect();
        let expected = reference(&summed, width, &gamma, &beta, 1e-12);
        let constants = quarter(geometry);
        let spec = KernelSpec {
            spirv: shaders::EMBED,
            bindings: 10,
            push_bytes: 8,
            constants: &constants,
            subgroup_size: None,
        };
        for device in devices() {
            let upload = |values: &[f32]| Buffer::from_slice(&device, values).unwrap();
            let tables = [
                upload(&word),
                upload(&position),
                upload(&token_type),
                upload(&gamma),
                upload(&beta),
            ];
            let ids_gpu = Buffer::from_slice(&device, &ids).unwrap();
            let types_gpu = Buffer::from_slice(&device, &types).unwrap();
            let positions_gpu = Buffer::from_slice(&device, &positions).unwrap();
            let out16 = Buffer::<f16>::zeroed(&device, tokens * width).unwrap();
            let out32 = Buffer::<f32>::zeroed(&device, tokens * width).unwrap();
            dispatch(
                &device,
                &spec,
                &[
                    ids_gpu.binding(),
                    types_gpu.binding(),
                    positions_gpu.binding(),
                    tables[0].binding(),
                    tables[1].binding(),
                    tables[2].binding(),
                    tables[3].binding(),
                    tables[4].binding(),
                    out16.binding(),
                    out32.binding(),
                ],
                &Push::new().u32(tokens as u32).f32(1e-12),
                [tokens as u32, 1, 1],
            );
            for (&s, &r) in out32.read(&device).unwrap().iter().zip(&expected) {
                assert!(
                    (f64::from(s) - r).abs() <= 1e-5 * (1.0 + r.abs()),
                    "{} {geometry:?}: {s} vs {r}",
                    device.name
                );
            }
        }
    }
}

#[test]
fn layernorm_matches_a_double_precision_reference_on_every_gpu() {
    let rows = 37;
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let mut rng = Rng::new(7);
        let mut x = rng.fill(rows * width, 3.0);
        for (index, value) in x.iter_mut().enumerate() {
            *value += 0.5;
            if index % width == 308 {
                *value *= 40.0;
            }
        }
        let gamma: Vec<f32> = (0..width).map(|_| 1.0 + rng.symmetric() * 0.5).collect();
        let beta = rng.fill(width, 0.2);
        let expected = reference(&x, width, &gamma, &beta, 1e-12);
        let constants = quarter(geometry);
        let spec = KernelSpec {
            spirv: shaders::LAYERNORM,
            bindings: 5,
            push_bytes: 12,
            constants: &constants,
            subgroup_size: None,
        };
        for device in devices() {
            let x_gpu = Buffer::from_slice(&device, &x).unwrap();
            let gamma_gpu = Buffer::from_slice(&device, &gamma).unwrap();
            let beta_gpu = Buffer::from_slice(&device, &beta).unwrap();
            let out16 = Buffer::<f16>::zeroed(&device, rows * width).unwrap();
            let out32 = Buffer::<f32>::zeroed(&device, rows * width).unwrap();
            dispatch(
                &device,
                &spec,
                &[
                    x_gpu.binding(),
                    gamma_gpu.binding(),
                    beta_gpu.binding(),
                    out16.binding(),
                    out32.binding(),
                ],
                &Push::new().u32(rows as u32).u32(1).f32(1e-12),
                [rows as u32, 1, 1],
            );
            let (half_values, single) =
                (out16.read(&device).unwrap(), out32.read(&device).unwrap());
            for ((&h, &s), &r) in half_values.iter().zip(&single).zip(&expected) {
                assert!(
                    (f64::from(s) - r).abs() <= 1e-5 * (1.0 + r.abs()),
                    "{} {geometry:?}: fp32 {s} vs {r}",
                    device.name
                );
                let bound = r.abs() * f64::powi(2.0, -11) + 1e-5 * (1.0 + r.abs());
                assert!(
                    (f64::from(h.to_f32()) - r).abs() <= bound,
                    "{} {geometry:?}: fp16 {h} vs {r}",
                    device.name
                );
            }
        }
    }
}

#[test]
fn normalization_overflow_is_visible_in_both_output_precisions() {
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        for (amplitude, eps) in [(1e20_f32, 1e-12_f32), (1e16, f32::MAX)] {
            // The first row overflows; the second stays an ordinary zero row.
            let mut x = vec![0.0; 2 * width];
            for (index, value) in x[..width].iter_mut().enumerate() {
                *value = if index % 2 == 0 {
                    amplitude
                } else {
                    -amplitude
                };
            }
            for device in devices() {
                let x_gpu = Buffer::from_slice(&device, &x).unwrap();
                let gamma = Buffer::from_slice(&device, &vec![1.0f32; width]).unwrap();
                let zero = Buffer::from_slice(&device, &vec![0.0f32; width]).unwrap();
                let ids = Buffer::from_slice(&device, &[0i32, 1]).unwrap();
                let indexes = Buffer::from_slice(&device, &[0i32, 0]).unwrap();
                let constants = quarter(geometry);
                for embedding in [false, true] {
                    let out16 = Buffer::<f16>::zeroed(&device, 2 * width).unwrap();
                    let out32 = Buffer::<f32>::zeroed(&device, 2 * width).unwrap();
                    let spec = KernelSpec {
                        spirv: if embedding {
                            shaders::EMBED
                        } else {
                            shaders::LAYERNORM
                        },
                        bindings: if embedding { 10 } else { 5 },
                        push_bytes: if embedding { 8 } else { 12 },
                        constants: &constants,
                        subgroup_size: None,
                    };
                    let (bindings, push) = if embedding {
                        (
                            vec![
                                ids.binding(),
                                indexes.binding(),
                                indexes.binding(),
                                x_gpu.binding(),
                                zero.binding(),
                                zero.binding(),
                                gamma.binding(),
                                zero.binding(),
                                out16.binding(),
                                out32.binding(),
                            ],
                            Push::new().u32(2).f32(eps),
                        )
                    } else {
                        (
                            vec![
                                x_gpu.binding(),
                                gamma.binding(),
                                zero.binding(),
                                out16.binding(),
                                out32.binding(),
                            ],
                            Push::new().u32(2).u32(1).f32(eps),
                        )
                    };
                    dispatch(&device, &spec, &bindings, &push, [2, 1, 1]);
                    let half = out16.read(&device).unwrap();
                    let single = out32.read(&device).unwrap();
                    assert!(
                        half[..width].iter().all(|value| value.is_nan()),
                        "{} {geometry:?} embedding={embedding} eps={eps}: FP16 masked overflow",
                        device.name
                    );
                    assert!(
                        single[..width].iter().all(|value| value.is_nan()),
                        "{} {geometry:?} embedding={embedding} eps={eps}: FP32 masked overflow",
                        device.name
                    );
                    assert!(half[width..].iter().all(|value| value.to_f32() == 0.0));
                    assert!(single[width..].iter().all(|&value| value == 0.0));
                }
            }
        }
    }
}

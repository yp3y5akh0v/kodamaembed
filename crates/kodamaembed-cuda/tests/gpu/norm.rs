use crate::support::{GEOMETRIES, Rng};
use half::f16;
use kodamaembed_cuda::{
    Context, DeviceBuffer, Stream,
    kernels::{Embeddings, Kernels, Tokens},
};

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

/// Rows with a mean offset and one large outlier column, like BERT's residual stream.
fn inputs(rows: usize, width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(7);
    let mut x = rng.fill(rows * width, 3.0);
    for (index, value) in x.iter_mut().enumerate() {
        *value += 0.5;
        if index % width == 308 {
            *value *= 40.0;
        }
    }
    let gamma = (0..width).map(|_| 1.0 + rng.symmetric() * 0.5).collect();
    let beta = rng.fill(width, 0.2);
    (x, gamma, beta)
}

#[test]
fn layernorm_matches_a_double_precision_reference() {
    let rows = 37;
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let (x, gamma, beta) = inputs(rows, width);
        let expected = reference(&x, width, &gamma, &beta, 1e-12);
        let kernels = Kernels::load(&context, geometry).unwrap();
        let x_gpu = DeviceBuffer::from_slice(&context, &x).unwrap();
        let gamma_gpu = DeviceBuffer::from_slice(&context, &gamma).unwrap();
        let beta_gpu = DeviceBuffer::from_slice(&context, &beta).unwrap();
        let out16 = DeviceBuffer::<f16>::zeroed(&context, rows * width).unwrap();
        let out32 = DeviceBuffer::<f32>::zeroed(&context, rows * width).unwrap();
        // SAFETY: every buffer lives until the stream is synchronized below.
        unsafe {
            kernels
                .layernorm(
                    &stream,
                    &x_gpu,
                    &gamma_gpu,
                    &beta_gpu,
                    &out16,
                    Some(&out32),
                    rows,
                    1e-12,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        let (half_values, single) = (out16.read().unwrap(), out32.read().unwrap());
        for ((&h, &s), &r) in half_values.iter().zip(&single).zip(&expected) {
            assert!(
                (f64::from(s) - r).abs() <= 1e-5 * (1.0 + r.abs()),
                "{geometry:?} fp32 {s} vs {r}"
            );
            let bound = r.abs() * f64::powi(2.0, -11) + 1e-5 * (1.0 + r.abs());
            assert!(
                (f64::from(h.to_f32()) - r).abs() <= bound,
                "{geometry:?} fp16 {h} vs {r}"
            );
        }
    }
}

#[test]
fn embedding_sums_word_type_and_position_rows_before_normalizing() {
    let (vocab, max_positions, type_vocab) = (100, 512, 2);
    let lengths = [3usize, 70];
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
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
        let count = ids.len();

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

        let kernels = Kernels::load(&context, geometry).unwrap();
        let upload = |values: &[f32]| DeviceBuffer::from_slice(&context, values).unwrap();
        let (word_gpu, position_gpu, type_gpu) =
            (upload(&word), upload(&position), upload(&token_type));
        let (gamma_gpu, beta_gpu) = (upload(&gamma), upload(&beta));
        let embeddings = Embeddings {
            word: &word_gpu,
            position: &position_gpu,
            token_type: &type_gpu,
            gamma: &gamma_gpu,
            beta: &beta_gpu,
        };
        let ids_gpu = DeviceBuffer::from_slice(&context, &ids).unwrap();
        let types_gpu = DeviceBuffer::from_slice(&context, &types).unwrap();
        let positions_gpu = DeviceBuffer::from_slice(&context, &positions).unwrap();
        let tokens = Tokens {
            ids: &ids_gpu,
            types: &types_gpu,
            positions: &positions_gpu,
            count,
        };
        let out16 = DeviceBuffer::<f16>::zeroed(&context, count * width).unwrap();
        let out32 = DeviceBuffer::<f32>::zeroed(&context, count * width).unwrap();
        // SAFETY: every buffer lives until the stream is synchronized below.
        unsafe {
            kernels
                .embed_layernorm(&stream, &embeddings, &tokens, &out16, &out32, 1e-12)
                .unwrap();
        }
        stream.synchronize().unwrap();
        for (&s, &r) in out32.read().unwrap().iter().zip(&expected) {
            assert!(
                (f64::from(s) - r).abs() <= 1e-5 * (1.0 + r.abs()),
                "{geometry:?} {s} vs {r}"
            );
        }
    }
}

#[test]
fn layernorm_without_an_fp32_output_writes_only_fp16() {
    let rows = 8;
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let (x, gamma, beta) = inputs(rows, width);
        let kernels = Kernels::load(&context, geometry).unwrap();
        let x_gpu = DeviceBuffer::from_slice(&context, &x).unwrap();
        let gamma_gpu = DeviceBuffer::from_slice(&context, &gamma).unwrap();
        let beta_gpu = DeviceBuffer::from_slice(&context, &beta).unwrap();
        let out16 = DeviceBuffer::<f16>::zeroed(&context, rows * width).unwrap();
        // SAFETY: every buffer lives until the stream is synchronized below.
        unsafe {
            kernels
                .layernorm(
                    &stream, &x_gpu, &gamma_gpu, &beta_gpu, &out16, None, rows, 1e-12,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        let expected = reference(&x, width, &gamma, &beta, 1e-12);
        let got = out16.read().unwrap();
        assert!(
            got.iter()
                .zip(&expected)
                .all(|(h, &r)| (f64::from(h.to_f32()) - r).abs() <= r.abs() * 1e-3 + 1e-4),
            "{geometry:?}"
        );
    }
}

#[test]
fn layernorm_does_not_turn_overflowing_statistics_into_finite_zeros() {
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let kernels = Kernels::load(&context, geometry).unwrap();
        let gamma = DeviceBuffer::from_slice(&context, &vec![1.0; width]).unwrap();
        let beta = DeviceBuffer::from_slice(&context, &vec![0.0; width]).unwrap();
        // The variance overflows, then variance plus epsilon does: rsqrt(Inf)
        // == 0 must not turn either row into finite zeros.
        for (magnitude, eps, uniform) in [
            (1e20, 1e-12, false),
            (1e17, f32::MAX, false),
            (f32::MAX, 1e-12, true),
        ] {
            let x: Vec<f32> = (0..width)
                .map(|column| {
                    if uniform || column % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                })
                .collect();
            assert!(x.iter().all(|value| value.is_finite()));
            let x = DeviceBuffer::from_slice(&context, &x).unwrap();
            let out16 = DeviceBuffer::<f16>::zeroed(&context, width).unwrap();
            let out32 = DeviceBuffer::<f32>::zeroed(&context, width).unwrap();
            for single in [Some(&out32), None] {
                // SAFETY: all buffers outlive the synchronization below.
                unsafe {
                    kernels
                        .layernorm(&stream, &x, &gamma, &beta, &out16, single, 1, eps)
                        .unwrap();
                }
                stream.synchronize().unwrap();
                assert!(
                    out16.read().unwrap().iter().all(|value| value.is_nan()),
                    "{geometry:?}, magnitude {magnitude}, eps {eps}: FP16"
                );
                if single.is_some() {
                    assert!(
                        out32.read().unwrap().iter().all(|value| value.is_nan()),
                        "{geometry:?}, magnitude {magnitude}, eps {eps}: FP32"
                    );
                }
            }
        }
    }
}

#[test]
fn embedding_layernorm_propagates_overflow_without_contaminating_other_rows() {
    let context = Context::new(0).unwrap();
    let stream = Stream::new(&context).unwrap();
    for geometry in GEOMETRIES {
        let width = geometry.hidden();
        let kernels = Kernels::load(&context, geometry).unwrap();
        let upload = |values: &[f32]| DeviceBuffer::from_slice(&context, values).unwrap();
        let gamma = upload(&vec![1.0; width]);
        let beta = upload(&vec![0.0; width]);
        let position = upload(&vec![0.0; width]);
        let token_type = upload(&vec![0.0; width]);
        let ids = DeviceBuffer::from_slice(&context, &[0, 1]).unwrap();
        let types = DeviceBuffer::from_slice(&context, &[0, 0]).unwrap();
        let positions = DeviceBuffer::from_slice(&context, &[0, 0]).unwrap();
        let tokens = Tokens {
            ids: &ids,
            types: &types,
            positions: &positions,
            count: 2,
        };
        for (magnitude, eps, uniform) in [
            (1e20, 1e-12, false),
            (1e17, f32::MAX, false),
            (f32::MAX, 1e-12, true),
        ] {
            let word: Vec<f32> = (0..2 * width)
                .map(|column| {
                    let value = if column < width { magnitude } else { 1.0 };
                    if (uniform && column < width) || column % 2 == 0 {
                        value
                    } else {
                        -value
                    }
                })
                .collect();
            let word = upload(&word);
            let embeddings = Embeddings {
                word: &word,
                position: &position,
                token_type: &token_type,
                gamma: &gamma,
                beta: &beta,
            };
            let out16 = DeviceBuffer::<f16>::zeroed(&context, 2 * width).unwrap();
            let out32 = DeviceBuffer::<f32>::zeroed(&context, 2 * width).unwrap();
            // SAFETY: ids select the two uploaded rows; all buffers outlive the work.
            unsafe {
                kernels
                    .embed_layernorm(&stream, &embeddings, &tokens, &out16, &out32, eps)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let half = out16.read().unwrap();
            let single = out32.read().unwrap();
            assert!(
                half[..width].iter().all(|value| value.is_nan()),
                "{geometry:?}"
            );
            assert!(
                single[..width].iter().all(|value| value.is_nan()),
                "{geometry:?}"
            );
            assert!(
                half[width..].iter().all(|value| value.is_finite()),
                "{geometry:?}"
            );
            assert!(
                single[width..].iter().all(|value| value.is_finite()),
                "{geometry:?}"
            );
            if eps == 1e-12 {
                for (column, (&single, &half)) in
                    single[width..].iter().zip(&half[width..]).enumerate()
                {
                    let expected = if column % 2 == 0 { 1.0 } else { -1.0 };
                    assert_eq!(single, expected, "{geometry:?}: ordinary FP32 row");
                    assert_eq!(half.to_f32(), expected, "{geometry:?}: ordinary FP16 row");
                }
            }
        }
    }
}

use super::*;

/// The scalar recipe the SIMD quantizer must reproduce, a row's second digits behind its first.
fn quantized_scalar(values: &[f32], k: usize, digits: usize) -> Result<(Vec<i8>, Vec<f32>), Error> {
    let in_range = |q: f32| (-127.0..=127.0).contains(&q);
    let (mut quantized, mut steps) = (Vec::new(), Vec::new());
    for row in values.chunks_exact(k) {
        let firsts = row_scales(row, HALF)?;
        let mut behind = Vec::new();
        for (half, &step) in row.as_chunks::<HALF>().0.iter().zip(&firsts) {
            for &x in half {
                let q = crate::math::on_step(x, step).round_ties_even();
                if !in_range(q) {
                    return Err(Error::FloatRange);
                }
                quantized.push(q as i8);
                if digits == 2 {
                    // What the first digit leaves, in one FMA, on the step over 254.
                    let left = (-q).mul_add(step, x);
                    let rest = crate::math::on_step(left, step / 254.0).round_ties_even();
                    if !in_range(rest) {
                        return Err(Error::FloatRange);
                    }
                    behind.push(rest as i8);
                }
            }
        }
        quantized.extend(behind);
        steps.extend(&firsts);
        if digits == 2 {
            steps.extend(firsts.iter().map(|&step| step / 254.0));
        }
    }
    Ok((quantized, steps))
}

#[test]
fn e5_depths_scalar_simd_match_for_band_tails_and_finite_half_extremes() {
    if !is_supported() {
        return;
    }
    for k in [1024, 4096] {
        let signs: Vec<f32> = (0..k)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        for m in [95, 96, 97] {
            let raw: Vec<f32> = (0..m * k)
                .map(|at| {
                    let r = at / k;
                    let t = at % k;
                    match r % 4 {
                        0 => {
                            if t % 2 == 0 {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        1 => {
                            f16::from_bits((t % 1023 + 1) as u16).to_f32()
                                * if t % 2 == 0 { 1.0 } else { -1.0 }
                        }
                        2 => 65504.0 * if t % 2 == 0 { 1.0 } else { -1.0 },
                        _ => ((t * 17 + r * 11) % 257) as f32 / 64.0 - 2.0,
                    }
                })
                .collect();
            let rotated = rotated_rows_scalar(&raw, m, k, &signs).unwrap();
            let (expected, steps) = quantized_scalar(&rotated, k, 1).unwrap();
            let (actual, actual_steps) =
                avx2::quantized_blocks(&raw, m, k, &signs, None, 1).unwrap();
            assert_eq!(actual, expected, "k={k},m={m}");
            same_bits(&actual_steps, &steps, "E5 SIMD steps");
        }
        for bad in [65520.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut raw = vec![0.0; k];
            raw[k - 1] = bad;
            assert!(rotated_rows_scalar(&raw, 1, k, &signs).is_err());
            for digits in [1, 2] {
                assert!(avx2::quantized_blocks(&raw, 1, k, &signs, None, digits).is_err());
            }
        }
    }
}

#[test]
fn simd_preparation_matches_scalar_for_half_boundaries_and_signed_zero() {
    if !avx2::is_supported() {
        return;
    }
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    for k in [128, 384, 1536] {
        for case in 0..24 {
            let m = 7;
            let a: Vec<_> = (0..m * k)
                .map(|t| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    match case {
                        0 => {
                            if t % 2 == 0 {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        1 => {
                            if t % 2 == 0 {
                                65504.0
                            } else {
                                -65504.0
                            }
                        }
                        2 => f16::from_bits(1 + (state % 1023) as u16).to_f32(),
                        3 => {
                            let lo = f16::from_bits(0x0400 + (state % 0x73fe) as u16).to_f32();
                            let hi = f16::from_bits(f16::from_f32(lo).to_bits() + 1).to_f32();
                            (lo + hi) * 0.5
                        }
                        _ => ((state & 0xffff) as i32 - 32768) as f32 / 1024.0,
                    }
                })
                .collect();
            let signs: Vec<_> = (0..k)
                .map(|t| if t % 3 == 0 { -1.0 } else { 1.0 })
                .collect();
            let rotated = rotated_rows_scalar(&a, m, k, &signs).unwrap();
            for digits in [1, 2] {
                let (expected, expected_steps) = quantized_scalar(&rotated, k, digits).unwrap();
                let (actual, actual_steps) =
                    avx2::quantized_blocks(&a, m, k, &signs, None, digits).unwrap();
                assert_eq!(actual, expected, "k={k},case={case},digits={digits}");
                same_bits(&actual_steps, &expected_steps, "SIMD steps");
            }
        }
    }
}

#[test]
fn invalid_weights_and_activations_reject() {
    let signs = [1.0; 128];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 65520.0] {
        let mut values = [0.0; 128];
        values[3] = bad;
        assert!(matches!(
            rotated_weights(&values, 1, 128, &signs),
            Err(Error::FloatRange)
        ));
        if avx2::is_supported() {
            assert_eq!(
                avx2::quantized_blocks(&values, 1, 128, &signs, None, 1),
                Err(Error::FloatRange)
            );
        }
    }
    if avx2::is_supported() {
        // The quantizer checks what its kernel reads: whole blocks, signs, rows, one or two digits.
        let zeros = [0.0; 257];
        for (rows, k, values, depth, digits) in [
            (1, 0, 0, 0, 1),
            (1, 127, 127, 127, 1),
            (1, 128, 128, 127, 1),
            (2, 128, 255, 128, 1),
            (2, 128, 257, 128, 1),
            (2, 128, 256, 128, 0),
            (2, 128, 256, 128, 3),
        ] {
            assert_eq!(
                avx2::quantized_blocks(&zeros[..values], rows, k, &signs[..depth], None, digits),
                Err(Error::Dimensions)
            );
        }
    }
    assert!(matches!(
        rotated_weights(&[0.0; 127], 1, 127, &signs),
        Err(Error::Dimensions)
    ));
    let mut invalid = signs;
    invalid[0] = 0.0;
    assert!(matches!(
        rotated_weights(&[0.0; 128], 1, 128, &invalid),
        Err(Error::FloatRange)
    ));
}

fn same_bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label} index={i}: actual={a}, expected={b}"
        );
    }
}

#[test]
fn half_rounding_is_correct_at_every_fp16_boundary() {
    let up = |x: f64| f64::from_bits(x.to_bits() + 1);
    let down = |x: f64| f64::from_bits(x.to_bits() - 1);
    for bits in 0_u16..0x7bff {
        let (low, high) = (
            f16::from_bits(bits).to_f64(),
            f16::from_bits(bits + 1).to_f64(),
        );
        let middle = (low + high) / 2.0;
        let even = if bits % 2 == 0 { low } else { high };
        for sign in [1.0, -1.0] {
            let checks = [
                (sign * low, sign * low),
                (sign * middle, sign * even),
                (sign * up(middle), sign * high),
                (sign * down(middle), sign * low),
            ];
            for (value, expected) in checks {
                assert_eq!(
                    half_of(value).to_bits(),
                    (expected as f32).to_bits(),
                    "bits={bits:#06x}, value={value:e}"
                );
            }
        }
    }
    assert_eq!(half_of(65520.0), f32::INFINITY);
    assert_eq!(half_of(-65520.0), f32::NEG_INFINITY);
    assert_eq!(half_of(down(65520.0)), 65504.0);
    assert_eq!(half_of(1e300), f32::INFINITY);
    assert_eq!(half_of(1e-300).to_bits(), 0.0_f32.to_bits());
    assert!(half_of(f64::NAN).is_nan());
}

#[test]
fn static_centers_round_once_and_take_exact_products() {
    let (n, k) = (37, 1024);
    let raw: Vec<f32> = (0..n * k)
        .map(|at| ((at * 41 % 251) as f32 - 125.0) / 128.0)
        .collect();
    let means: Vec<f64> = (0..3 * k)
        .map(|at| ((at * 31 % 211) as f64 - 105.0) / 9.0 + 1e-3)
        .collect();
    let statics = StaticCenters::exact(&raw, n, k, &means).unwrap();
    for (center, &mean) in statics.centers.iter().zip(&means) {
        assert_eq!(center.to_bits(), half_of(mean).to_bits());
    }
    // Each shared value: f64 sums of exact products, rounded once.
    for class in 0..3 {
        for j in 0..n {
            let mut sum = 0.0_f64;
            for t in 0..k {
                sum += f64::from(raw[j * k + t]) * f64::from(statics.centers[class * k + t]);
            }
            assert_eq!(
                statics.shared[class * n + j].to_bits(),
                (sum as f32).to_bits(),
                "class {class} column {j}"
            );
        }
    }
    for threads in [1, 3, 24] {
        crate::worker_pool("static-test", threads)
            .unwrap()
            .install(|| {
                let again = StaticCenters::exact(&raw, n, k, &means).unwrap();
                same_bits(
                    &again.shared,
                    &statics.shared,
                    &format!("threads {threads}"),
                );
            });
    }
    assert!(matches!(
        StaticCenters::exact(&raw, n, k, &means[..k + 1]),
        Err(Error::Dimensions)
    ));
    assert!(matches!(
        StaticCenters::exact(&raw[..k], n, k, &means),
        Err(Error::Dimensions)
    ));
    assert!(matches!(
        StaticCenters::exact(&raw, n, k, &vec![1e9; k]),
        Err(Error::FloatRange)
    ));
}

fn centered_rows(raw: &[f32], k: usize, means: &[f32], seq: &[u32], signs: &[f32]) -> Vec<f32> {
    gained_rows(raw, k, &vec![1.0; k], means, seq, signs)
}

fn gained_rows(
    raw: &[f32],
    k: usize,
    gains: &[f32],
    means: &[f32],
    seq: &[u32],
    signs: &[f32],
) -> Vec<f32> {
    let mut rows = vec![0.0; raw.len()];
    for (r, row) in rows.chunks_exact_mut(k).enumerate() {
        let mean = &means[seq[r] as usize * k..][..k];
        for t in 0..k {
            row[t] = (f16::from_f32(raw[r * k + t]).to_f32() * gains[t] - mean[t]) * signs[t];
        }
        hadamard(row);
    }
    rows
}

#[test]
fn centering_rejects_classes_outside_the_centers() {
    if !avx2::is_supported() {
        return;
    }
    let (m, k) = (9, 384);
    let signs: Vec<f32> = (0..k)
        .map(|t| if t % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let raw: Vec<f32> = (0..m * k)
        .map(|at| ((at * 17 % 257) as f32 - 128.0) / 61.0)
        .collect();
    let means: Vec<f32> = (0..3 * k)
        .map(|at| f16::from_f32(((at * 31 % 89) as f32 - 44.0) / 23.0).to_f32())
        .collect();
    let outside = vec![3_u32; m];
    let centering = avx2::Centering {
        means: &means,
        seq: &outside,
        gains: None,
    };
    assert_eq!(
        avx2::quantized_blocks(&raw, m, k, &signs, Some(centering), 1),
        Err(Error::Dimensions)
    );
}

fn calibrated(
    raw: &[f32],
    n: usize,
    k: usize,
    signs: &[f32],
    means: &[f64],
    norm: Option<DeferredNorm<'_>>,
) -> PreparedWeight {
    let m = 96;
    let rows: Vec<f32> = (0..m * k)
        .map(|at| ((at * 29 % 131) as f32 - 40.0) / 97.0)
        .collect();
    let classes: Vec<u32> = (0..m as u32).map(|r| r % 3).collect();
    let operand = Operand {
        rows: &rows,
        classes: &classes,
        means,
        weights: None,
    };
    PreparedWeight::calibrated(raw, n, k, signs, operand, norm, 1).unwrap()
}

/// The same tables read for an operand of two digits, as a calibration cache installs them.
fn in_two_digits(one: &PreparedWeight) -> PreparedWeight {
    let mut tables = Vec::new();
    one.write_tables(&mut tables).unwrap();
    let gains = one.deferred.as_ref().map(|deferred| &deferred.gains[..]);
    let mut left = &tables[..];
    let two =
        PreparedWeight::from_tables(one.n(), one.k(), &one.signs, gains, &mut left, 2).unwrap();
    assert!(left.is_empty(), "every table is read");
    two
}

/// Each row's weight digits once for each of the operand's digits, as the dot walks them.
fn walked(prepared: &PreparedWeight) -> Vec<i8> {
    let depth = prepared.digits * prepared.k();
    prepared
        .q_by_row()
        .unwrap()
        .chunks_exact(prepared.k())
        .flat_map(|row| row.iter().cycle().take(depth).copied())
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[test]
fn column_groups_and_whole_bands_compute_the_same_bits() {
    use crate::gemm::Epilogue;
    use kodamaembed_model::repro::restored_class;
    if !is_supported() {
        return;
    }
    // 65 panels, the last short; the thread counts cover bands and every split.
    let (n, k) = (1032, 1024);
    let signs: Vec<f32> = (0..k)
        .map(|t| if t % 5 == 0 { -1.0 } else { 1.0 })
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|at| ((at * 37 % 251) as f32 - 125.0) / 128.0)
        .collect();
    let means: Vec<f64> = (0..3 * k)
        .map(|at| ((at * 7 % 41) as f64 - 20.0) / 9.0)
        .collect();
    let restored = calibrated(&w, n, k, &signs, &means, None);
    let bias: Vec<f32> = (0..n).map(|j| (j % 9) as f32 / 8.0 - 0.5).collect();
    for lengths in [&[3, 13][..], &[60, 37], &[100, 150, 38]] {
        let m: usize = lengths.iter().sum();
        let a: Vec<f32> = (0..m * k)
            .map(|at| ((at * 23 % 241) as f32 - 120.0) / 32.0 + 0.25)
            .collect();
        let classes: Vec<u32> = lengths
            .iter()
            .flat_map(|&len| (0..len).map(move |t| restored_class(t, len) as u32))
            .collect();
        let residual: Vec<f32> = (0..m * n)
            .map(|at| ((at % 17) as f32 - 8.0) / 16.0)
            .collect();
        let outputs = |threads: usize| {
            crate::worker_pool("split-test", threads)
                .unwrap()
                .install(|| {
                    let mut all = Vec::new();
                    let mut out = vec![f32::NAN; m * n];
                    for epi in [
                        Epilogue::Bias(&bias),
                        Epilogue::BiasGelu(&bias),
                        Epilogue::BiasResidual(&bias, &residual),
                    ] {
                        for segment in [k, 256] {
                            restored
                                .project_restored_into(&a, m, &classes, segment, epi, &mut out)
                                .unwrap();
                            all.push(out.clone());
                        }
                    }
                    all
                })
        };
        let whole = outputs(1);
        for threads in [3, 24] {
            for (route, (split, whole)) in outputs(threads).iter().zip(&whole).enumerate() {
                same_bits(
                    split,
                    whole,
                    &format!("m={m} threads={threads} route={route}"),
                );
            }
        }
    }
}

#[test]
fn transform_matches_independent_integer_hadamard_matrix() {
    // Integer sums are exact in FP32, so every H128 block's signs check exactly.
    let source: Vec<_> = (0..384).map(|j| ((j * 7 % 31) - 15) as f32).collect();
    let mut transformed = source.clone();
    hadamard(&mut transformed);
    for block in 0..3 {
        for column in 0_usize..128 {
            let exact: i32 = (0_usize..128)
                .map(|row| {
                    let sign = if (row & column).count_ones().is_multiple_of(2) {
                        1
                    } else {
                        -1
                    };
                    source[block * 128 + row] as i32 * sign
                })
                .sum();
            assert_eq!(
                transformed[block * 128 + column].to_bits(),
                (exact as f32 * RECIP_SQRT_BLOCK).to_bits()
            );
        }
    }
}

#[test]
fn half_block_steps_match_the_scalar_recipe() {
    if !avx2::is_supported() {
        return;
    }
    let (m, k) = (7, 384);
    let signs: Vec<f32> = (0..k)
        .map(|t| if t % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    // Row 3's middle block is all 2.5 (its class-1 center's value there,
    // so its centered block is zeros, step 1); row 5 holds one large value.
    let mut raw: Vec<f32> = (0..m * k)
        .map(|at| ((at * 17 % 257) as f32 - 128.0) / 61.0)
        .collect();
    raw[3 * k + 128..3 * k + 256].fill(2.5);
    raw[5 * k + 300] = 900.0;
    let mut centers = vec![0.0; k];
    centers.extend((0..2 * k).map(|at| {
        if at < k && (128..256).contains(&at) {
            2.5
        } else {
            f16::from_f32(((at * 31 % 89) as f32 - 44.0) / 23.0).to_f32()
        }
    }));
    let of_center: Vec<u32> = vec![0, 1, 0, 1, 2, 0, 2];
    let centering = avx2::Centering {
        means: &centers,
        seq: &of_center,
        gains: None,
    };
    let (actual, steps) = avx2::quantized_blocks(&raw, m, k, &signs, Some(centering), 1).unwrap();
    let (expected, expected_steps) =
        quantized_scalar(&centered_rows(&raw, k, &centers, &of_center, &signs), k, 1).unwrap();
    assert_eq!(actual, expected);
    same_bits(&steps, &expected_steps, "half block steps");
    let parts = k / HALF;
    assert_eq!(steps.len(), m * parts);
    assert_eq!(steps[3 * parts + 2..][..2], [1.0; 2], "zero block's steps");
    // H128 spreads the outlier over its block: about 900 / sqrt(128) in both halves.
    let (outlier, other) = (&steps[5 * parts + 4..][..2], &steps[5 * parts..][..2]);
    assert!(
        outlier
            .iter()
            .all(|&step| step > 10.0 * other[0].max(other[1])),
        "the outlier's block alone"
    );
}

/// A deferred LayerNorm's input stage: gain, center, then quantization in one and two digits.
#[test]
fn gained_blocks_match_the_scalar_recipe_in_one_and_two_digits() {
    if !avx2::is_supported() {
        return;
    }
    let (m, k) = (5, 256);
    let signs: Vec<f32> = (0..k).map(|t| if t % 7 < 3 { -1.0 } else { 1.0 }).collect();
    // Residual sums with a few large channels, and gains a hundredfold apart.
    let raw: Vec<f32> = (0..m * k)
        .map(|at| {
            let base = ((at * 23 % 211) as f32 - 105.0) / 7.0;
            if at % k % 53 == 0 { base * 90.0 } else { base }
        })
        .collect();
    let gains: Vec<f32> = (0..k)
        .map(|t| 0.03 + (t * 37 % 101) as f32 / 29.0)
        .collect();
    let centers: Vec<f32> = (0..3 * k)
        .map(|at| f16::from_f32(((at * 31 % 89) as f32 - 44.0) / 9.0).to_f32())
        .collect();
    let classes: Vec<u32> = vec![0, 2, 1, 2, 2];
    let centering = avx2::Centering {
        means: &centers,
        seq: &classes,
        gains: Some(&gains),
    };
    let (actual, steps) = avx2::quantized_blocks(&raw, m, k, &signs, Some(centering), 1).unwrap();
    let rotated = gained_rows(&raw, k, &gains, &centers, &classes, &signs);
    let (expected, expected_steps) = quantized_scalar(&rotated, k, 1).unwrap();
    assert_eq!(actual, expected);
    same_bits(&steps, &expected_steps, "half block steps");
    let plain = avx2::Centering {
        gains: None,
        ..centering
    };
    let (ungained, _) = avx2::quantized_blocks(&raw, m, k, &signs, Some(plain), 1).unwrap();
    assert_ne!(ungained, actual, "gains applied");
    let short = avx2::Centering {
        gains: Some(&gains[1..]),
        ..centering
    };
    assert_eq!(
        avx2::quantized_blocks(&raw, m, k, &signs, Some(short), 1),
        Err(Error::Dimensions)
    );

    // Of two digits the first are one digit's; both give a value back within half the second step.
    let (two, two_steps) = avx2::quantized_blocks(&raw, m, k, &signs, Some(centering), 2).unwrap();
    let (expected, expected_steps) = quantized_scalar(&rotated, k, 2).unwrap();
    assert_eq!(two, expected);
    same_bits(&two_steps, &expected_steps, "both digits' steps");
    let parts = k / HALF;
    let mut large = 0;
    for r in 0..m {
        let (digits, steps_of_row) = (&two[r * 2 * k..][..2 * k], &two_steps[r * 2 * parts..]);
        assert_eq!(
            digits[..k],
            actual[r * k..][..k],
            "row {r}: the first digits"
        );
        same_bits(
            &steps_of_row[..parts],
            &steps[r * parts..][..parts],
            "the first steps",
        );
        for t in 0..k {
            let (step, second) = (steps_of_row[t / HALF], steps_of_row[parts + t / HALF]);
            let back = f32::from(digits[t]) * step + f32::from(digits[k + t]) * second;
            assert!(
                (rotated[r * k + t] - back).abs() <= 0.51 * second,
                "row {r} value {t}: {} for {back}",
                rotated[r * k + t]
            );
            large += usize::from(digits[k + t].unsigned_abs() > 100);
        }
    }
    assert!(large > m * k / 20, "second digits use their range: {large}");
}

/// The ordered recipe; each part's dot is below 2^24, so exact in f32.
fn restored_oracle(qa: &[i8], q: &[i8], steps: &[f32], segment: usize) -> f32 {
    let dots: Vec<f32> = qa
        .as_chunks::<HALF>()
        .0
        .iter()
        .zip(q.as_chunks::<HALF>().0)
        .map(|(a, w)| {
            let dot: i32 = a
                .iter()
                .zip(w)
                .map(|(&a, &w)| i32::from(a) * 14 * i32::from(w))
                .sum();
            assert!(
                dot.unsigned_abs() < 1 << 24,
                "a part's dot converts exactly"
            );
            dot as f32
        })
        .collect();
    // A segment's chain in part order from zero; the segments add in order.
    dots.chunks(segment / HALF)
        .zip(steps.chunks(segment / HALF))
        .map(|(dots, steps)| {
            dots.iter()
                .zip(steps)
                .fold(0.0_f32, |f, (&dot, &step)| dot.mul_add(step, f))
        })
        .reduce(|total, f| total + f)
        .unwrap()
}

/// `project_restored_into` against its ordered parts at band, tail and split shapes.
#[test]
fn restored_projection_matches_its_parts() {
    use crate::gemm::Epilogue;
    use kodamaembed_model::repro::restored_class;
    if !is_supported() {
        return;
    }
    let shapes: [(usize, usize, &[usize], &[usize]); 5] = [
        (19, 384, &[1, 2, 38, 70, 2], &[384, 128, 64]),
        (1030, 1024, &[9], &[1024, 256]),
        (37, 256, &[4], &[256, 64]),
        (37, 256, &[1], &[256]),
        (37, 256, &[2], &[128]),
    ];
    for (n, k, lengths, segments) in shapes {
        let m: usize = lengths.iter().sum();
        let classes: Vec<u32> = lengths
            .iter()
            .flat_map(|&len| (0..len).map(move |t| restored_class(t, len) as u32))
            .collect();
        let signs: Vec<f32> = (0..k).map(|t| if t % 5 < 2 { -1.0 } else { 1.0 }).collect();
        let w: Vec<f32> = (0..n * k)
            .map(|at| ((at * 13 % 127) as f32 - 63.0) / 64.0)
            .collect();
        let class_means: Vec<f64> = (0..3 * k)
            .map(|at| ((at * 7 % 41) as f64 - 20.0) / 9.0)
            .collect();
        let one = calibrated(&w, n, k, &signs, &class_means, None);
        // GELU-like rows: mostly small, a few large channels.
        let a: Vec<f32> = (0..m * k)
            .map(|at| {
                let base = ((at * 29 % 131) as f32 - 40.0) / 97.0;
                if at % k % 61 == 0 { base * 40.0 } else { base }
            })
            .collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32 / 32.0 - 0.25).collect();
        let residual: Vec<f32> = (0..m * n)
            .map(|at| ((at % 13) as f32 - 6.0) / 32.0)
            .collect();
        for prepared in [&one, &in_two_digits(&one)] {
            let digits = prepared.digits;
            let (depth, parts) = (digits * k, digits * k / HALF);
            let statics = prepared.statics.as_ref().unwrap();
            let qw = walked(prepared);
            let centered = centered_rows(&a, k, &statics.centers, &classes, &signs);
            let (qa, steps) = quantized_scalar(&centered, k, digits).unwrap();
            for &segment in segments {
                let mut actual = vec![f32::NAN; m * n];
                let projected = prepared.project_restored_into(
                    &a,
                    m,
                    &classes,
                    segment,
                    Epilogue::BiasResidual(&bias, &residual),
                    &mut actual,
                );
                if digits == 2 && segment != k {
                    assert_eq!(projected, Err(Error::Dimensions), "k {k} segment {segment}");
                    assert!(actual.iter().all(|value| value.is_nan()));
                    continue;
                }
                projected.unwrap();
                for r in 0..m {
                    let class = classes[r] as usize;
                    for j in 0..n {
                        // Second digits run on in the chain of the first: one segment, the depth.
                        let f = restored_oracle(
                            &qa[r * depth..][..depth],
                            &qw[j * depth..][..depth],
                            &steps[r * parts..][..parts],
                            digits * segment,
                        );
                        let expected = f * (prepared.scales[j] / 14.0)
                            + (statics.shared[class * n + j] + bias[j])
                            + residual[r * n + j];
                        assert_eq!(
                            actual[r * n + j].to_bits(),
                            expected.to_bits(),
                            "{n}x{k} digits {digits} segment {segment} r {r} class {class} j {j}"
                        );
                    }
                }
            }
        }
    }
}

/// `project_deferred_into` against its parts; the sums against f64 references.
#[test]
fn deferred_projection_matches_its_parts() {
    use kodamaembed_model::repro::restored_class;
    if !is_supported() {
        return;
    }
    let shapes: [(usize, usize, &[usize], &[usize]); 3] = [
        (19, 384, &[1, 2, 38, 70, 2], &[384, 128]),
        (1030, 1024, &[9], &[1024, 256]),
        (37, 256, &[1], &[256, 64]),
    ];
    for (n, k, lengths, segments) in shapes {
        let m: usize = lengths.iter().sum();
        let classes: Vec<u32> = lengths
            .iter()
            .flat_map(|&len| (0..len).map(move |t| restored_class(t, len) as u32))
            .collect();
        let signs: Vec<f32> = (0..k).map(|t| if t % 5 < 2 { -1.0 } else { 1.0 }).collect();
        let w: Vec<f32> = (0..n * k)
            .map(|at| ((at * 13 % 127) as f32 - 63.0) / 64.0)
            .collect();
        let gains: Vec<f32> = (0..k).map(|t| 0.2 + (t * 17 % 23) as f32 / 11.0).collect();
        let beta: Vec<f32> = (0..k).map(|t| ((t * 5 % 19) as f32 - 9.0) / 40.0).collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32 / 32.0 - 0.25).collect();
        let class_means: Vec<f64> = (0..3 * k)
            .map(|at| ((at * 7 % 41) as f64 - 20.0) / 3.0)
            .collect();
        let norm = DeferredNorm {
            gains: &gains,
            beta: &beta,
            bias: &bias,
        };
        let one = calibrated(&w, n, k, &signs, &class_means, Some(norm));
        // FP16 residual sums: mostly moderate, a few massive channels.
        let z: Vec<f32> = (0..m * k)
            .map(|at| {
                let base = ((at * 29 % 131) as f32 - 60.0) / 11.0;
                let value = if at % k % 61 == 0 { base * 300.0 } else { base };
                f16::from_f32(value).to_f32()
            })
            .collect();
        let mean: Vec<f32> = (0..m).map(|r| (r % 7) as f32 / 5.0 - 0.6).collect();
        let rstd: Vec<f32> = (0..m).map(|r| 0.02 + (r % 11) as f32 / 37.0).collect();

        let deferred = one.deferred.as_ref().unwrap();
        for j in 0..n {
            let row = &w[j * k..][..k];
            let dot = |by: &[f32]| {
                row.iter()
                    .zip(by)
                    .fold(0.0_f64, |sum, (&w, &x)| sum + f64::from(w) * f64::from(x))
            };
            let (sum, shift) = (dot(&gains), dot(&beta));
            assert_eq!(
                deferred.sums[j].to_bits(),
                (sum as f32).to_bits(),
                "W gamma, j {j}"
            );
            assert_eq!(
                deferred.constants[j].to_bits(),
                ((shift + f64::from(bias[j])) as f32).to_bits(),
                "W beta + b, j {j}"
            );
        }
        for prepared in [&one, &in_two_digits(&one)] {
            let digits = prepared.digits;
            let (depth, parts) = (digits * k, digits * k / HALF);
            let statics = prepared.statics.as_ref().unwrap();
            let qw = walked(prepared);
            let gained = gained_rows(&z, k, &gains, &statics.centers, &classes, &signs);
            let (qa, steps) = quantized_scalar(&gained, k, digits).unwrap();
            for &segment in segments {
                for gelu in [false, true] {
                    let mut actual = vec![f32::NAN; m * n];
                    let projected = prepared.project_deferred_into(
                        &z,
                        m,
                        &classes,
                        &mean,
                        &rstd,
                        gelu,
                        segment,
                        &mut actual,
                    );
                    if digits == 2 && segment != k {
                        assert_eq!(projected, Err(Error::Dimensions), "k {k} segment {segment}");
                        assert!(actual.iter().all(|value| value.is_nan()));
                        continue;
                    }
                    projected.unwrap();
                    for r in 0..m {
                        let class = classes[r] as usize;
                        for j in 0..n {
                            // Second digits run on in the first's chain: one segment, the depth.
                            let f = restored_oracle(
                                &qa[r * depth..][..depth],
                                &qw[j * depth..][..depth],
                                &steps[r * parts..][..parts],
                                digits * segment,
                            );
                            let u = f * (prepared.scales[j] / 14.0) + statics.shared[class * n + j];
                            let value = (-mean[r])
                                .mul_add(deferred.sums[j], u)
                                .mul_add(rstd[r], deferred.constants[j]);
                            let expected = if gelu {
                                crate::math::gelu(value)
                            } else {
                                value
                            };
                            assert_eq!(
                                actual[r * n + j].to_bits(),
                                expected.to_bits(),
                                "{n}x{k} digits {digits} segment {segment} r {r} j {j} gelu {gelu}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Deferred and plain projections reject each other's inputs before writing.
#[test]
fn deferred_projection_requires_its_tables() {
    use crate::gemm::Epilogue;
    if !is_supported() {
        return;
    }
    let (n, k) = (16, 128);
    let signs = vec![1.0; k];
    let w: Vec<f32> = (0..n * k).map(|at| ((at % 9) as f32 - 4.0) / 8.0).collect();
    let z = vec![0.5; 2 * k];
    let (mean, rstd) = ([0.0; 2], [1.0; 2]);
    let mut out = vec![42.0; 2 * n];
    let means = vec![0.25; 3 * k];
    let plain = calibrated(&w, n, k, &signs, &means, None);
    assert_eq!(
        plain.project_deferred_into(&z, 2, &[0, 1], &mean, &rstd, false, k, &mut out),
        Err(Error::Unsupported)
    );
    let (gains, beta, bias) = (vec![1.0; k], vec![0.0; k], vec![0.0; n]);
    let norm = DeferredNorm {
        gains: &gains,
        beta: &beta,
        bias: &bias,
    };
    let prepared = calibrated(&w, n, k, &signs, &means, Some(norm));
    assert_eq!(
        prepared.project_deferred_into(&z, 2, &[0, 1], &mean[..1], &rstd, false, k, &mut out),
        Err(Error::Dimensions)
    );
    // A chain's restart must be a multiple of 64 inputs dividing the depth.
    for segment in [0, 32, 96, 256] {
        assert_eq!(
            prepared.project_deferred_into(&z, 2, &[0, 1], &mean, &rstd, false, segment, &mut out),
            Err(Error::Dimensions),
            "segment {segment}"
        );
    }
    assert_eq!(
        prepared.project_restored_into(&z, 2, &[0, 1], k, Epilogue::Bias(&bias), &mut out),
        Err(Error::Unsupported)
    );
    assert_eq!(out, vec![42.0; 2 * n]);
    let rows = vec![0.5_f32; 3 * k];
    let operand = Operand {
        rows: &rows,
        classes: &[0, 1, 2],
        means: &means,
        weights: None,
    };
    let short = DeferredNorm {
        gains: &gains[1..],
        beta: &beta,
        bias: &bias,
    };
    assert!(matches!(
        PreparedWeight::calibrated(&w, n, k, &signs, operand, Some(short), 1),
        Err(Error::Dimensions)
    ));
    // An operand's digits are one or two, and no table depends on them.
    let mut tables = Vec::new();
    prepared.write_tables(&mut tables).unwrap();
    for digits in [0, 3] {
        assert!(matches!(
            PreparedWeight::calibrated(&w, n, k, &signs, operand, None, digits),
            Err(Error::Dimensions)
        ));
        assert!(matches!(
            PreparedWeight::from_tables(n, k, &signs, Some(&gains), &mut &tables[..], digits),
            Err(Error::Dimensions)
        ));
    }
    let tables_of = |digits: usize| {
        let norm = DeferredNorm {
            gains: &gains,
            beta: &beta,
            bias: &bias,
        };
        let weight =
            PreparedWeight::calibrated(&w, n, k, &signs, operand, Some(norm), digits).unwrap();
        let mut tables = Vec::new();
        weight.write_tables(&mut tables).unwrap();
        tables
    };
    assert_eq!(tables_of(2), tables_of(1));
}

/// At the extremes a part's dot reaches 14,450,688 and a block's 28,901,376, still exact in f32.
#[test]
fn restored_dots_are_exact_at_their_largest() {
    use crate::gemm::Epilogue;
    if !is_supported() {
        return;
    }
    let (m, n, k) = (2, 16, 128);
    let qa: Vec<i8> = (0..m * k)
        .map(|at| if at < k { 127 } else { -127 })
        .collect();
    let a = Activation::new(&qa, m, k).unwrap();
    for q in [127_i8, -127] {
        let w = Weight::new(&vec![q; n * k], n, k).unwrap();
        let corrections = vec![128 * 64 * i32::from(q); n * k / 64];
        let mut out = vec![f32::NAN; m * n];
        super::super::fused_epilogue::dot_restored_into(
            &a,
            &w,
            &corrections,
            &[1.0; 4],
            &[1.0; 16],
            k,
            Epilogue::Bias(&[0.0; 16]),
            &[0, 0],
            &mut Target::whole(&mut out, n).unwrap(),
        )
        .unwrap();
        let block = 2.0 * 64.0 * 127.0 * 14.0 * f32::from(q);
        for (r, row) in out.chunks_exact(n).enumerate() {
            let expected = if r == 0 { block } else { -block };
            assert!(row.iter().all(|&v| v == expected), "q {q} row {r}: {row:?}");
        }
    }
}

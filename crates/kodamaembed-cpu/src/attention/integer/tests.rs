use super::*;
use crate::math::on_step;

fn supported() -> bool {
    crate::int8::is_supported()
}

/// A shared offset per channel gives centering something to remove.
fn sequence(len: usize, d: usize, heads: usize, seed: u64) -> Vec<f32> {
    let hidden = heads * d;
    let mut state = seed;
    (0..len * 3 * hidden)
        .map(|at| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            let channel = at % (3 * hidden);
            noise + ((channel % 7) as f32 - 3.0) * 0.75
        })
        .collect()
}

/// One head as attention runs it: prepared once, then every query block, or NaN throughout.
///
/// # Safety
/// AVX-VNNI, AVX2, FMA and F16C.
#[allow(clippy::too_many_arguments)]
unsafe fn attend_head(
    qkv: &[f32],
    stride: usize,
    hidden: usize,
    column: usize,
    len: usize,
    d: usize,
    key_center: Option<&[f32]>,
    head: &mut Head,
    block: &mut Block,
    mut store: impl FnMut(usize, &[f32]),
) {
    // SAFETY: the caller vouches for the features.
    unsafe {
        if head.prepare(qkv, stride, hidden, column, len, d, key_center) {
            for index in 0..head.blocks() {
                head.attend(&qkv[column..], stride, index, block, &mut store);
            }
        } else {
            let nan = vec![f32::NAN; d];
            for row in 0..len {
                store(row, &nan);
            }
        }
    }
}

fn run(qkv: &[f32], len: usize, d: usize, heads: usize, head: usize) -> Vec<f32> {
    run_centered(qkv, len, d, heads, head, None)
}

fn run_centered(
    qkv: &[f32],
    len: usize,
    d: usize,
    heads: usize,
    head: usize,
    key_center: Option<&[f32]>,
) -> Vec<f32> {
    let hidden = heads * d;
    let mut out = vec![f32::NAN; len * d];
    // SAFETY: callers check AVX-VNNI, AVX2, FMA and F16C support first.
    unsafe {
        attend_head(
            qkv,
            3 * hidden,
            hidden,
            head * d,
            len,
            d,
            key_center,
            &mut Head::default(),
            &mut Block::default(),
            |row, values| out[row * d..(row + 1) * d].copy_from_slice(values),
        )
    };
    out
}

fn fp16(x: f32) -> f32 {
    // SAFETY: callers check F16C support first.
    unsafe {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(
            _mm_set_ss(x),
        )))
    }
}

/// Each channel's offset less a quarter: near the keys, not their mean.
fn key_center(d: usize, heads: usize, head: usize) -> Vec<f32> {
    let hidden = heads * d;
    (0..d)
        .map(|t| (((hidden + head * d + t) % 7) as f32 - 3.0) * 0.75 - 0.25)
        .collect()
}

fn centered_oracle(rows: &[f32], stride: usize, len: usize, d: usize) -> Vec<f32> {
    let mean: Vec<f32> = (0..d)
        .map(|t| {
            ((0..len).fold(0.0_f64, |s, j| s + f64::from(rows[j * stride + t])) / len as f64) as f32
        })
        .collect();
    (0..len)
        .flat_map(|j| (0..d).map(move |t| (j, t)))
        .map(|(j, t)| rows[j * stride + t] - mean[t])
        .collect()
}

fn scale_oracle(largest: f32, levels: f32) -> f32 {
    let scale = largest / levels;
    if scale > 0.0 { scale } else { 1.0 }
}

fn rows_oracle(values: &[f32], d: usize) -> (Vec<i8>, Vec<f32>) {
    let mut out = Vec::new();
    let mut scales = Vec::new();
    for row in values.chunks_exact(d) {
        let scale = scale_oracle(row.iter().fold(0.0_f32, |m, &x| m.max(x.abs())), 127.0);
        scales.push(scale);
        out.extend(
            row.iter()
                .map(|&x| on_step(x, scale).round_ties_even() as i8),
        );
    }
    (out, scales)
}

fn weight_oracle(e: f32) -> i64 {
    let low = -(TABLE_SPAN as f32);
    let clamped = if e.is_nan() || e < low {
        low
    } else if e > 0.0 {
        0.0
    } else {
        e
    };
    i64::from(weight_table()[(-clamped.round_ties_even()) as usize])
}

/// The same recipe in scalar code with i64 integer dots.
fn oracle(qkv: &[f32], len: usize, d: usize, heads: usize, head: usize) -> Vec<f32> {
    oracle_centered(qkv, len, d, heads, head, None)
}

fn oracle_centered(
    qkv: &[f32],
    len: usize,
    d: usize,
    heads: usize,
    head: usize,
    key_center: Option<&[f32]>,
) -> Vec<f32> {
    let (hidden, stride, column) = (heads * d, 3 * heads * d, head * d);
    let raw_values = (0..len)
        .flat_map(|j| (0..d).map(move |t| (j, t)))
        .map(|(j, t)| qkv[j * stride + 2 * hidden + column + t]);
    let (keys, values): (Vec<f32>, Vec<f32>) = match key_center {
        None => (
            centered_oracle(&qkv[hidden + column..], stride, len, d),
            raw_values.collect(),
        ),
        Some(center) => (
            (0..len)
                .flat_map(|j| (0..d).map(move |t| (j, t)))
                .map(|(j, t)| qkv[j * stride + hidden + column + t] - center[t])
                .collect(),
            raw_values.map(fp16).collect(),
        ),
    };
    let queries: Vec<f32> = (0..len)
        .flat_map(|i| qkv[i * stride + column..][..d].iter().copied())
        .collect();
    let (k8, ks) = rows_oracle(&keys, d);
    let (q8, qs) = rows_oracle(&queries, d);
    let is_sink = |j: usize| j == 0 || j + 1 == len;
    let vs: Vec<f32> = (0..d)
        .map(|t| {
            let largest = (0..len)
                .filter(|&j| !is_sink(j))
                .fold(0.0_f32, |m, j| m.max(values[j * d + t].abs()));
            scale_oracle(largest, V_LEVELS)
        })
        .collect();
    let v8: Vec<i64> = values
        .iter()
        .enumerate()
        .map(|(at, &v)| {
            if is_sink(at / d) {
                0
            } else {
                on_step(v, vs[at % d]).round_ties_even() as i64
            }
        })
        .collect();
    let factor = 1.0 / (d as f32).sqrt() * std::f32::consts::LOG2_E * 256.0;
    let mut out = vec![0.0; len * d];
    for i in 0..len {
        let x: Vec<f32> = (0..len)
            .map(|j| {
                let dot: i64 = (0..d)
                    .map(|t| i64::from(q8[i * d + t]) * i64::from(k8[j * d + t]))
                    .sum();
                dot as f32 * ks[j]
            })
            .collect();
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<i64> = if max.is_finite() {
            let row_factor = qs[i] * factor;
            x.iter()
                .map(|&x| weight_oracle((x - max) * row_factor))
                .collect()
        } else {
            vec![0; len]
        };
        let total: i64 = w.iter().sum();
        let reciprocal = 1.0 / total as f32;
        for t in 0..d {
            if total == 0 {
                out[i * d + t] = f32::NAN;
                continue;
            }
            let p: i64 = (0..len).map(|j| w[j] * v8[j * d + t]).sum();
            let mut sum = p as f32 * vs[t];
            sum = (w[0] as f32).mul_add(values[t], sum);
            if len > 1 {
                sum = (w[len - 1] as f32).mul_add(values[(len - 1) * d + t], sum);
            }
            out[i * d + t] = sum * reciprocal;
        }
    }
    out
}

fn assert_bits(actual: &[f32], expected: &[f32], what: &str) {
    for (at, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "{what}, at={at}: {a} vs {e}");
    }
}

#[test]
fn integer_attention_matches_its_scalar_recipe_bit_for_bit() {
    if !supported() {
        return;
    }
    for (len, d, heads) in [
        (1, 32, 2),
        (2, 32, 1),
        (3, 16, 1),
        (7, 32, 2),
        (16, 16, 1),
        (33, 64, 3),
        (49, 32, 1),
        (96, 64, 2),
        (223, 64, 1),
        (512, 64, 1),
    ] {
        let qkv = sequence(len, d, heads, len as u64);
        for head in 0..heads {
            let actual = run(&qkv, len, d, heads, head);
            let expected = oracle(&qkv, len, d, heads, head);
            assert_bits(&actual, &expected, &format!("len={len},d={d},head={head}"));
        }
    }
}

#[test]
fn static_key_centers_match_their_scalar_recipe_bit_for_bit() {
    if !supported() {
        return;
    }
    for (len, d, heads) in [
        (1, 32, 2),
        (2, 16, 1),
        (7, 32, 2),
        (16, 16, 1),
        (33, 64, 3),
        (96, 64, 2),
        (223, 64, 1),
        (512, 64, 1),
    ] {
        let qkv = sequence(len, d, heads, 31 + len as u64);
        for head in 0..heads {
            let center = key_center(d, heads, head);
            let actual = run_centered(&qkv, len, d, heads, head, Some(&center));
            let expected = oracle_centered(&qkv, len, d, heads, head, Some(&center));
            assert_bits(&actual, &expected, &format!("len={len},d={d},head={head}"));
        }
    }
}

/// Past FP16's range a value makes the calibrated head NaN, not the uncalibrated.
#[test]
fn static_key_centers_reject_values_beyond_fp16() {
    if !supported() {
        return;
    }
    let (len, d, heads) = (9, 32, 2);
    let mut qkv = sequence(len, d, heads, 7);
    let hidden = heads * d;
    // Head 1's value at row 4, channel 3.
    qkv[4 * 3 * hidden + 2 * hidden + d + 3] = 7.0e4;
    let center = key_center(d, heads, 1);
    assert!(
        run_centered(&qkv, len, d, heads, 1, Some(&center))
            .iter()
            .all(|x| x.is_nan())
    );
    assert!(run(&qkv, len, d, heads, 1).iter().all(|x| x.is_finite()));
    let other = key_center(d, heads, 0);
    assert!(
        run_centered(&qkv, len, d, heads, 0, Some(&other))
            .iter()
            .all(|x| x.is_finite())
    );
}

#[test]
fn reused_scratch_matches_the_oracle() {
    if !supported() {
        return;
    }
    let (mut head_buffers, mut block) = (Head::default(), Block::default());
    for (len, d, heads, seed) in [
        (512, 64, 1, 1),
        (7, 32, 2, 2),
        (96, 64, 2, 3),
        (1, 32, 1, 4),
        (223, 64, 1, 5),
        (2, 32, 1, 7),
        (49, 32, 3, 6),
    ] {
        let qkv = sequence(len, d, heads, seed);
        for head in 0..heads {
            let mut out = vec![f32::NAN; len * d];
            // SAFETY: support was checked above.
            unsafe {
                attend_head(
                    &qkv,
                    3 * heads * d,
                    heads * d,
                    head * d,
                    len,
                    d,
                    None,
                    &mut head_buffers,
                    &mut block,
                    |row, values| out[row * d..(row + 1) * d].copy_from_slice(values),
                )
            };
            assert_bits(
                &out,
                &oracle(&qkv, len, d, heads, head),
                &format!("len={len},head={head}"),
            );
        }
    }
}

/// Values too small for any scale quantize to zeros, leaving the sinks' share.
#[test]
fn tiny_keys_and_values_quantize_to_zero() {
    if !supported() {
        return;
    }
    let (len, d) = (9, 32);
    let mut qkv = sequence(len, d, 1, 11);
    for (at, x) in qkv.iter_mut().enumerate() {
        if at % (3 * d) >= d {
            *x *= 1e-44;
        }
    }
    let actual = run(&qkv, len, d, 1, 0);
    assert_bits(&actual, &oracle(&qkv, len, d, 1, 0), "tiny");
    let top = f32::from(weight_table()[0]);
    let reciprocal = 1.0 / (top * len as f32);
    let share: Vec<f32> = (0..d)
        .map(|t| {
            let first = qkv[2 * d + t];
            let last = qkv[(len - 1) * 3 * d + 2 * d + t];
            top.mul_add(last, top.mul_add(first, 0.0)) * reciprocal
        })
        .collect();
    for row in actual.chunks_exact(d) {
        assert_bits(row, &share, "the sinks' share");
    }
}

/// A subnormal scale must not round down past the quantizer's endpoint.
#[test]
fn positive_subnormal_scales_keep_packed_integers_in_range() {
    if !supported() {
        return;
    }
    let (len, d) = (2, 32);
    let key_top = f32::from_bits(128);
    let keys: Vec<f32> = [key_top, -key_top]
        .into_iter()
        .flat_map(|x| std::iter::repeat_n(x, d))
        .collect();
    let mean = vec![0.0; d];
    let (mut panels, mut scales, mut corrections) = (Vec::new(), Vec::new(), Vec::new());
    // SAFETY: support was checked above, and both rows have d values.
    unsafe {
        assert!(pack_keys(
            &keys,
            d,
            len,
            d,
            &mean,
            &mut panels,
            &mut scales,
            &mut corrections,
        ));
    }
    for j in 0..len {
        let packed: Vec<i8> = (0..d)
            .map(|t| panels[t / GROUP * NR * GROUP + j * GROUP + t % GROUP])
            .collect();
        let sum: i32 = packed.iter().map(|&x| i32::from(x)).sum();
        assert_eq!(corrections[j], 128 * sum, "key {j} correction");
        // The least positive F32 scale that can represent this endpoint
        // without saturation is 2 * 2^-149, giving exactly +/-64.
        assert_eq!(scales[j].to_bits(), 2, "key {j} scale");
        assert!(packed.iter().all(|&x| x == if j == 0 { 64 } else { -64 }));
    }
    let (mut words, mut query_scales) = (Vec::new(), Vec::new());
    // SAFETY: support was checked above, and both rows have d values.
    unsafe { pack_queries(&keys, d, len, d, &mut words, &mut query_scales) };
    for (j, row) in words.chunks_exact(d / GROUP).enumerate() {
        assert_eq!(query_scales[j].to_bits(), 2);
        assert!(
            row.iter()
                .flat_map(|word| word.to_le_bytes())
                .all(|x| { i16::from(x) - 128 == if j == 0 { 64 } else { -64 } })
        );
    }
    // Large sinks; the others alone set the step, 2 * 2^-149 again: +/-64.
    let value_top = f32::from_bits(128);
    let values: Vec<f32> = [1.0e3, value_top, -value_top, -1.0e3]
        .into_iter()
        .flat_map(|x| std::iter::repeat_n(x, d))
        .collect();
    let (mut value_scales, mut packed) = (Vec::new(), Vec::new());
    // SAFETY: support was checked above, and every row has d values.
    unsafe {
        assert!(pack_values(
            &values,
            d,
            4,
            d,
            &mut value_scales,
            &mut packed
        ))
    };
    assert!(value_scales.iter().all(|s| s.to_bits() == 2));
    // Panel layout [d / 16][pairs][16 channels][2 keys]: key j of pair j / 2.
    for panel in packed.as_chunks::<{ 2 * 2 * NR }>().0 {
        for (pair, group) in panel.as_chunks::<{ 2 * NR }>().0.iter().enumerate() {
            for channel in group.as_chunks::<2>().0 {
                let expected = if pair == 0 { [0, 64] } else { [-64, 0] };
                assert_eq!(*channel, expected, "pair {pair}");
            }
        }
    }
}

/// Every entry off a whole octave lies at least 1e-6 from a rounding tie.
#[test]
fn weight_table_is_far_from_ties() {
    let table = weight_table();
    assert_eq!(table.len(), TABLE_SPAN + 1);
    assert_eq!(table[0], 16383);
    assert_eq!(table[256], 8192);
    assert_eq!(table[TABLE_SPAN], 0);
    assert!(table.windows(2).all(|w| w[1] <= w[0]));
    for (u, &w) in table.iter().enumerate() {
        let (octaves, rest) = (u / 256, u % 256);
        if rest == 0 {
            let exact = 16383.0 / f64::from(1u32 << octaves);
            assert_eq!(f64::from(w), exact.round_ties_even(), "u={u}");
            continue;
        }
        let exact = 16383.0 * (-(u as f64) / 256.0).exp2();
        let fraction = exact - exact.floor();
        assert!((fraction - 0.5).abs() > 1e-6, "u={u}: {exact}");
        assert_eq!(f64::from(w), exact.round_ties_even(), "u={u}");
    }
}

/// `weigh` against the scalar recipe at rounding ties, table ends and -inf.
#[test]
fn weights_are_table_entries_of_rounded_exponents() {
    if !supported() {
        return;
    }
    for len in [8, 16, 32, 64, 80, 512] {
        for max in [-128.0_f32, -1.0, -0.0, 0.0, 1.0, 128.0] {
            for factor in [0.125_f32, 2.0, 45.75, 256.0] {
                let mut row: Vec<f32> = (0..len)
                    .map(|at| max - ((at * 37 % 193) as f32) * 0.125)
                    .collect();
                // Exponents exactly -0.5, -1.5 and -2.5 (ties both ways).
                row[1 % len] = max - 0.5 / factor;
                row[2 % len] = max - 1.5 / factor;
                row[3 % len] = max - 2.5 / factor;
                row[len - 1] = f32::NEG_INFINITY;
                let mut weights = vec![-1; len];
                // SAFETY: support was checked above, and len is a multiple of 8.
                let sum = unsafe { weigh(&row, max, factor, &mut weights) };
                let expected: Vec<i16> = row
                    .iter()
                    .map(|&x| weight_oracle((x - max) * factor) as i16)
                    .collect();
                assert_eq!(weights, expected, "len={len},max={max},factor={factor}");
                assert_eq!(sum, expected.iter().map(|&x| i32::from(x)).sum::<i32>());
                assert_eq!(weights[0], 16383, "largest weighs most");
            }
        }
    }
}

/// The largest score's exponent is zero however large it is.
#[test]
fn large_finite_scores_keep_weights_bounded_and_totals_exact() {
    if !supported() {
        return;
    }
    for factor in [0.0625_f32, 45.75, 1.0e6] {
        for max in [536_870_976.0_f32, -536_870_976.0, f32::MAX, -f32::MAX] {
            let mut row = vec![f32::NEG_INFINITY; 80];
            row[0] = max;
            row[7] = max;
            row[32] = max;
            if max != -f32::MAX {
                row[1] = if max > 0.0 { max / 2.0 } else { max * 2.0 };
            }
            let mut weights = vec![-1; row.len()];
            // SAFETY: support was checked above, and the row is vector-aligned.
            let sum = unsafe { weigh(&row, max, factor, &mut weights) };
            assert!(
                weights.iter().all(|&x| (0..=16383).contains(&x)),
                "max={max}"
            );
            assert_eq!(sum, weights.iter().map(|&x| i32::from(x)).sum::<i32>());
            assert_eq!(sum, 3 * 16383, "max={max},factor={factor}");
        }
    }
}

#[test]
fn sinks_enter_exactly() {
    if !supported() {
        return;
    }
    let (len, d) = (12, 32);
    let mut qkv = sequence(len, d, 1, 17);
    for j in 0..len {
        for t in 0..d {
            let v = &mut qkv[j * 3 * d + 2 * d + t];
            *v = match j {
                0 => 3.0e4 + t as f32,
                11 => -2.5e4 - t as f32,
                _ => 0.0,
            };
        }
    }
    let actual = run(&qkv, len, d, 1, 0);
    assert_bits(&actual, &oracle(&qkv, len, d, 1, 0), "sinks");
    assert!(actual.iter().all(|x| x.is_finite()));
    // A weighted average of the sinks and zeros, within three roundings.
    let margin = 1.0 + 4.0 * f32::EPSILON / 2.0;
    for row in actual.chunks_exact(d) {
        for (t, &x) in row.iter().enumerate() {
            assert!(x <= (3.0e4 + t as f32) * margin && x >= (-2.5e4 - t as f32) * margin);
        }
    }
}

/// A query near FP32's limit keeps finite scores and one winning key.
#[test]
fn a_huge_query_attends_to_its_largest_score_alone() {
    if !supported() {
        return;
    }
    let (len, d) = (20, 32);
    let mut qkv = sequence(len, d, 1, 13);
    let row = 6;
    for x in &mut qkv[row * 3 * d..][..d] {
        *x *= 1e37;
    }
    let actual = run(&qkv, len, d, 1, 0);
    assert_bits(&actual, &oracle(&qkv, len, d, 1, 0), "huge query");
    assert!(actual[row * d..][..d].iter().all(|x| x.is_finite()));
    let mut weights = vec![-1; 24];
    let mut scores = vec![f32::NEG_INFINITY; 24];
    scores[..len].copy_from_slice(&[1.0; 20]);
    scores[3] = 2.0;
    // SAFETY: support was checked above, and the row is vector-aligned.
    let total = unsafe { weigh(&scores, 2.0, 1.0e37, &mut weights) };
    assert_eq!(total, 16383);
    assert!(
        weights
            .iter()
            .enumerate()
            .all(|(j, &w)| w == if j == 3 { 16383 } else { 0 })
    );
}

/// Exactly the rows whose scores overflow become NaN; the others keep values.
#[test]
fn overflowing_scores_give_nan_rows() {
    if !supported() {
        return;
    }
    let (len, d) = (20, 32);
    let mut qkv = sequence(len, d, 1, 13);
    let center = key_center(d, 1, 0);
    for t in 0..d {
        let mean = (0..len).map(|i| qkv[i * 3 * d + t]).sum::<f32>() / len as f32;
        qkv[9 * 3 * d + d + t] = center[t] + (qkv[3 * 3 * d + t] - mean) * 1e37;
    }
    let actual = run_centered(&qkv, len, d, 1, 0, Some(&center));
    let expected = oracle_centered(&qkv, len, d, 1, 0, Some(&center));
    assert!(actual[3 * d..][..d].iter().all(|x| x.is_nan()), "row 3");
    let mut nan_rows = 0;
    for (i, (a, e)) in actual
        .chunks_exact(d)
        .zip(expected.chunks_exact(d))
        .enumerate()
    {
        if a.iter().all(|x| x.is_nan()) {
            nan_rows += 1;
        } else {
            assert!(a.iter().all(|x| x.is_finite()), "row {i}");
        }
        assert_bits(a, e, &format!("row {i}"));
    }
    assert!(nan_rows > 0 && nan_rows < len, "{nan_rows} NaN rows");
}

#[test]
fn integer_attention_tracks_fp32_softmax_attention() {
    if !supported() {
        return;
    }
    let (len, d, heads) = (200, 64, 2);
    let qkv = sequence(len, d, heads, 9);
    let (hidden, stride) = (heads * d, 3 * heads * d);
    for head in 0..heads {
        let actual = run(&qkv, len, d, heads, head);
        let column = head * d;
        for i in 0..len {
            let q = &qkv[i * stride + column..][..d];
            let scores: Vec<f64> = (0..len)
                .map(|j| {
                    let k = &qkv[j * stride + hidden + column..][..d];
                    q.iter()
                        .zip(k)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>()
                        / (d as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for t in 0..d {
                let exact: f64 = (0..len)
                    .map(|j| weights[j] * f64::from(qkv[j * stride + 2 * hidden + column + t]))
                    .sum::<f64>()
                    / total;
                let error = (f64::from(actual[i * d + t]) - exact).abs();
                assert!(
                    error < 0.02,
                    "head={head},i={i},t={t}: {} vs {exact}",
                    actual[i * d + t]
                );
            }
        }
    }
}

#[test]
fn non_finite_inputs_give_nan_rows() {
    if !supported() {
        return;
    }
    let (len, d, heads) = (9, 32, 1);
    for part in 0..3 {
        for row in [0, 4, len - 1] {
            for bad in [f32::NAN, f32::INFINITY] {
                let mut qkv = sequence(len, d, heads, 3);
                qkv[row * 3 * d + part * d + 5] = bad;
                assert!(
                    run(&qkv, len, d, heads, 0).iter().all(|x| x.is_nan()),
                    "part {part}, row {row}, {bad}"
                );
            }
        }
    }
}

/// One worker thread at the engine's quality of service, for timing.
fn timing_worker() -> rayon::ThreadPool {
    crate::workers::worker_pool("timing", 1).expect("timing worker starts")
}

/// Peak instruction rates, twelve independent chains each (`--ignored`).
#[test]
#[ignore]
fn instruction_rates() {
    if !supported() {
        return;
    }
    // SAFETY: support was checked above.
    timing_worker().install(|| unsafe { rates() });
}

#[target_feature(enable = "avx2,fma,avxvnni")]
fn rates() {
    use std::hint::black_box;
    const ROUNDS: usize = 20_000_000;
    let per_second =
        |started: std::time::Instant| (ROUNDS * 12) as f64 / started.elapsed().as_secs_f64() / 1e9;
    let (a, b) = (
        black_box(_mm256_set1_ps(0.999_999)),
        black_box(_mm256_set1_ps(1e-7)),
    );
    let mut x = [_mm256_set1_ps(1.0); 12];
    let started = std::time::Instant::now();
    for _ in 0..ROUNDS {
        for x in &mut x {
            *x = _mm256_fmadd_ps(*x, a, b);
        }
    }
    let fma = per_second(started);
    black_box(x);
    let (u, w) = (
        black_box(_mm256_set1_epi8(3)),
        black_box(_mm256_set1_epi8(-2)),
    );
    let mut y = [_mm256_setzero_si256(); 12];
    let started = std::time::Instant::now();
    for _ in 0..ROUNDS {
        for y in &mut y {
            *y = _mm256_dpbusd_avx_epi32(*y, u, w);
        }
    }
    let dpbusd = per_second(started);
    black_box(y);
    let mut z = [_mm256_setzero_si256(); 12];
    let started = std::time::Instant::now();
    for _ in 0..ROUNDS {
        for z in &mut z {
            *z = _mm256_dpwssd_avx_epi32(*z, u, w);
        }
    }
    let dpwssd = per_second(started);
    black_box(z);
    eprintln!(
        "G instructions/s: FMA {fma:.2} ({:.0} GFLOP), VPDPBUSD {dpbusd:.2} ({:.0} GMAC), \
         VPDPWSSD {dpwssd:.2} ({:.0} GMAC)",
        fma * 16.0,
        dpbusd * 32.0,
        dpwssd * 16.0
    );
}

/// Time per head, integer and FP32, at e5 and MiniLM widths (`--ignored`).
#[test]
#[ignore]
fn integer_attention_time_per_head() {
    if !supported() {
        return;
    }
    let repeats = 200;
    timing_worker().install(|| {
        for (d, len) in [
            (64, 64),
            (64, 224),
            (64, 512),
            (32, 64),
            (32, 128),
            (32, 256),
        ] {
            let qkv = sequence(len, d, 1, 5);
            let mut context = vec![0.0f32; len * d];
            let mut time = |integer: bool| {
                let mut once = || {
                    crate::attention::attention_current_pool(
                        crate::Kernel::Avx2,
                        &qkv,
                        &[len],
                        d,
                        1,
                        integer,
                        None,
                        None,
                        None,
                        &mut context,
                    )
                };
                once();
                let started = std::time::Instant::now();
                for _ in 0..repeats {
                    once();
                }
                started.elapsed().as_secs_f64() * 1e6 / repeats as f64
            };
            let (integer, fp32) = (time(true), time(false));
            eprintln!(
                "d {d}, len {len}: integer {integer:.1} us, FP32 {fp32:.1} us per head ({:.2}x)",
                fp32 / integer
            );
        }
    });
}

/// Time per phase of one width-64 head (`--ignored`).
#[test]
#[ignore]
fn integer_attention_time_per_phase() {
    if !supported() {
        return;
    }
    let names = [
        "keys and values",
        "queries",
        "INT8 scores",
        "weights",
        "16-bit products",
        "outputs",
    ];
    let repeats = 100;
    timing_worker().install(|| {
        for len in [224, 512] {
            let line: Vec<String> = names
                .iter()
                .zip(phases(len, 64, repeats))
                .map(|(name, seconds)| format!("{name} {:.1}", seconds * 1e6 / repeats as f64))
                .collect();
            eprintln!("len {len} us per head: {}", line.join(", "));
        }
    });
}

fn phases(len: usize, d: usize, repeats: usize) -> [f64; 6] {
    let (stride, width, pairs) = (3 * d, len.next_multiple_of(NR), len.div_ceil(2));
    let qkv = sequence(len, d, 1, 5);
    let (mut h, mut s) = (Head::default(), Block::default());
    let mut phase = [0.0f64; 6];
    let values = &qkv[2 * d..];
    let (first_values, last_values) = (&values[..d], &values[(len - 1) * stride..][..d]);
    for _ in 0..repeats {
        // SAFETY: callers check AVX-VNNI, AVX2 and FMA support first.
        unsafe {
            let t = std::time::Instant::now();
            assert!(finite(&qkv, stride, len, d));
            means(&qkv[d..], stride, len, d, &mut h.sums, &mut h.key_mean);
            assert!(pack_keys(
                &qkv[d..],
                stride,
                len,
                d,
                &h.key_mean,
                &mut h.keys,
                &mut h.key_scales,
                &mut h.corrections,
            ));
            assert!(pack_values(
                values,
                stride,
                len,
                d,
                &mut h.value_scales,
                &mut h.values,
            ));
            phase[0] += t.elapsed().as_secs_f64();
            let keys = Keys {
                panels: &h.keys,
                scales: &h.key_scales,
                corrections: &h.corrections,
                len,
                d,
            };
            s.out.resize(d, 0.0);
            for first in (0..len).step_by(QUERY_BLOCK) {
                let block = QUERY_BLOCK.min(len - first);
                let t = std::time::Instant::now();
                pack_queries(
                    &qkv[first * stride..],
                    stride,
                    block,
                    d,
                    &mut s.queries,
                    &mut s.query_scales,
                );
                phase[1] += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                s.scores.resize(block * width, 0.0);
                s.maxima.resize(block, 0.0);
                scores(&keys, &s.queries, &mut s.scores, &mut s.maxima);
                phase[2] += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                s.weights.resize(block * width, 0);
                s.totals.clear();
                for (((row, weights), &max), &query_scale) in s
                    .scores
                    .chunks_exact(width)
                    .zip(s.weights.chunks_exact_mut(width))
                    .zip(&s.maxima)
                    .zip(&s.query_scales)
                {
                    s.totals
                        .push(weigh(row, max, row_factor(query_scale, d), weights));
                }
                phase[3] += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                s.products.resize(block * d, 0);
                products(
                    &s.weights,
                    width,
                    block,
                    &h.values,
                    pairs,
                    d,
                    &mut s.products,
                );
                phase[4] += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                for ((p, &total), weights) in s
                    .products
                    .chunks_exact(d)
                    .zip(&s.totals)
                    .zip(s.weights.chunks_exact(width))
                {
                    finish(
                        p,
                        &h.value_scales,
                        (weights[0], first_values),
                        Some((weights[len - 1], last_values)),
                        total,
                        &mut s.out,
                    );
                }
                phase[5] += t.elapsed().as_secs_f64();
            }
        }
    }
    phase
}

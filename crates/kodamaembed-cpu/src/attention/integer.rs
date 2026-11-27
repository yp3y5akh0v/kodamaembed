//! Integer attention for the centered INT8 preset: centered keys shift a row's scores
//! alike, which softmax ignores, and 14-bit weights keep products exact to 512 keys.

use crate::math::avx2::{Steps, round_to_i32};
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// The weight of a row's largest score.
const WEIGHT_LEVELS: f64 = 16383.0;
/// Exponent units per octave: exponents are rounded to 2^-8 in log2 units.
const EXPONENT_UNITS: f32 = 256.0;
/// Exponents at or below minus this weigh nothing (16383 2^-15 rounds to 0).
const TABLE_SPAN: usize = 15 * 256;
const V_LEVELS: f32 = 127.0;
const QK_LEVELS: f32 = 127.0;
// The supported limit keeps weight-value sums in INT32: 512 * 16383 * 127 < 2^31.
const MAX_KEYS: usize = 512;
const MR: usize = 6;
const NR: usize = 16;
/// Depth values per VPDPBUSD lane.
const GROUP: usize = 4;
/// Query rows per block: a block's scores and weights stay in L2.
const QUERY_BLOCK: usize = 48;

/// RNE(16383 2^(-u / 256)); any faithful exp2 builds it: off whole octaves no entry is near a tie.
pub(crate) fn weight_table() -> &'static [u16] {
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let units = EXPONENT_UNITS as usize;
        (0..=TABLE_SPAN)
            .map(|u| {
                let (octaves, rest) = (u / units, u % units);
                let fraction = if rest == 0 {
                    1.0
                } else {
                    (-(rest as f64) / units as f64).exp2()
                };
                let weight = WEIGHT_LEVELS * fraction / f64::from(1u32 << octaves);
                weight.round_ties_even() as u16
            })
            .collect()
    })
}

fn weight_entries() -> &'static [i32] {
    static ENTRIES: OnceLock<Vec<i32>> = OnceLock::new();
    ENTRIES.get_or_init(|| weight_table().iter().map(|&w| i32::from(w)).collect())
}

/// One head's centered keys and values, prepared once for all its query blocks.
#[derive(Default)]
pub(super) struct Head {
    len: usize,
    d: usize,
    sums: Vec<f64>,
    key_mean: Vec<f32>,
    keys: Vec<i8>,
    key_scales: Vec<f32>,
    corrections: Vec<i32>,
    values16: Vec<f32>,
    sinks: Vec<f32>,
    value_scales: Vec<f32>,
    values: Vec<i16>,
}

/// One query block's buffers, kept across blocks and batches.
#[derive(Default)]
pub(super) struct Block {
    queries: Vec<u32>,
    query_scales: Vec<f32>,
    scores: Vec<f32>,
    maxima: Vec<f32>,
    weights: Vec<i16>,
    totals: Vec<i32>,
    products: Vec<i32>,
    out: Vec<f32>,
}

struct Keys<'a> {
    panels: &'a [i8],
    scales: &'a [f32],
    corrections: &'a [i32],
    len: usize,
    d: usize,
}

#[target_feature(enable = "avx2,fma")]
fn magnitude(x: __m256) -> __m256 {
    _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX)))
}

/// All-ones lanes where x is finite: NaN and infinity both fail |x| < inf.
#[target_feature(enable = "avx2,fma")]
fn finite_lanes(x: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_LT_OQ>(magnitude(x), _mm256_set1_ps(f32::INFINITY))
}

/// The largest of eight values, none NaN.
#[target_feature(enable = "avx2,fma")]
fn largest(x: __m256) -> f32 {
    let m = _mm_max_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps::<1>(x));
    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    _mm_cvtss_f32(_mm_max_ss(m, _mm_movehdup_ps(m)))
}

#[target_feature(enable = "avx2,fma")]
fn total(x: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256::<1>(x));
    let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32::<1>(s)))
}

/// Quotients must lie within 511 in size.
#[target_feature(enable = "avx2,fma")]
fn rounded(x: __m256, steps: Steps) -> __m256i {
    round_to_i32(steps.on(x))
}

/// largest / levels, or one for zero; a subnormal scale rounds up, so no quotient saturates.
fn scale_for(largest: f32, levels: f32) -> f32 {
    let scale = largest / levels;
    if scale.is_subnormal() && f64::from(scale) * f64::from(levels) < f64::from(largest) {
        scale.next_up()
    } else if scale > 0.0 {
        scale
    } else {
        1.0
    }
}

#[target_feature(enable = "avx2,fma")]
fn finite(rows: &[f32], stride: usize, len: usize, d: usize) -> bool {
    let mut finite = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    for j in 0..len {
        for x in rows[j * stride..][..d].as_chunks::<8>().0 {
            // SAFETY: x holds eight values.
            finite = _mm256_and_ps(finite, finite_lanes(unsafe { _mm256_loadu_ps(x.as_ptr()) }));
        }
    }
    _mm256_movemask_ps(finite) == 0xFF
}

#[target_feature(enable = "avx2,fma,f16c")]
fn fp16_rows(rows: &[f32], stride: usize, len: usize, d: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(len * d, 0.0);
    for (j, row) in out.chunks_exact_mut(d).enumerate() {
        for (x, y) in rows[j * stride..][..d]
            .as_chunks::<8>()
            .0
            .iter()
            .zip(row.as_chunks_mut::<8>().0)
        {
            // SAFETY: x and y hold eight values each.
            unsafe {
                let half =
                    _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(x.as_ptr()));
                _mm256_storeu_ps(y.as_mut_ptr(), _mm256_cvtph_ps(half));
            }
        }
    }
}

/// f64 sums in row order, rounded once per column.
#[target_feature(enable = "avx2,fma")]
fn means(
    rows: &[f32],
    stride: usize,
    len: usize,
    d: usize,
    sums: &mut Vec<f64>,
    mean: &mut Vec<f32>,
) {
    sums.clear();
    sums.resize(d, 0.0);
    for j in 0..len {
        let row = rows[j * stride..][..d].as_chunks::<4>().0;
        for (sum, x) in sums.as_chunks_mut::<4>().0.iter_mut().zip(row) {
            // SAFETY: sum and x hold four values each.
            unsafe {
                let next = _mm256_add_pd(
                    _mm256_loadu_pd(sum.as_ptr()),
                    _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr())),
                );
                _mm256_storeu_pd(sum.as_mut_ptr(), next);
            }
        }
    }
    mean.clear();
    mean.extend(sums.iter().map(|&sum| (sum / len as f64) as f32));
}

/// Also 128 times each key's integer sum, which undoes the queries' unsigned offset in a score.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
fn pack_keys(
    rows: &[f32],
    stride: usize,
    len: usize,
    d: usize,
    mean: &[f32],
    panels: &mut Vec<i8>,
    scales: &mut Vec<f32>,
    corrections: &mut Vec<i32>,
) -> bool {
    let width = len.next_multiple_of(NR);
    panels.clear();
    panels.resize(width * d, 0);
    scales.clear();
    scales.resize(width, 0.0);
    corrections.clear();
    corrections.resize(width, 0);
    let mean = mean.as_chunks::<8>().0;
    // Scales first: each key's is a short chain the next key's can overlap.
    for (j, scale) in scales[..len].iter_mut().enumerate() {
        let row = rows[j * stride..][..d].as_chunks::<8>().0;
        let mut most = _mm256_setzero_ps();
        let mut finite = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
        for (x, m) in row.iter().zip(mean) {
            // SAFETY: x and m hold eight values each.
            let c =
                unsafe { _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr()), _mm256_loadu_ps(m.as_ptr())) };
            finite = _mm256_and_ps(finite, finite_lanes(c));
            most = _mm256_max_ps(most, magnitude(c));
        }
        if _mm256_movemask_ps(finite) != 0xFF {
            return false;
        }
        *scale = scale_for(largest(most), QK_LEVELS);
    }
    for j in 0..len {
        let row = rows[j * stride..][..d].as_chunks::<8>().0;
        let steps = Steps::new(_mm256_set1_ps(scales[j]));
        let mut sum = _mm256_setzero_si256();
        let at = j / NR * d * NR + j % NR * GROUP;
        for (half, (x, m)) in row.iter().zip(mean).enumerate() {
            // SAFETY: x and m hold eight values each.
            let c =
                unsafe { _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr()), _mm256_loadu_ps(m.as_ptr())) };
            let q = rounded(c, steps);
            sum = _mm256_add_epi32(sum, q);
            let q16 = _mm256_packs_epi32(q, q);
            // Values 0-3 lead the low lane, 4-7 the high one.
            let q8 = _mm256_packs_epi16(q16, q16);
            let group = at + 2 * half * NR * GROUP;
            // SAFETY: both words lie within the key's panel (width * d bytes).
            unsafe {
                let out = panels.as_mut_ptr();
                std::ptr::write_unaligned(out.add(group).cast(), _mm256_extract_epi32::<0>(q8));
                std::ptr::write_unaligned(
                    out.add(group + NR * GROUP).cast(),
                    _mm256_extract_epi32::<4>(q8),
                );
            }
        }
        corrections[j] = 128 * total(sum);
    }
    true
}

/// In _mm256_packs_epi32's order: channels 0-3, 8-11, 4-7, 12-15.
#[target_feature(enable = "avx2,fma")]
fn channels_i16(row: &[f32], panel: usize, scales: (Steps, Steps)) -> __m256i {
    let x = &row[panel * NR..][..NR];
    // SAFETY: x holds sixteen values.
    unsafe {
        _mm256_packs_epi32(
            rounded(_mm256_loadu_ps(x.as_ptr()), scales.0),
            rounded(_mm256_loadu_ps(x.as_ptr().add(8)), scales.1),
        )
    }
}

/// The first and last tokens draw the most attention, so their values stay exact.
fn sink(j: usize, len: usize) -> bool {
    j == 0 || j + 1 == len
}

/// The scale spans the non-sink tokens only, so large sinks cannot coarsen it.
#[target_feature(enable = "avx2,fma")]
fn pack_values(
    rows: &[f32],
    stride: usize,
    len: usize,
    d: usize,
    scales: &mut Vec<f32>,
    panels: &mut Vec<i16>,
) -> bool {
    scales.clear();
    scales.resize(d, 0.0);
    let mut finite = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    for j in 0..len {
        let row = rows[j * stride..][..d].as_chunks::<8>().0;
        for (x, top) in row.iter().zip(scales.as_chunks_mut::<8>().0) {
            // SAFETY: x and top hold eight values each.
            unsafe {
                let v = _mm256_loadu_ps(x.as_ptr());
                finite = _mm256_and_ps(finite, finite_lanes(v));
                if !sink(j, len) {
                    let most = _mm256_max_ps(_mm256_loadu_ps(top.as_ptr()), magnitude(v));
                    _mm256_storeu_ps(top.as_mut_ptr(), most);
                }
            }
        }
    }
    if _mm256_movemask_ps(finite) != 0xFF {
        return false;
    }
    for scale in scales.iter_mut() {
        *scale = scale_for(*scale, V_LEVELS);
    }
    let pairs = len.div_ceil(2);
    panels.clear();
    panels.resize(d * pairs * 2, 0);
    for (panel, out) in panels.chunks_exact_mut(pairs * NR * 2).enumerate() {
        // SAFETY: scales holds d values, sixteen from 16 panel.
        let s = unsafe {
            let s = scales.as_ptr().add(panel * NR);
            (
                Steps::new(_mm256_loadu_ps(s)),
                Steps::new(_mm256_loadu_ps(s.add(8))),
            )
        };
        for (g, group) in out.as_chunks_mut::<{ NR * 2 }>().0.iter_mut().enumerate() {
            let mut pair = [_mm256_setzero_si256(); 2];
            for (e, value) in pair.iter_mut().enumerate() {
                let j = 2 * g + e;
                if j < len && !sink(j, len) {
                    *value = channels_i16(&rows[j * stride..][..d], panel, s);
                }
            }
            let [even, odd] = pair;
            // SAFETY: group holds 32 values.
            unsafe {
                _mm256_storeu_si256(group.as_mut_ptr().cast(), _mm256_unpacklo_epi16(even, odd));
                _mm256_storeu_si256(
                    group.as_mut_ptr().add(NR).cast(),
                    _mm256_unpackhi_epi16(even, odd),
                );
            }
        }
    }
    true
}

/// Offset to unsigned bytes, since VPDPBUSD multiplies u8 by i8.
#[target_feature(enable = "avx2,fma")]
fn pack_queries(
    rows: &[f32],
    stride: usize,
    block: usize,
    d: usize,
    words: &mut Vec<u32>,
    scales: &mut Vec<f32>,
) {
    let groups = d / GROUP;
    words.clear();
    words.resize(block * groups, 0);
    scales.clear();
    // Scales first: each row's is a short chain the next row's can overlap.
    for i in 0..block {
        let mut most = _mm256_setzero_ps();
        for x in rows[i * stride..][..d].as_chunks::<8>().0 {
            // SAFETY: x holds eight values.
            most = _mm256_max_ps(most, magnitude(unsafe { _mm256_loadu_ps(x.as_ptr()) }));
        }
        scales.push(scale_for(largest(most), QK_LEVELS));
    }
    let offset = _mm256_set1_epi8(i8::MIN);
    for ((i, out), &scale) in words.chunks_exact_mut(groups).enumerate().zip(&*scales) {
        let row = rows[i * stride..][..d].as_chunks::<8>().0;
        let steps = Steps::new(_mm256_set1_ps(scale));
        for (x, pair) in row.iter().zip(out.as_chunks_mut::<2>().0) {
            // SAFETY: x holds eight values.
            let q = rounded(unsafe { _mm256_loadu_ps(x.as_ptr()) }, steps);
            let q16 = _mm256_packs_epi32(q, q);
            let q8 = _mm256_xor_si256(_mm256_packs_epi16(q16, q16), offset);
            *pair = [
                _mm256_extract_epi32::<0>(q8) as u32,
                _mm256_extract_epi32::<4>(q8) as u32,
            ];
        }
    }
}

/// Exact dots: VPDPBUSD of the unsigned queries less 128 times the key sums.
#[target_feature(enable = "avx2,fma,avxvnni")]
fn scores(keys: &Keys, queries: &[u32], out: &mut [f32], maxima: &mut [f32]) {
    let rows = maxima.len();
    let width = keys.len.next_multiple_of(NR);
    assert!(keys.d.is_multiple_of(NR) && keys.len <= MAX_KEYS);
    assert!(keys.panels.len() >= width * keys.d);
    assert!(keys.scales.len() >= width && keys.corrections.len() >= width);
    assert!(queries.len() >= rows * keys.d / GROUP);
    assert!(out.len() >= rows * width);
    for row in (0..rows).step_by(MR) {
        // SAFETY: the lengths were checked above.
        unsafe {
            match (rows - row).min(MR) {
                1 => score_rows::<1>(keys, queries, row, out, maxima),
                2 => score_rows::<2>(keys, queries, row, out, maxima),
                3 => score_rows::<3>(keys, queries, row, out, maxima),
                4 => score_rows::<4>(keys, queries, row, out, maxima),
                5 => score_rows::<5>(keys, queries, row, out, maxima),
                6 => score_rows::<6>(keys, queries, row, out, maxima),
                _ => unreachable!(),
            }
        }
    }
}

/// # Safety
/// The lengths `scores` checks.
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn score_rows<const R: usize>(
    keys: &Keys,
    queries: &[u32],
    row: usize,
    out: &mut [f32],
    maxima: &mut [f32],
) {
    let groups = keys.d / GROUP;
    let width = keys.len.next_multiple_of(NR);
    let below = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut most = [below; R];
    // SAFETY: the caller vouches for every index below.
    unsafe {
        let words = queries.as_ptr().add(row * groups);
        for panel in 0..width / NR {
            let mut lo = [_mm256_setzero_si256(); R];
            let mut hi = [_mm256_setzero_si256(); R];
            let packed = keys.panels.as_ptr().add(panel * keys.d * NR);
            for g in 0..groups {
                let b0 = _mm256_loadu_si256(packed.add(g * NR * GROUP).cast());
                let b1 = _mm256_loadu_si256(packed.add(g * NR * GROUP + 32).cast());
                for r in 0..R {
                    let x = _mm256_set1_epi32(*words.add(r * groups + g) as i32);
                    lo[r] = _mm256_dpbusd_avx_epi32(lo[r], x, b0);
                    hi[r] = _mm256_dpbusd_avx_epi32(hi[r], x, b1);
                }
            }
            let key = panel * NR;
            let c0 = _mm256_loadu_si256(keys.corrections.as_ptr().add(key).cast());
            let c1 = _mm256_loadu_si256(keys.corrections.as_ptr().add(key + 8).cast());
            let k0 = _mm256_loadu_ps(keys.scales.as_ptr().add(key));
            let k1 = _mm256_loadu_ps(keys.scales.as_ptr().add(key + 8));
            // Lanes past the last key score -inf.
            let left = _mm256_set1_epi32((keys.len - key) as i32);
            let m0 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(
                left,
                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
            ));
            let m1 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(
                left,
                _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15),
            ));
            let partial = key + NR > keys.len;
            for r in 0..R {
                let mut x0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(lo[r], c0)), k0);
                let mut x1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(hi[r], c1)), k1);
                if partial {
                    x0 = _mm256_blendv_ps(below, x0, m0);
                    x1 = _mm256_blendv_ps(below, x1, m1);
                }
                let at = out.as_mut_ptr().add((row + r) * width + key);
                _mm256_storeu_ps(at, x0);
                _mm256_storeu_ps(at.add(8), x1);
                most[r] = _mm256_max_ps(most[r], _mm256_max_ps(x0, x1));
            }
        }
    }
    for (r, most) in most.into_iter().enumerate() {
        maxima[row + r] = largest(most);
    }
}

fn row_factor(query_scale: f32, d: usize) -> f32 {
    query_scale * exponent_factor(d)
}

/// log2(e) / sqrt(d) in exponent units: fl(fl(1 / fl(sqrt d)) log2 e) 256.
fn exponent_factor(d: usize) -> f32 {
    1.0 / (d as f32).sqrt() * std::f32::consts::LOG2_E * EXPONENT_UNITS
}

/// A NaN exponent clamps to -TABLE_SPAN: maxps returns its second operand.
#[target_feature(enable = "avx2,fma")]
fn weight_indices(e: __m256) -> __m256i {
    let low = _mm256_set1_ps(-(TABLE_SPAN as f32));
    let e = _mm256_min_ps(_mm256_max_ps(e, low), _mm256_setzero_ps());
    _mm256_sub_epi32(_mm256_setzero_si256(), round_to_i32(e))
}

/// For the exhaustive cross-device check; needs AVX2 and FMA.
pub(crate) fn weight_indices_of(e: &[f32], out: &mut [u16]) {
    assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
    assert!(
        e.len() == out.len() && e.len().is_multiple_of(8),
        "whole vectors"
    );
    for (e, out) in e.as_chunks::<8>().0.iter().zip(out.as_chunks_mut::<8>().0) {
        let mut indices = [0_i32; 8];
        // SAFETY: AVX2 and FMA were detected; e and indices hold eight values.
        unsafe {
            let index = weight_indices(_mm256_loadu_ps(e.as_ptr()));
            _mm256_storeu_si256(indices.as_mut_ptr().cast(), index);
        }
        for (out, index) in out.iter_mut().zip(indices) {
            *out = u16::try_from(index).expect("index in table");
        }
    }
}

/// Exponents of (x - max) never exceed zero, however large the scores.
#[target_feature(enable = "avx2,fma")]
fn weigh(row: &[f32], max: f32, factor: f32, weights: &mut [i16]) -> i32 {
    assert!(row.len() == weights.len() && row.len().is_multiple_of(8));
    let entries = weight_entries();
    let (max, factor) = (_mm256_set1_ps(max), _mm256_set1_ps(factor));
    let mut sum = _mm256_setzero_si256();
    for (x, w) in row
        .as_chunks::<8>()
        .0
        .iter()
        .zip(weights.as_chunks_mut::<8>().0)
    {
        // SAFETY: x and w hold eight values; indices lie within the table.
        unsafe {
            let e = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr()), max), factor);
            let index = weight_indices(e);
            let q = _mm256_i32gather_epi32::<4>(entries.as_ptr(), index);
            sum = _mm256_add_epi32(sum, q);
            // Values lie in [0, 16383]: pack to 16 bits, then the two halves.
            let packed = _mm256_permute4x64_epi64::<0b1000>(_mm256_packs_epi32(q, q));
            _mm_storeu_si128(w.as_mut_ptr().cast(), _mm256_castsi256_si128(packed));
        }
    }
    total(sum)
}

/// ((products scales + w_first v_first) + w_last v_last) (1 / total) in order; NaN on overflow.
#[target_feature(enable = "avx2,fma")]
fn finish(
    products: &[i32],
    scales: &[f32],
    first: (i16, &[f32]),
    last: Option<(i16, &[f32])>,
    total: i32,
    out: &mut [f32],
) {
    if total == 0 {
        out.fill(f32::NAN);
        return;
    }
    let reciprocal = _mm256_set1_ps(1.0 / total as f32);
    let first_weight = _mm256_set1_ps(f32::from(first.0));
    for (at, ((p, s), o)) in products
        .as_chunks::<8>()
        .0
        .iter()
        .zip(scales.as_chunks::<8>().0)
        .zip(out.as_chunks_mut::<8>().0)
        .enumerate()
    {
        // SAFETY: p, s and o hold eight values; sink rows hold d.
        unsafe {
            let mut sum = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(p.as_ptr().cast())),
                _mm256_loadu_ps(s.as_ptr()),
            );
            sum = _mm256_fmadd_ps(
                first_weight,
                _mm256_loadu_ps(first.1[8 * at..].as_ptr()),
                sum,
            );
            if let Some((weight, values)) = last {
                let weight = _mm256_set1_ps(f32::from(weight));
                sum = _mm256_fmadd_ps(weight, _mm256_loadu_ps(values[8 * at..].as_ptr()), sum);
            }
            _mm256_storeu_ps(o.as_mut_ptr(), _mm256_mul_ps(sum, reciprocal));
        }
    }
}

impl Head {
    /// False when an input is not finite, which makes the whole head NaN.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub(super) fn prepare(
        &mut self,
        qkv: &[f32],
        stride: usize,
        hidden: usize,
        column: usize,
        len: usize,
        d: usize,
        key_center: Option<&[f32]>,
    ) -> bool {
        assert!(len > 0 && len <= MAX_KEYS && d.is_multiple_of(NR));
        assert!(key_center.is_none_or(|center| center.len() == d));
        (self.len, self.d) = (len, d);
        let (queries, keys, values) = (
            &qkv[column..],
            &qkv[hidden + column..],
            &qkv[2 * hidden + column..],
        );
        // The packers reject non-finite centered keys and values themselves.
        finite(queries, stride, len, d)
            && {
                let center = match key_center {
                    Some(center) => center,
                    None => {
                        means(keys, stride, len, d, &mut self.sums, &mut self.key_mean);
                        &self.key_mean
                    }
                };
                pack_keys(
                    keys,
                    stride,
                    len,
                    d,
                    center,
                    &mut self.keys,
                    &mut self.key_scales,
                    &mut self.corrections,
                )
            }
            && {
                let (rows, row_stride) = if key_center.is_some() {
                    fp16_rows(values, stride, len, d, &mut self.values16);
                    (&self.values16[..], d)
                } else {
                    (values, stride)
                };
                self.sinks.clear();
                self.sinks.extend_from_slice(&rows[..d]);
                self.sinks
                    .extend_from_slice(&rows[(len - 1) * row_stride..][..d]);
                pack_values(
                    rows,
                    row_stride,
                    len,
                    d,
                    &mut self.value_scales,
                    &mut self.values,
                )
            }
    }

    pub(super) fn blocks(&self) -> usize {
        self.len.div_ceil(QUERY_BLOCK)
    }

    /// One block of the prepared head's query rows; an overflowing row is NaN.
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub(super) fn attend(
        &self,
        queries: &[f32],
        stride: usize,
        block: usize,
        s: &mut Block,
        mut store: impl FnMut(usize, &[f32]),
    ) {
        let (len, d) = (self.len, self.d);
        let keys = Keys {
            panels: &self.keys,
            scales: &self.key_scales,
            corrections: &self.corrections,
            len,
            d,
        };
        let (width, pairs) = (len.next_multiple_of(NR), len.div_ceil(2));
        let (first_values, last_values) = self.sinks.split_at(d);
        let first = block * QUERY_BLOCK;
        let block = QUERY_BLOCK.min(len - first);
        s.out.clear();
        s.out.resize(d, 0.0);
        pack_queries(
            &queries[first * stride..],
            stride,
            block,
            d,
            &mut s.queries,
            &mut s.query_scales,
        );
        s.scores.resize(block * width, 0.0);
        s.maxima.resize(block, 0.0);
        scores(&keys, &s.queries, &mut s.scores, &mut s.maxima);
        s.weights.resize(block * width, 0);
        s.totals.clear();
        for (((row, weights), &max), &query_scale) in s
            .scores
            .chunks_exact(width)
            .zip(s.weights.chunks_exact_mut(width))
            .zip(&s.maxima)
            .zip(&s.query_scales)
        {
            // An overflowed row keeps no weights; `finish` makes it NaN.
            s.totals.push(if max.is_finite() {
                weigh(row, max, row_factor(query_scale, d), weights)
            } else {
                weights.fill(0);
                0
            });
        }
        s.products.resize(block * d, 0);
        products(
            &s.weights,
            width,
            block,
            &self.values,
            pairs,
            d,
            &mut s.products,
        );
        for (i, ((p, &total), weights)) in s
            .products
            .chunks_exact(d)
            .zip(&s.totals)
            .zip(s.weights.chunks_exact(width))
            .enumerate()
        {
            let last = (len > 1).then(|| (weights[len - 1], last_values));
            finish(
                p,
                &self.value_scales,
                (weights[0], first_values),
                last,
                total,
                &mut s.out,
            );
            store(first + i, &s.out);
        }
    }
}

/// Exact INT32 products; weights pair up for VPDPWSSD.
#[target_feature(enable = "avx2,avxvnni")]
fn products(
    weights: &[i16],
    pitch: usize,
    rows: usize,
    values: &[i16],
    pairs: usize,
    d: usize,
    out: &mut [i32],
) {
    assert!(d.is_multiple_of(NR) && pitch >= 2 * pairs && weights.len() >= rows * pitch);
    assert!(values.len() >= d * pairs * 2 && out.len() >= rows * d);
    for row in (0..rows).step_by(MR) {
        for panel in 0..d / NR {
            // SAFETY: the lengths were checked above.
            unsafe {
                match (rows - row).min(MR) {
                    1 => tile::<1>(weights, pitch, row, values, pairs, panel, d, out),
                    2 => tile::<2>(weights, pitch, row, values, pairs, panel, d, out),
                    3 => tile::<3>(weights, pitch, row, values, pairs, panel, d, out),
                    4 => tile::<4>(weights, pitch, row, values, pairs, panel, d, out),
                    5 => tile::<5>(weights, pitch, row, values, pairs, panel, d, out),
                    6 => tile::<6>(weights, pitch, row, values, pairs, panel, d, out),
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// # Safety
/// The lengths `products` checks.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn tile<const R: usize>(
    weights: &[i16],
    pitch: usize,
    row: usize,
    values: &[i16],
    pairs: usize,
    panel: usize,
    d: usize,
    out: &mut [i32],
) {
    // SAFETY: the caller vouches for the lengths.
    unsafe {
        let mut lo = [_mm256_setzero_si256(); R];
        let mut hi = [_mm256_setzero_si256(); R];
        let values = values.as_ptr().add(panel * pairs * NR * 2);
        for g in 0..pairs {
            let b0 = _mm256_loadu_si256(values.add(g * NR * 2).cast());
            let b1 = _mm256_loadu_si256(values.add(g * NR * 2 + NR).cast());
            for r in 0..R {
                let pair = std::ptr::read_unaligned(
                    weights
                        .as_ptr()
                        .add((row + r) * pitch + 2 * g)
                        .cast::<i32>(),
                );
                let x = _mm256_set1_epi32(pair);
                lo[r] = _mm256_dpwssd_avx_epi32(lo[r], x, b0);
                hi[r] = _mm256_dpwssd_avx_epi32(hi[r], x, b1);
            }
        }
        for r in 0..R {
            let at = out.as_mut_ptr().add((row + r) * d + panel * NR);
            _mm256_storeu_si256(at.cast(), lo[r]);
            _mm256_storeu_si256(at.add(8).cast(), hi[r]);
        }
    }
}

#[cfg(test)]
mod tests;

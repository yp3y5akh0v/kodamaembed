//! SIMD activation preparation with FP16 rounding and fixed quantization order.
use super::{BLOCK, RECIP_SQRT_BLOCK};
use crate::int8::{Error, HALF};
use crate::math::avx2::Steps;
use std::arch::x86_64::*;

pub(super) fn is_supported() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("fma")
}

/// FP16 values, times a deferred norm's gains, less their class's mean.
#[derive(Clone, Copy)]
pub(super) struct Centering<'a> {
    pub means: &'a [f32],
    pub seq: &'a [u32],
    pub gains: Option<&'a [f32]>,
}

/// One or two digits on a step each half block: its largest magnitude over 127, 1 for zeros.
pub(super) fn quantized_blocks(
    raw: &[f32],
    rows: usize,
    k: usize,
    signs: &[f32],
    centering: Option<Centering>,
    digits: usize,
) -> Result<(Vec<i8>, Vec<f32>), Error> {
    if !is_supported() {
        return Err(Error::Unsupported);
    }
    if k == 0
        || !k.is_multiple_of(BLOCK)
        || signs.len() != k
        || rows.checked_mul(k) != Some(raw.len())
        || !(1..=2).contains(&digits)
    {
        return Err(Error::Dimensions);
    }
    if let Some(c) = centering
        && (c.seq.len() < rows
            || c.seq[..rows]
                .iter()
                .any(|&s| (s as usize + 1) * k > c.means.len())
            || c.gains.is_some_and(|gains| gains.len() != k))
    {
        return Err(Error::Dimensions);
    }
    // SAFETY: K is whole rotation blocks and every slice is checked against it.
    unsafe { prepare(raw, rows, k, signs, centering, digits) }
}

#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn prepare(
    raw: &[f32],
    rows: usize,
    k: usize,
    signs: &[f32],
    centering: Option<Centering>,
    digits: usize,
) -> Result<(Vec<i8>, Vec<f32>), Error> {
    // Constant gains and means leave the block loops without a branch.
    unsafe {
        match centering {
            Some(c) if c.gains.is_some() => {
                prepare_rows::<true, true>(raw, rows, k, signs, centering, digits)
            }
            Some(_) => prepare_rows::<false, true>(raw, rows, k, signs, centering, digits),
            None => prepare_rows::<false, false>(raw, rows, k, signs, centering, digits),
        }
    }
}

const VECTORS: usize = BLOCK / 8;
/// The vectors of half a block: the 64 inputs one part's dot covers.
const HALF_VECTORS: usize = VECTORS / 2;
/// A first digit leaves at most half its step: 127 of a step 254 times finer.
const SECOND_LEVELS: f32 = 254.0;

/// Every block stays in registers from its load to its bytes.
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn prepare_rows<const GAINS: bool, const MEAN: bool>(
    raw: &[f32],
    rows: usize,
    k: usize,
    signs: &[f32],
    centering: Option<Centering>,
    digits: usize,
) -> Result<(Vec<i8>, Vec<f32>), Error> {
    unsafe {
        // A row's bytes and steps hold its second digits behind its first.
        let parts = k / HALF;
        let mut quantized = vec![0_i8; digits * raw.len()];
        let mut scales = vec![0.0_f32; rows * digits * parts];
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
        let recip = _mm256_set1_ps(RECIP_SQRT_BLOCK);
        let gains = centering.and_then(|c| c.gains).unwrap_or(&[]);
        for row in 0..rows {
            let output = quantized.as_mut_ptr().add(row * digits * k);
            let steps = &mut scales[row * digits * parts..][..digits * parts];
            let mean = match centering {
                Some(c) => &c.means[c.seq[row] as usize * k..][..k],
                None => &[],
            };
            let inputs = Inputs {
                raw: &raw[row * k..][..k],
                gains,
                mean,
                signs,
            };
            // Every failure is one error for the whole call, so a row checks once.
            let mut fine = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
            for first in (0..k).step_by(BLOCK) {
                let low = half::<GAINS, MEAN>(inputs, first, &mut fine);
                let high = half::<GAINS, MEAN>(inputs, first + BLOCK / 2, &mut fine);
                let mut v = [_mm256_setzero_ps(); VECTORS];
                for j in 0..8 {
                    v[j] = _mm256_add_ps(low[j], high[j]);
                    v[j + 8] = _mm256_sub_ps(low[j], high[j]);
                }
                let mut magnitudes = [_mm256_setzero_ps(); VECTORS];
                for (x, m) in v.iter_mut().zip(&mut magnitudes) {
                    *x = _mm256_mul_ps(*x, recip);
                    *m = _mm256_and_ps(*x, abs_mask);
                    fine = _mm256_and_ps(fine, finite(*x));
                }
                let halves = v
                    .as_chunks::<HALF_VECTORS>()
                    .0
                    .iter()
                    .zip(magnitudes.as_chunks_mut::<HALF_VECTORS>().0);
                for (side, (vectors, magnitudes)) in halves.enumerate() {
                    // Finite maxima are exact in any order; a NaN already fails `fine`.
                    for width in [4, 2, 1] {
                        for i in 0..width {
                            magnitudes[i] = _mm256_max_ps(magnitudes[i], magnitudes[i + width]);
                        }
                    }
                    let scale = step_of(magnitudes[0])?;
                    let part = first / HALF + side;
                    steps[part] = scale;
                    let out = output.add(first + side * HALF);
                    let rounded = if digits == 2 {
                        let second = scale / SECOND_LEVELS;
                        steps[parts + part] = second;
                        quantize_half_twice(vectors, scale, second, out, out.add(k))
                    } else {
                        quantize_half(vectors, scale, out)
                    };
                    fine = _mm256_and_ps(fine, rounded);
                }
            }
            if _mm256_movemask_ps(fine) != 255 {
                return Err(Error::FloatRange);
            }
        }
        Ok((quantized, scales))
    }
}

/// All ones in each lane whose magnitude is below infinity.
#[target_feature(enable = "avx2")]
fn finite(x: __m256) -> __m256 {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
    _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(x, abs_mask), _mm256_set1_ps(f32::INFINITY))
}

/// The butterflies 1, 2 and 4 apart, one swap a stage: the kept lanes of x + y and y - x are
/// the scalar recipe's a + b and a - b, operands in order.
#[target_feature(enable = "avx2")]
fn butterflies_within(x: __m256) -> __m256 {
    let y = _mm256_permute_ps::<0xB1>(x);
    let x = _mm256_blend_ps::<0xAA>(_mm256_add_ps(x, y), _mm256_sub_ps(y, x));
    let y = _mm256_permute_ps::<0x4E>(x);
    let x = _mm256_blend_ps::<0xCC>(_mm256_add_ps(x, y), _mm256_sub_ps(y, x));
    let y = _mm256_permute2f128_ps::<0x01>(x, x);
    _mm256_blend_ps::<0xF0>(_mm256_add_ps(x, y), _mm256_sub_ps(y, x))
}

/// A constant stride unrolls into register operations.
#[target_feature(enable = "avx2")]
fn stage<const H: usize>(v: &mut [__m256; 8]) {
    for group in (0..8).step_by(2 * H) {
        for j in group..group + H {
            let (a, b) = (v[j], v[j + H]);
            v[j] = _mm256_add_ps(a, b);
            v[j + H] = _mm256_sub_ps(a, b);
        }
    }
}

/// One row's raw values, its gains and class mean when used, and the signs.
#[derive(Clone, Copy)]
struct Inputs<'a> {
    raw: &'a [f32],
    gains: &'a [f32],
    mean: &'a [f32],
    signs: &'a [f32],
}

/// The butterflies 1 to 32 apart never pair values from different halves, so 64 values
/// rotate in registers before the halves meet.
///
/// # Safety
/// AVX2 and F16C; each slice used holds 64 values from `t`.
#[target_feature(enable = "avx2,f16c")]
unsafe fn half<const GAINS: bool, const MEAN: bool>(
    inputs: Inputs<'_>,
    t: usize,
    fine: &mut __m256,
) -> [__m256; 8] {
    let mut v = [_mm256_setzero_ps(); 8];
    for (i, x) in v.iter_mut().enumerate() {
        let at = t + 8 * i;
        // SAFETY: the caller vouches for 64 values from `t` in every slice read.
        unsafe {
            let raw_x = _mm256_loadu_ps(inputs.raw.as_ptr().add(at));
            *fine = _mm256_and_ps(*fine, finite(raw_x));
            let mut y = _mm256_cvtph_ps(_mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(raw_x));
            if GAINS {
                y = _mm256_mul_ps(y, _mm256_loadu_ps(inputs.gains.as_ptr().add(at)));
            }
            if MEAN {
                y = _mm256_sub_ps(y, _mm256_loadu_ps(inputs.mean.as_ptr().add(at)));
            }
            let scaled = _mm256_mul_ps(y, _mm256_loadu_ps(inputs.signs.as_ptr().add(at)));
            *fine = _mm256_and_ps(*fine, finite(scaled));
            *x = butterflies_within(scaled);
        }
    }
    stage::<1>(&mut v);
    stage::<2>(&mut v);
    stage::<4>(&mut v);
    v
}

/// The largest of eight magnitudes over 127, 1 for zeros.
#[target_feature(enable = "avx2")]
fn step_of(x: __m256) -> Result<f32, Error> {
    let x = _mm256_max_ps(x, _mm256_permute2f128_ps::<0x01>(x, x));
    let x = _mm256_max_ps(x, _mm256_permute_ps::<0x4E>(x));
    let largest = _mm256_cvtss_f32(_mm256_max_ps(x, _mm256_permute_ps::<0xB1>(x)));
    let scale = if largest > 0.0 { largest / 127.0 } else { 1.0 };
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::FloatRange);
    }
    Ok(scale)
}

/// Rounds half a rotated block on its step into 64 bytes; lanes past 127 clear the returned mask.
///
/// # Safety
/// AVX2; `out` holds 64 bytes.
#[target_feature(enable = "avx2")]
unsafe fn quantize_half(block: &[__m256; HALF_VECTORS], scale: f32, out: *mut i8) -> __m256 {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
    let cap = _mm256_set1_ps(127.0);
    // The packs interleave 128-bit lanes; this puts the 32 bytes back in order.
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let steps = Steps::new(_mm256_set1_ps(scale));
    let mut fine = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    for (quad, vectors) in block.as_chunks::<4>().0.iter().enumerate() {
        let mut q32 = [_mm256_setzero_si256(); 4];
        for (q, &x) in q32.iter_mut().zip(vectors) {
            let rounded =
                _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(steps.on(x));
            let in_range = _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_and_ps(rounded, abs_mask), cap);
            fine = _mm256_and_ps(fine, in_range);
            *q = _mm256_cvttps_epi32(rounded);
        }
        let bytes = _mm256_packs_epi16(
            _mm256_packs_epi32(q32[0], q32[1]),
            _mm256_packs_epi32(q32[2], q32[3]),
        );
        // SAFETY: four vectors are 32 of the half's 64 bytes.
        unsafe {
            _mm256_storeu_si256(
                out.add(32 * quad).cast(),
                _mm256_permutevar8x32_epi32(bytes, order),
            );
        }
    }
    fine
}

/// Half a block in two digits, the second of what the first leaves; lanes past 127 clear the mask.
///
/// # Safety
/// AVX2 and FMA; `out` and `behind` hold 64 bytes each.
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_half_twice(
    block: &[__m256; HALF_VECTORS],
    scale: f32,
    second: f32,
    out: *mut i8,
    behind: *mut i8,
) -> __m256 {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
    let cap = _mm256_set1_ps(127.0);
    // The packs interleave 128-bit lanes; this puts the 32 bytes back in order.
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let step = _mm256_set1_ps(scale);
    let (steps, seconds) = (Steps::new(step), Steps::new(_mm256_set1_ps(second)));
    let mut fine = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    for (quad, vectors) in block.as_chunks::<4>().0.iter().enumerate() {
        let mut q32 = [_mm256_setzero_si256(); 4];
        let mut r32 = [_mm256_setzero_si256(); 4];
        for ((q, r), &x) in q32.iter_mut().zip(&mut r32).zip(vectors) {
            let first =
                _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(steps.on(x));
            // One FMA: what the first digit leaves has one rounding on every device.
            let left = _mm256_fnmadd_ps(first, step, x);
            let rest = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                seconds.on(left),
            );
            let in_range = _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_and_ps(first, abs_mask), cap),
                _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_and_ps(rest, abs_mask), cap),
            );
            fine = _mm256_and_ps(fine, in_range);
            *q = _mm256_cvttps_epi32(first);
            *r = _mm256_cvttps_epi32(rest);
        }
        for (digits, to) in [(q32, out), (r32, behind)] {
            let bytes = _mm256_packs_epi16(
                _mm256_packs_epi32(digits[0], digits[1]),
                _mm256_packs_epi32(digits[2], digits[3]),
            );
            // SAFETY: four vectors are 32 of the half's 64 bytes, in each digit's row.
            unsafe {
                _mm256_storeu_si256(
                    to.add(32 * quad).cast(),
                    _mm256_permutevar8x32_epi32(bytes, order),
                );
            }
        }
    }
    fine
}

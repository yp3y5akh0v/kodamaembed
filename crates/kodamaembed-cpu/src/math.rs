//! Elementwise functions: eight lanes at a time with AVX2 or NEON, else one at a time.

use std::f32::consts::FRAC_1_SQRT_2;

/// Abramowitz and Stegun 7.1.28, erf within 3e-7: for z >= 0,
/// 1 - erf(z) = (1 + a1 z + ... + a6 z^6)^-16.
pub(crate) const GELU_A: [f32; 6] = [
    0.070_523_076,
    0.042_282_01,
    0.009_270_527,
    0.000_152_014_3,
    0.000_276_567_2,
    0.000_043_063_8,
];
/// Centered INT8 denominator cap, 2^100, keeps its reciprocal normal.
pub(crate) const GELU_CAP: f32 = 1.267_650_6e30;

/// x / 2 (2 - r) for x >= 0, x / 2 r otherwise, r = 1 / min(p^16, 2^100): no
/// exponential and no cancellation below zero; within 7.4e-7 on [-10, 10].
#[cfg(target_arch = "x86_64")]
pub(crate) fn gelu(x: f32) -> f32 {
    gelu_with_cap::<true>(x)
}

/// An overflowing denominator gives a zero complement, so large negative inputs approach zero.
pub(crate) fn gelu_unbounded(x: f32) -> f32 {
    gelu_with_cap::<false>(x)
}

fn gelu_with_cap<const CAPPED: bool>(x: f32) -> f32 {
    let z = (x * FRAC_1_SQRT_2).abs();
    let mut p = GELU_A[5];
    for &a in GELU_A[..5].iter().rev() {
        p = p.mul_add(z, a);
    }
    p = p.mul_add(z, 1.0);
    let p2 = p * p;
    let p4 = p2 * p2;
    let p8 = p4 * p4;
    let p16 = p8 * p8;
    let r = 1.0 / if CAPPED { p16.min(GELU_CAP) } else { p16 };
    (0.5 * x) * if x >= 0.0 { 2.0 - r } else { r }
}

/// x * sigmoid(x); a large negative input gives a zero of its sign.
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// x / step for a subnormal step, whose reciprocal would overflow; quotients
/// of L steps stay within L(1 + 2^-22).
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn on_step(x: f32, step: f32) -> f32 {
    if step.is_normal() {
        x * (1.0 / step)
    } else {
        x / step
    }
}

/// Replaces `row` with e^((x - max) scale) and returns their sum.
pub(crate) fn softmax(row: &mut [f32], scale: f32) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in row {
        *x = ((*x - max) * scale).exp();
        sum += *x;
    }
    sum
}

/// The SIMD kernels' e^x constants.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
))]
mod exp {
    /// ln 2 split for range reduction: the high part has 9 significant bits, so
    /// n * LN2_HIGH is exact for the n exp can meet.
    pub(super) const LN2_HIGH: f32 = 355.0 / 512.0;
    pub(super) const LN2_LOW: f32 = -2.121_944_4e-4;

    /// 1/7! down to 1/0!, in Horner order.
    pub(super) const EXP_TAYLOR: [f32; 8] = [
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ];

    /// Adding 1.5 * 2^23 rounds |x| < 2^22 to an integer, nearest-even as
    /// _mm256_round_ps does: bits(x + ROUNDER) = bits(ROUNDER) + round(x).
    pub(super) const ROUNDER: f32 = 12_582_912.0;
}

/// Softmax exponents never pass max * scale's rounding error: at most 64 for |max * scale| <= 2^30.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
))]
fn below_88(max: f32, scale: f32) -> bool {
    (max * scale).abs() <= 1_073_741_824.0
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use super::exp::{EXP_TAYLOR, LN2_HIGH, LN2_LOW, ROUNDER};
    use super::{FRAC_1_SQRT_2, GELU_A, GELU_CAP};
    use std::arch::x86_64::*;

    const ROUNDER_BITS: i32 = 0x4B40_0000;

    #[target_feature(enable = "avx2,f16c")]
    pub(crate) fn round_to_fp16(values: &mut [f32]) {
        let (chunks, tail) = values.as_chunks_mut::<8>();
        for chunk in chunks {
            // SAFETY: chunk holds eight values.
            unsafe {
                let half =
                    _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(chunk.as_ptr()));
                _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_cvtph_ps(half));
            }
        }
        for x in tail {
            *x = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(
                _mm_set_ss(*x),
            )));
        }
    }

    /// |x| must be below 2^22.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) fn round_to_i32(x: __m256) -> __m256i {
        let shifted = _mm256_add_ps(x, _mm256_set1_ps(ROUNDER));
        _mm256_sub_epi32(
            _mm256_castps_si256(shifted),
            _mm256_set1_epi32(ROUNDER_BITS),
        )
    }

    /// About one ulp on [-87, 88], 0 below, NaN kept; without UPPER, inputs above 88 are out of
    /// range. N vectors interleave because one vector's steps form a dependent chain.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) fn exp_n<const N: usize, const UPPER: bool>(mut x: [__m256; N]) -> [__m256; N] {
        let low = _mm256_set1_ps(-87.0);
        let rounder = _mm256_set1_ps(ROUNDER);
        // t = x log2 e + ROUNDER holds n in its low bits: t - ROUNDER is n,
        // and (bits(t) + 127) << 23 is 2^n.
        let mut n = [_mm256_setzero_ps(); N];
        let mut power = [_mm256_setzero_ps(); N];
        for ((x, n), power) in x.iter_mut().zip(&mut n).zip(&mut power) {
            let underflow = _mm256_cmp_ps::<_CMP_LT_OQ>(*x, low);
            // The second operand is returned when either is NaN.
            let raised = _mm256_max_ps(low, *x);
            *x = if UPPER {
                _mm256_min_ps(_mm256_set1_ps(88.0), raised)
            } else {
                raised
            };
            let t = _mm256_add_ps(
                _mm256_mul_ps(*x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
                rounder,
            );
            *n = _mm256_sub_ps(t, rounder);
            let exponent = _mm256_add_epi32(_mm256_castps_si256(t), _mm256_set1_epi32(127));
            *power = _mm256_andnot_ps(
                underflow,
                _mm256_castsi256_ps(_mm256_slli_epi32::<23>(exponent)),
            );
        }
        let mut r = x;
        for (r, n) in r.iter_mut().zip(&n) {
            *r = _mm256_fnmadd_ps(*n, _mm256_set1_ps(LN2_HIGH), *r);
        }
        for (r, n) in r.iter_mut().zip(&n) {
            *r = _mm256_fnmadd_ps(*n, _mm256_set1_ps(LN2_LOW), *r);
        }
        let mut p = [_mm256_set1_ps(EXP_TAYLOR[0]); N];
        for &coefficient in &EXP_TAYLOR[1..] {
            for (p, r) in p.iter_mut().zip(&r) {
                *p = _mm256_fmadd_ps(*p, *r, _mm256_set1_ps(coefficient));
            }
        }
        for (p, power) in p.iter_mut().zip(&power) {
            *p = _mm256_mul_ps(*p, *power);
        }
        p
    }

    /// A NaN enters min first so the bound returns, as `f32::min` does.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) fn gelu(x: __m256) -> __m256 {
        let [y] = gelu_with_cap::<true, 1>([x]);
        y
    }

    /// N vectors interleave because one vector's steps form a dependent chain.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) fn gelu_unbounded_n<const N: usize>(x: [__m256; N]) -> [__m256; N] {
        gelu_with_cap::<false, N>(x)
    }

    #[target_feature(enable = "avx2,fma")]
    fn gelu_with_cap<const CAPPED: bool, const N: usize>(x: [__m256; N]) -> [__m256; N] {
        let z = x.map(|x| {
            _mm256_andnot_ps(
                _mm256_set1_ps(-0.0),
                _mm256_mul_ps(x, _mm256_set1_ps(FRAC_1_SQRT_2)),
            )
        });
        let mut p = [_mm256_set1_ps(GELU_A[5]); N];
        for &a in GELU_A[..5].iter().rev() {
            for (p, &z) in p.iter_mut().zip(&z) {
                *p = _mm256_fmadd_ps(*p, z, _mm256_set1_ps(a));
            }
        }
        for (p, &z) in p.iter_mut().zip(&z) {
            *p = _mm256_fmadd_ps(*p, z, _mm256_set1_ps(1.0));
        }
        let p16 = p.map(|p| {
            let p2 = _mm256_mul_ps(p, p);
            let p4 = _mm256_mul_ps(p2, p2);
            let p8 = _mm256_mul_ps(p4, p4);
            _mm256_mul_ps(p8, p8)
        });
        let mut y = x;
        for ((y, &x), &p16) in y.iter_mut().zip(&x).zip(&p16) {
            let denominator = if CAPPED {
                _mm256_min_ps(p16, _mm256_set1_ps(GELU_CAP))
            } else {
                p16
            };
            let r = _mm256_div_ps(_mm256_set1_ps(1.0), denominator);
            let nonnegative = _mm256_cmp_ps::<_CMP_GE_OQ>(x, _mm256_setzero_ps());
            let t = _mm256_blendv_ps(r, _mm256_sub_ps(_mm256_set1_ps(2.0), r), nonnegative);
            *y = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), x), t);
        }
        y
    }

    /// Reciprocals are taken once; lanes with a subnormal step divide.
    #[derive(Clone, Copy)]
    pub(crate) struct Steps {
        step: __m256,
        reciprocal: __m256,
        subnormal: __m256,
        any_subnormal: bool,
    }

    impl Steps {
        #[target_feature(enable = "avx2")]
        pub(crate) fn new(step: __m256) -> Self {
            let magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0), step);
            let subnormal =
                _mm256_cmp_ps::<_CMP_LT_OQ>(magnitude, _mm256_set1_ps(f32::MIN_POSITIVE));
            Self {
                step,
                reciprocal: _mm256_div_ps(_mm256_set1_ps(1.0), step),
                subnormal,
                any_subnormal: _mm256_movemask_ps(subnormal) != 0,
            }
        }

        #[target_feature(enable = "avx2")]
        pub(crate) fn on(self, x: __m256) -> __m256 {
            let times = _mm256_mul_ps(x, self.reciprocal);
            if self.any_subnormal {
                _mm256_blendv_ps(times, _mm256_div_ps(x, self.step), self.subnormal)
            } else {
                times
            }
        }
    }

    /// Quads of exponentials overlap; the sum keeps its order, and the max is exact in any order.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) fn softmax(row: &mut [f32], scale: f32) -> f32 {
        assert_eq!(row.len() % 8, 0, "whole vectors");
        let vectors = row.as_chunks_mut::<8>().0;
        let mut most = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
        for (at, chunk) in vectors.iter().enumerate() {
            // SAFETY: the chunk holds eight values.
            most[at % 4] = _mm256_max_ps(most[at % 4], unsafe { _mm256_loadu_ps(chunk.as_ptr()) });
        }
        let mut m = _mm256_max_ps(
            _mm256_max_ps(most[0], most[1]),
            _mm256_max_ps(most[2], most[3]),
        );
        m = _mm256_max_ps(m, _mm256_permute2f128_ps::<1>(m, m));
        m = _mm256_max_ps(m, _mm256_permute_ps::<0x4E>(m));
        m = _mm256_max_ps(m, _mm256_permute_ps::<0xB1>(m));
        let max = _mm256_cvtss_f32(m);
        let bounded = super::below_88(max, scale);
        let (scale, shift) = (_mm256_set1_ps(scale), _mm256_set1_ps(-max * scale));
        if bounded {
            exponentials::<false>(vectors, scale, shift)
        } else {
            exponentials::<true>(vectors, scale, shift)
        }
    }

    /// Each value x becomes e^(x scale + shift); returns their sum.
    #[target_feature(enable = "avx2,fma")]
    fn exponentials<const UPPER: bool>(
        vectors: &mut [[f32; 8]],
        scale: __m256,
        shift: __m256,
    ) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let (quads, rest) = vectors.as_chunks_mut::<4>();
        for quad in quads {
            let x = std::array::from_fn(|i| {
                // SAFETY: each chunk holds eight values.
                _mm256_fmadd_ps(unsafe { _mm256_loadu_ps(quad[i].as_ptr()) }, scale, shift)
            });
            for (chunk, e) in quad.iter_mut().zip(exp_n::<4, UPPER>(x)) {
                // SAFETY: the chunk holds eight values.
                unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), e) };
                sum = _mm256_add_ps(sum, e);
            }
        }
        for chunk in rest {
            // SAFETY: the chunk holds eight values.
            let x = _mm256_fmadd_ps(unsafe { _mm256_loadu_ps(chunk.as_ptr()) }, scale, shift);
            let [e] = exp_n::<1, UPPER>([x]);
            // SAFETY: as above.
            unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), e) };
            sum = _mm256_add_ps(sum, e);
        }
        lanes(sum).into_iter().sum()
    }

    #[target_feature(enable = "avx2")]
    fn lanes(x: __m256) -> [f32; 8] {
        let mut values = [0.0; 8];
        // SAFETY: values holds eight floats.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr(), x) };
        values
    }
}

/// The avx2 module's operations in its order: an eight-lane vector is two
/// four-lane halves, so every result keeps its bits.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod neon {
    use super::exp::{EXP_TAYLOR, LN2_HIGH, LN2_LOW, ROUNDER};
    use super::{FRAC_1_SQRT_2, GELU_A};
    use std::arch::aarch64::*;

    /// _mm256_max_ps: `a` when greater, else `b`, so `b` when either is NaN.
    #[target_feature(enable = "neon")]
    fn max(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vbslq_f32(vcgtq_f32(a, b), a, b)
    }

    /// _mm256_min_ps: `a` when less, else `b`.
    #[target_feature(enable = "neon")]
    fn min(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vbslq_f32(vcltq_f32(a, b), a, b)
    }

    #[target_feature(enable = "neon")]
    fn halves(chunk: &[f32; 8]) -> [float32x4_t; 2] {
        // SAFETY: the chunk holds eight values.
        unsafe { [vld1q_f32(chunk.as_ptr()), vld1q_f32(chunk.as_ptr().add(4))] }
    }

    #[target_feature(enable = "neon")]
    fn store(chunk: &mut [f32; 8], [low, high]: [float32x4_t; 2]) {
        // SAFETY: as above.
        unsafe {
            vst1q_f32(chunk.as_mut_ptr(), low);
            vst1q_f32(chunk.as_mut_ptr().add(4), high);
        }
    }

    /// avx2::exp_n.
    #[target_feature(enable = "neon")]
    pub(crate) fn exp_n<const N: usize, const UPPER: bool>(
        mut x: [float32x4_t; N],
    ) -> [float32x4_t; N] {
        let low = vdupq_n_f32(-87.0);
        let rounder = vdupq_n_f32(ROUNDER);
        let mut n = [vdupq_n_f32(0.0); N];
        let mut power = [vdupq_n_f32(0.0); N];
        for ((x, n), power) in x.iter_mut().zip(&mut n).zip(&mut power) {
            let underflow = vcltq_f32(*x, low);
            let raised = max(low, *x);
            *x = if UPPER {
                min(vdupq_n_f32(88.0), raised)
            } else {
                raised
            };
            let t = vaddq_f32(
                vmulq_f32(*x, vdupq_n_f32(std::f32::consts::LOG2_E)),
                rounder,
            );
            *n = vsubq_f32(t, rounder);
            let exponent = vaddq_s32(vreinterpretq_s32_f32(t), vdupq_n_s32(127));
            let bits = vreinterpretq_u32_s32(vshlq_n_s32::<23>(exponent));
            *power = vreinterpretq_f32_u32(vbicq_u32(bits, underflow));
        }
        let mut r = x;
        for (r, n) in r.iter_mut().zip(&n) {
            *r = vfmsq_f32(*r, *n, vdupq_n_f32(LN2_HIGH));
        }
        for (r, n) in r.iter_mut().zip(&n) {
            *r = vfmsq_f32(*r, *n, vdupq_n_f32(LN2_LOW));
        }
        let mut p = [vdupq_n_f32(EXP_TAYLOR[0]); N];
        for &coefficient in &EXP_TAYLOR[1..] {
            for (p, r) in p.iter_mut().zip(&r) {
                *p = vfmaq_f32(vdupq_n_f32(coefficient), *p, *r);
            }
        }
        for (p, power) in p.iter_mut().zip(&power) {
            *p = vmulq_f32(*p, *power);
        }
        p
    }

    /// avx2::gelu_unbounded_n for one vector.
    #[target_feature(enable = "neon")]
    pub(crate) fn gelu_unbounded(x: float32x4_t) -> float32x4_t {
        let z = vabsq_f32(vmulq_f32(x, vdupq_n_f32(FRAC_1_SQRT_2)));
        let mut p = vdupq_n_f32(GELU_A[5]);
        for &a in GELU_A[..5].iter().rev() {
            p = vfmaq_f32(vdupq_n_f32(a), p, z);
        }
        p = vfmaq_f32(vdupq_n_f32(1.0), p, z);
        let p2 = vmulq_f32(p, p);
        let p4 = vmulq_f32(p2, p2);
        let p8 = vmulq_f32(p4, p4);
        let p16 = vmulq_f32(p8, p8);
        let r = vdivq_f32(vdupq_n_f32(1.0), p16);
        let nonnegative = vcgeq_f32(x, vdupq_n_f32(0.0));
        let t = vbslq_f32(nonnegative, vsubq_f32(vdupq_n_f32(2.0), r), r);
        vmulq_f32(vmulq_f32(vdupq_n_f32(0.5), x), t)
    }

    /// avx2::softmax; grouping chunks differently leaves every lane's operations unchanged.
    #[target_feature(enable = "neon")]
    pub(crate) fn softmax(row: &mut [f32], scale: f32) -> f32 {
        assert_eq!(row.len() % 8, 0, "whole vectors");
        let vectors = row.as_chunks_mut::<8>().0;
        let mut most = [[vdupq_n_f32(f32::NEG_INFINITY); 2]; 4];
        for (at, chunk) in vectors.iter().enumerate() {
            for (most, half) in most[at % 4].iter_mut().zip(halves(chunk)) {
                *most = max(*most, half);
            }
        }
        let [m0, m1, m2, m3] = most;
        let low_half = max(max(m0[0], m1[0]), max(m2[0], m3[0]));
        let high_half = max(max(m0[1], m1[1]), max(m2[1], m3[1]));
        // AVX2 swaps the 128-bit halves, then pairs, then neighbors, and reads lane 0.
        let mut low = max(low_half, high_half);
        low = max(low, vextq_f32::<2>(low, low));
        low = max(low, vrev64q_f32(low));
        let largest = vgetq_lane_f32::<0>(low);
        let bounded = super::below_88(largest, scale);
        let (scale, shift) = (vdupq_n_f32(scale), vdupq_n_f32(-largest * scale));
        if bounded {
            exponentials::<false>(vectors, scale, shift)
        } else {
            exponentials::<true>(vectors, scale, shift)
        }
    }

    /// avx2::exponentials.
    #[target_feature(enable = "neon")]
    fn exponentials<const UPPER: bool>(
        vectors: &mut [[f32; 8]],
        scale: float32x4_t,
        shift: float32x4_t,
    ) -> f32 {
        let mut sum = [vdupq_n_f32(0.0); 2];
        let (pairs, rest) = vectors.as_chunks_mut::<2>();
        for pair in pairs {
            let [a, b] = halves(&pair[0]);
            let [c, d] = halves(&pair[1]);
            let [a, b, c, d] = exp_n::<4, UPPER>([
                vfmaq_f32(shift, a, scale),
                vfmaq_f32(shift, b, scale),
                vfmaq_f32(shift, c, scale),
                vfmaq_f32(shift, d, scale),
            ]);
            for (chunk, e) in pair.iter_mut().zip([[a, b], [c, d]]) {
                store(chunk, e);
                sum = [vaddq_f32(sum[0], e[0]), vaddq_f32(sum[1], e[1])];
            }
        }
        for chunk in rest {
            let [low, high] = halves(chunk);
            let e =
                exp_n::<2, UPPER>([vfmaq_f32(shift, low, scale), vfmaq_f32(shift, high, scale)]);
            store(chunk, e);
            sum = [vaddq_f32(sum[0], e[0]), vaddq_f32(sum[1], e[1])];
        }
        let mut lanes = [0.0; 8];
        store(&mut lanes, sum);
        lanes.into_iter().sum()
    }
}

#[cfg(all(
    test,
    any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )
))]
mod tests;

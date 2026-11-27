//! Each SIMD kernel against one scalar recipe, bit for bit: the kernels then match each other.

#[cfg(target_arch = "x86_64")]
mod x86;

/// This processor's SIMD kernel behind one interface.
#[cfg(target_arch = "x86_64")]
mod simd {
    use crate::math::avx2;
    use std::arch::x86_64::*;

    pub(super) fn supported() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    pub(super) fn exp32<const UPPER: bool>(x: &[f32; 32]) -> [f32; 32] {
        assert!(supported());
        // SAFETY: AVX2 and FMA are present (asserted).
        unsafe { exp32_avx2::<UPPER>(x) }
    }

    #[target_feature(enable = "avx2,fma")]
    fn exp32_avx2<const UPPER: bool>(x: &[f32; 32]) -> [f32; 32] {
        let mut vectors = [_mm256_setzero_ps(); 4];
        for (v, x) in vectors.iter_mut().zip(x.as_chunks::<8>().0) {
            // SAFETY: x holds eight values.
            *v = unsafe { _mm256_loadu_ps(x.as_ptr()) };
        }
        let mut out = [0.0; 32];
        for (e, o) in avx2::exp_n::<4, UPPER>(vectors)
            .iter()
            .zip(out.as_chunks_mut::<8>().0)
        {
            // SAFETY: o holds eight values.
            unsafe { _mm256_storeu_ps(o.as_mut_ptr(), *e) };
        }
        out
    }

    pub(super) fn softmax(row: &mut [f32], scale: f32) -> f32 {
        assert!(supported());
        // SAFETY: as above.
        unsafe { avx2::softmax(row, scale) }
    }

    pub(super) fn gelu_unbounded8(x: &[f32; 8]) -> [f32; 8] {
        assert!(supported());
        // SAFETY: as above.
        unsafe { gelu_unbounded8_avx2(x) }
    }

    #[target_feature(enable = "avx2,fma")]
    fn gelu_unbounded8_avx2(x: &[f32; 8]) -> [f32; 8] {
        let mut out = [0.0; 8];
        // SAFETY: x and out hold eight values each.
        unsafe {
            let [value] = avx2::gelu_unbounded_n([_mm256_loadu_ps(x.as_ptr())]);
            _mm256_storeu_ps(out.as_mut_ptr(), value);
        }
        out
    }
}

/// This processor's SIMD kernel behind one interface.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod simd {
    use crate::math::neon;
    use std::arch::aarch64::*;

    pub(super) fn supported() -> bool {
        true
    }

    #[target_feature(enable = "neon")]
    fn load<const N: usize>(x: &[f32]) -> [float32x4_t; N] {
        assert_eq!(x.len(), 4 * N);
        // SAFETY: x holds 4 N values (asserted).
        std::array::from_fn(|i| unsafe { vld1q_f32(x.as_ptr().add(4 * i)) })
    }

    #[target_feature(enable = "neon")]
    fn save<const N: usize>(vectors: [float32x4_t; N], out: &mut [f32]) {
        assert_eq!(out.len(), 4 * N);
        for (i, v) in vectors.into_iter().enumerate() {
            // SAFETY: out holds 4 N values (asserted).
            unsafe { vst1q_f32(out.as_mut_ptr().add(4 * i), v) };
        }
    }

    pub(super) fn exp32<const UPPER: bool>(x: &[f32; 32]) -> [f32; 32] {
        // SAFETY: this module builds only for targets with NEON.
        unsafe { exp32_neon::<UPPER>(x) }
    }

    #[target_feature(enable = "neon")]
    fn exp32_neon<const UPPER: bool>(x: &[f32; 32]) -> [f32; 32] {
        let mut out = [0.0; 32];
        save(neon::exp_n::<8, UPPER>(load::<8>(x)), &mut out);
        out
    }

    pub(super) fn softmax(row: &mut [f32], scale: f32) -> f32 {
        // SAFETY: as above.
        unsafe { neon::softmax(row, scale) }
    }

    pub(super) fn gelu_unbounded8(x: &[f32; 8]) -> [f32; 8] {
        // SAFETY: as above.
        unsafe { gelu_unbounded8_neon(x) }
    }

    #[target_feature(enable = "neon")]
    fn gelu_unbounded8_neon(x: &[f32; 8]) -> [f32; 8] {
        let mut out = [0.0; 8];
        let [low, high] = load::<2>(x);
        save(
            [neon::gelu_unbounded(low), neon::gelu_unbounded(high)],
            &mut out,
        );
        out
    }
}

/// An independent exp: explicit rounding, integer exponent, underflow mask last.
fn exp_reference(x: f32) -> f32 {
    let underflow = x < -87.0;
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round_ties_even();
    let r = (-n).mul_add(355.0 / 512.0, x);
    let r = (-n).mul_add(-2.121_944_4e-4, r);
    let mut p = 1.0_f32 / 5040.0;
    for coefficient in [
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ] {
        p = p.mul_add(r, coefficient);
    }
    let value = p * f32::from_bits(((n as i32 + 127) << 23) as u32);
    if underflow { 0.0 } else { value }
}

/// A grid, the near-ties of n's rounding, clamp edges, zeros, infinities, the least subnormal.
fn exp_inputs() -> Vec<f32> {
    let mut inputs: Vec<f32> = (0..100.0_f32.to_bits())
        .step_by(4099)
        .flat_map(|bits| [f32::from_bits(bits), -f32::from_bits(bits)])
        .collect();
    for k in -130..=130 {
        let tie = (k as f32 + 0.5) / std::f32::consts::LOG2_E;
        let (mut below, mut above) = (tie, tie);
        inputs.push(tie);
        for _ in 0..8 {
            below = below.next_down();
            above = above.next_up();
            inputs.extend([below, above]);
        }
    }
    for edge in [-87.0_f32, 88.0] {
        inputs.extend([edge.next_down(), edge, edge.next_up()]);
    }
    inputs.extend([
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
    ]);
    inputs
}

#[test]
fn exp_matches_its_original_recipe_bit_for_bit() {
    if !simd::supported() {
        return;
    }
    let mut inputs = exp_inputs();
    inputs.resize(inputs.len().next_multiple_of(32), 1.0);
    for chunk in inputs.as_chunks::<32>().0 {
        let actual = simd::exp32::<true>(chunk);
        for (&x, e) in chunk.iter().zip(actual) {
            let expected = exp_reference(x);
            assert_eq!(e.to_bits(), expected.to_bits(), "e^{x}: {e} vs {expected}");
        }
    }
    let nan = simd::exp32::<true>(&[f32::NAN; 32]);
    assert!(nan.iter().all(|e| e.is_nan()));
}

#[test]
fn exp_without_its_upper_clamp_matches_up_to_88() {
    if !simd::supported() {
        return;
    }
    let mut inputs: Vec<f32> = exp_inputs().into_iter().filter(|&x| x <= 88.0).collect();
    inputs.resize(inputs.len().next_multiple_of(32), 1.0);
    for chunk in inputs.as_chunks::<32>().0 {
        for (&x, e) in chunk.iter().zip(simd::exp32::<false>(chunk)) {
            let expected = exp_reference(x);
            assert_eq!(e.to_bits(), expected.to_bits(), "e^{x}: {e} vs {expected}");
        }
    }
    let nan = simd::exp32::<false>(&[f32::NAN; 32]);
    assert!(nan.iter().all(|e| e.is_nan()));
}

/// Softmax reference with one vector at a time and eight running lane sums.
fn softmax_reference(row: &mut [f32], scale: f32) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shift = -max * scale;
    let mut lanes = [0.0_f32; 8];
    for (j, x) in row.iter_mut().enumerate() {
        *x = exp_reference(x.mul_add(scale, shift));
        lanes[j % 8] += *x;
    }
    lanes.into_iter().sum()
}

#[test]
fn softmax_matches_its_original_recipe_bit_for_bit() {
    if !simd::supported() {
        return;
    }
    let mut state = 7_u64;
    // A spread of 6e11 puts |max * scale| past 2^30, where exponents pass 88 and exp clamps them.
    for (spread, len) in [60.0, 6e11]
        .into_iter()
        .flat_map(|spread| [8, 16, 24, 32, 40, 56, 64, 72, 200, 512].map(|len| (spread, len)))
    {
        let row: Vec<f32> = (0..len)
            .map(|j| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if j + 3 > len && len > 8 {
                    f32::NEG_INFINITY
                } else {
                    ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * spread
                }
            })
            .collect();
        let (mut actual, mut expected) = (row.clone(), row);
        let sum = simd::softmax(&mut actual, 0.125);
        let expected_sum = softmax_reference(&mut expected, 0.125);
        assert_eq!(
            sum.to_bits(),
            expected_sum.to_bits(),
            "spread {spread}, len {len} sum"
        );
        for (at, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "spread {spread}, len {len}, at {at}: {a} vs {e}"
            );
        }
    }
}

/// A grid plus zeros, infinities, extremes and the least subnormal.
fn gelu_inputs() -> Vec<f32> {
    let mut inputs: Vec<f32> = (0..=f32::INFINITY.to_bits())
        .step_by(4099)
        .flat_map(|bits| [f32::from_bits(bits), -f32::from_bits(bits)])
        .collect();
    inputs.extend([
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        f32::from_bits(1),
    ]);
    inputs.resize(inputs.len().next_multiple_of(8), 1.0);
    inputs
}

#[test]
fn unbounded_gelu_vector_matches_the_scalar_bit_for_bit() {
    if !simd::supported() {
        return;
    }
    let mut inputs = gelu_inputs();
    inputs.extend([f32::NAN; 8]);
    for chunk in inputs.as_chunks::<8>().0 {
        let actual = simd::gelu_unbounded8(chunk);
        for (&x, g) in chunk.iter().zip(actual) {
            let expected = super::gelu_unbounded(x);
            assert!(
                g.to_bits() == expected.to_bits() || (g.is_nan() && expected.is_nan()),
                "gelu({x}): {g} vs {expected}"
            );
        }
    }
}

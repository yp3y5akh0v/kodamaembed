use super::{gelu_inputs, simd::supported};
use crate::math::{avx2, gelu, on_step};
use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
fn gelu8(x: &[f32; 8]) -> [f32; 8] {
    let mut out = [0.0; 8];
    // SAFETY: x and out hold eight values each.
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), avx2::gelu(_mm256_loadu_ps(x.as_ptr()))) };
    out
}

#[test]
fn int8_gelu_keeps_its_capped_negative_tail() {
    let x = -2.0_f32.powi(100);
    assert_eq!(gelu(x).to_bits(), (-0.5_f32).to_bits());
    if supported() {
        let mut out = [f32::NAN; 8];
        crate::gelus(&[x; 8], &mut out);
        assert!(
            out.iter()
                .all(|value| value.to_bits() == (-0.5_f32).to_bits())
        );
    }
}

#[test]
fn gelu_vector_matches_the_scalar_bit_for_bit() {
    if !supported() {
        return;
    }
    let mut inputs = gelu_inputs();
    inputs.extend([f32::NAN; 8]);
    for chunk in inputs.as_chunks::<8>().0 {
        // SAFETY: support was checked above.
        let actual = unsafe { gelu8(chunk) };
        for (&x, g) in chunk.iter().zip(actual) {
            let expected = gelu(x);
            assert!(
                g.to_bits() == expected.to_bits() || (g.is_nan() && expected.is_nan()),
                "gelu({x}): {g} vs {expected}"
            );
        }
    }
}

/// erf to about 1e-14: Taylor below 2.5, the erfc continued fraction above.
fn erf_reference(x: f64) -> f64 {
    if x.abs() < 2.5 {
        let (mut term, mut sum, mut n) = (x, x, 0.0);
        while term.abs() > 1e-17 * sum.abs().max(1e-300) {
            n += 1.0;
            term *= -x * x / n;
            sum += term / (2.0 * n + 1.0);
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        let z = x.abs();
        let mut fraction = 0.0;
        for n in (1..60).rev() {
            fraction = f64::from(n) / 2.0 / (z + fraction);
        }
        let erfc = (-z * z).exp() / std::f64::consts::PI.sqrt() / (z + fraction);
        (1.0 - erfc).copysign(x)
    }
}

/// erf's 3e-7 bound gives GELU 1.5e-7 |x|, rounding about 1.5e-7 |x| + 3.2e-7
/// more: 4e-7 max(|x|, 1) is asserted.
#[test]
fn gelu_is_within_its_bound_of_the_exact_gelu() {
    let mut worst = (0.0_f64, 0.0_f32);
    for x in gelu_inputs().into_iter().filter(|x| x.abs() <= 10.0) {
        let exact =
            0.5 * f64::from(x) * (1.0 + erf_reference(f64::from(x) / std::f64::consts::SQRT_2));
        let error = (f64::from(gelu(x)) - exact).abs();
        assert!(
            error <= 4e-7 * f64::from(x.abs()).max(1.0),
            "gelu({x}) = {} vs {exact} (error {error:e})",
            gelu(x)
        );
        if error > worst.0 {
            worst = (error, x);
        }
    }
    eprintln!("largest error {:e} at {}", worst.0, worst.1);
}

#[target_feature(enable = "avx2,fma")]
fn on_steps8(x: &[f32; 8], steps: &[f32; 8]) -> [f32; 8] {
    let mut out = [0.0; 8];
    // SAFETY: x, steps and out hold eight values each.
    unsafe {
        let rule = avx2::Steps::new(_mm256_loadu_ps(steps.as_ptr()));
        _mm256_storeu_ps(out.as_mut_ptr(), rule.on(_mm256_loadu_ps(x.as_ptr())));
    }
    out
}

/// Normal steps multiply by the reciprocal, subnormal steps divide, mixed or alone.
#[test]
fn steps_take_each_lane_onto_its_own_step() {
    if !supported() {
        return;
    }
    let steps = [
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE.next_down(),
        f32::from_bits(2),
        3.0 / 127.0,
        0.1,
        7.0,
        1.0e30,
        2.5e-30,
    ];
    let mut state = 11_u64;
    for _ in 0..4096 {
        let x: [f32; 8] = std::array::from_fn(|lane| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
            unit * steps[lane] * 127.0
        });
        // SAFETY: support was checked above.
        let mixed = unsafe { on_steps8(&x, &steps) };
        for lane in 0..8 {
            let expected = on_step(x[lane], steps[lane]);
            assert_eq!(
                mixed[lane].to_bits(),
                expected.to_bits(),
                "lane {lane}: {}",
                x[lane]
            );
            // SAFETY: support was checked above.
            let alone = unsafe { on_steps8(&[x[lane]; 8], &[steps[lane]; 8]) };
            assert_eq!(alone[0].to_bits(), expected.to_bits(), "alone, lane {lane}");
        }
    }
}

#[target_feature(enable = "avx2,fma")]
fn round8(x: &[f32; 8]) -> [i32; 8] {
    let mut out = [0; 8];
    // SAFETY: x and out hold eight values each.
    unsafe {
        let rounded = avx2::round_to_i32(_mm256_loadu_ps(x.as_ptr()));
        _mm256_storeu_si256(out.as_mut_ptr().cast(), rounded);
    }
    out
}

#[test]
fn round_to_i32_rounds_ties_to_even() {
    if !supported() {
        return;
    }
    let mut inputs: Vec<f32> = (-1200..=1200).map(|half| half as f32 * 0.5).collect();
    inputs.extend((0..(1 << 22) as u32).step_by(997).flat_map(|i| {
        let x = i as f32 + 0.5;
        [x, -x, x.next_down(), -x.next_up()]
    }));
    inputs.extend([
        4_194_303.5_f32,
        -4_194_303.5,
        0.499_999_97,
        -0.499_999_97,
        -0.0,
    ]);
    inputs.resize(inputs.len().next_multiple_of(8), 0.0);
    for chunk in inputs.as_chunks::<8>().0 {
        // SAFETY: support was checked above.
        let actual = unsafe { round8(chunk) };
        for (&x, q) in chunk.iter().zip(actual) {
            assert_eq!(q, x.round_ties_even() as i32, "{x}");
        }
    }
}

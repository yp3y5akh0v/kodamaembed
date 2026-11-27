//! CPU backend for KodamaEmbed: FP32 by default, an opt-in INT8 preset
//! ([`CpuPrecision`]), and every output independent of its batch.

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/crest.svg",
    html_favicon_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/favicon.ico"
)]

mod attention;
mod encoder;
mod gemm;
#[cfg(target_arch = "x86_64")]
mod int8;
mod math;
mod precision;
mod projection;
mod workers;

pub use attention::attention;
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
pub use attention::integer_attention;
#[doc(hidden)]
pub use encoder::pool_rows;
pub use encoder::{Encoder, EncoderError, TokenSequence};
pub use gemm::{Epilogue, Kernel, PackedWeight, gemm};
pub use precision::CpuPrecision;
pub use workers::{Workers, worker_pool};

/// Equal lengths divisible by eight; needs AVX2 and FMA.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
pub fn gelus(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
    assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
    assert!(
        x.len() == out.len() && x.len().is_multiple_of(8),
        "whole vectors"
    );
    for (x, out) in x.as_chunks::<8>().0.iter().zip(out.as_chunks_mut::<8>().0) {
        // SAFETY: AVX2 and FMA were detected; x and out hold eight values.
        unsafe {
            _mm256_storeu_ps(
                out.as_mut_ptr(),
                math::avx2::gelu(_mm256_loadu_ps(x.as_ptr())),
            )
        };
    }
}

/// Equal lengths divisible by eight; needs AVX2 and FMA.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
pub fn weight_indices(e: &[f32], out: &mut [u16]) {
    attention::weight_indices_of(e, out);
}

/// Hashed with the [`Kernel`] into fingerprints, so any change to the arithmetic
/// invalidates stored vectors and caches.
pub const NUMERICAL_SOURCES: &[&[u8]] = &[
    include_bytes!("encoder.rs"),
    include_bytes!("gemm.rs"),
    include_bytes!("math.rs"),
    include_bytes!("attention.rs"),
    include_bytes!("attention/integer.rs"),
    include_bytes!("workers.rs"),
    include_bytes!("lib.rs"),
    include_bytes!("precision.rs"),
    include_bytes!("projection.rs"),
    include_bytes!("int8/mod.rs"),
    include_bytes!("int8/prepared.rs"),
    include_bytes!("int8/prepared/avx2.rs"),
    include_bytes!("int8/activation_avx2.rs"),
    include_bytes!("int8/e5_profile.rs"),
    include_bytes!("int8/e5_v1_signs.bin"),
    include_bytes!("int8/fused_epilogue.rs"),
    include_bytes!("int8/gptq.rs"),
];

#[cfg(test)]
mod tests {
    use super::NUMERICAL_SOURCES;

    /// The manifest stays out of the fingerprint: `cargo publish` rewrites it.
    #[test]
    fn fingerprint_inputs_exclude_the_manifest() {
        let manifest: &[u8] = include_bytes!("../Cargo.toml");
        assert!(!NUMERICAL_SOURCES.contains(&manifest));
    }

    /// Every source outside the tests is hashed, so editing any of them changes the fingerprint.
    #[test]
    fn every_source_is_a_fingerprint_input() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut dirs = vec![src];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                let name = path.file_name().unwrap().to_str().unwrap();
                if name == "tests" || name == "tests.rs" || name.ends_with("_tests.rs") {
                    continue;
                }
                if path.is_dir() {
                    dirs.push(path);
                } else {
                    let source = std::fs::read(&path).unwrap();
                    assert!(
                        NUMERICAL_SOURCES.contains(&source.as_slice()),
                        "{} is not hashed",
                        path.display()
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
use crate::int8;
use crate::{EncoderError, Kernel};
use kodamaembed_model::{Output, Pooling, Positions, Shape};

/// Numerical policy for CPU projections. The INT8 preset accepts only its
/// evaluated shape; only the `kodamaembed` loader verifies the exact checkpoint.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CpuPrecision {
    /// FP32 projection kernels supporting all CPU model shapes.
    #[default]
    Fp32,
    /// E5-large-v1's centered INT8, batch-invariant through static centers; it needs the AVX2
    /// [`Kernel`], AVX-VNNI, F16C and [`crate::Encoder::calibrate`] before encoding.
    E5CenteredInt8,
}

impl CpuPrecision {
    pub(crate) fn validate(
        self,
        kernel: Kernel,
        shape: Shape,
        output: Output,
    ) -> Result<(), EncoderError> {
        if self == Self::Fp32 {
            return Ok(());
        }
        if output != Output::Embedding(Pooling::Mean)
            || shape.layers != 24
            || shape.hidden != 1024
            || shape.heads != 16
            || shape.intermediate != 4096
            || shape.vocab != 30_522
            || shape.max_positions != 512
            || shape.type_vocab != 2
            || shape.eps.to_bits() != 1e-12_f32.to_bits()
            || shape.positions != Positions::Sequential
        {
            return Err(EncoderError::Unsupported(
                "the E5 INT8 preset requires the evaluated E5-large-v1 shape and mean-pooled output"
                    .into(),
            ));
        }
        #[cfg(target_arch = "x86_64")]
        let supported_kernel = kernel == Kernel::Avx2 && int8::is_supported();
        #[cfg(not(target_arch = "x86_64"))]
        let supported_kernel = false;
        if !supported_kernel || !kernel.is_supported() {
            return Err(EncoderError::Unsupported(format!(
                "{self:?} requires AVX2/FMA, AVX-VNNI and F16C with Kernel::Avx2"
            )));
        }
        Ok(())
    }

    pub(crate) fn centered(self) -> bool {
        self == Self::E5CenteredInt8
    }

    pub(crate) fn integer_attention(self) -> bool {
        self == Self::E5CenteredInt8
    }

    pub(crate) fn identity(self) -> &'static str {
        match self {
            Self::Fp32 => "cpu-fp32-v1",
            Self::E5CenteredInt8 => {
                "cpu-e5-centered-int8-v1:half-rne:h128:w8-gptq-damp10:a8-rne:sha256-signs-k:static3-b128-chain-h-shared-exact-f64-deferred-ln:calibration-fp32-text:attention-qk8-w14-v8-sinks-static-keys:half-residual:avx2-f16c-avxvnni:row-major6x16:fused-epilogue-v1"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kodamaembed_model::{Head, Positions};
    fn e5() -> Shape {
        Shape {
            layers: 24,
            hidden: 1024,
            heads: 16,
            intermediate: 4096,
            vocab: 30522,
            max_positions: 512,
            type_vocab: 2,
            eps: 1e-12,
            positions: Positions::Sequential,
        }
    }
    #[test]
    fn e5_precision_is_distinct_and_scope_is_strict() {
        let p = CpuPrecision::E5CenteredInt8;
        assert_ne!(p.identity(), CpuPrecision::Fp32.identity());
        assert!(
            p.validate(Kernel::Portable, e5(), Output::Embedding(Pooling::Mean))
                .is_err()
        );
        assert!(
            p.validate(Kernel::detect(), e5(), Output::Relevance(Head::Pooler))
                .is_err()
        );
        let mut other = e5();
        other.layers = 6;
        assert!(
            p.validate(Kernel::detect(), other, Output::Embedding(Pooling::Mean))
                .is_err()
        );
        let mut other = e5();
        other.hidden = 384;
        assert!(
            p.validate(Kernel::detect(), other, Output::Embedding(Pooling::Mean))
                .is_err()
        );
        #[cfg(target_arch = "x86_64")]
        if int8::is_supported() && Kernel::Avx2.is_supported() {
            eprintln!(
                "native CPU: kernel={:?}, int8_supported={}, workers={}",
                Kernel::detect(),
                int8::is_supported(),
                crate::workers::POOL.current_num_threads()
            );
            assert!(
                p.validate(Kernel::Avx2, e5(), Output::Embedding(Pooling::Mean))
                    .is_ok()
            );
        }
    }

    #[test]
    fn every_preset_has_its_own_identity_and_the_centered_ones_center() {
        use CpuPrecision::*;
        assert_ne!(Fp32.identity(), E5CenteredInt8.identity());
        for (precision, centered) in [(Fp32, false), (E5CenteredInt8, true)] {
            assert_eq!(precision.centered(), centered, "{precision:?}");
            assert_eq!(precision.integer_attention(), centered, "{precision:?}");
        }
    }

    #[test]
    fn default_stays_fp32() {
        assert_eq!(CpuPrecision::default(), CpuPrecision::Fp32);
        assert_eq!(CpuPrecision::Fp32.identity(), "cpu-fp32-v1");
    }
}

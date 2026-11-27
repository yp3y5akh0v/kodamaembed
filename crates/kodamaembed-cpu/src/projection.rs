#[cfg(target_arch = "x86_64")]
use crate::{
    CpuPrecision,
    int8::{DeferredNorm, Operand, PreparedWeight, e5_profile},
};
use crate::{
    EncoderError,
    gemm::{Epilogue, Kernel, PackedWeight, gemm_current_pool},
};
#[cfg(target_arch = "x86_64")]
use kodamaembed_model::repro::Kind;

pub(crate) enum Projection {
    Float(PackedWeight),
    #[cfg(target_arch = "x86_64")]
    Int8(Box<PreparedWeight>),
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn signs(precision: CpuPrecision, layer: usize, kind: Kind) -> Vec<f32> {
    match precision {
        CpuPrecision::E5CenteredInt8 => e5_profile::signs(layer, kind),
        CpuPrecision::Fp32 => unreachable!("FP32 projections are not rotated"),
    }
}

#[cfg(target_arch = "x86_64")]
fn operand_digits(precision: CpuPrecision, layer: usize, kind: Kind) -> usize {
    match precision {
        CpuPrecision::E5CenteredInt8 => e5_profile::operand_digits(layer, kind),
        CpuPrecision::Fp32 => unreachable!("FP32 projections have no digits"),
    }
}

impl Projection {
    /// The centered preset stays FP32 until calibration quantizes it.
    pub(crate) fn new(raw: &[f32], n: usize, k: usize) -> Self {
        Self::Float(PackedWeight::new(raw, n, k))
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn calibrated(
        &self,
        layer: usize,
        kind: Kind,
        precision: CpuPrecision,
        operand: Operand<'_>,
        norm: Option<DeferredNorm<'_>>,
    ) -> Result<Self, EncoderError> {
        let Self::Float(w) = self else {
            return Err(EncoderError::Unsupported(
                "calibration quantizes FP32 weights".into(),
            ));
        };
        PreparedWeight::calibrated(
            &w.rows(),
            w.n(),
            w.k(),
            &signs(precision, layer, kind),
            operand,
            norm,
            operand_digits(precision, layer, kind),
        )
        .map(|w| Self::Int8(Box::new(w)))
        .map_err(|e| {
            EncoderError::Input(format!(
                "layer {layer} {} could not be calibrated: {e}",
                kind.name()
            ))
        })
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn write_calibration(&self, out: &mut Vec<u8>) -> Result<(), EncoderError> {
        match self {
            Self::Int8(w) => w
                .write_tables(out)
                .map_err(|e| EncoderError::Unsupported(format!("calibration tables: {e}"))),
            Self::Float(_) => Err(EncoderError::Unsupported(
                "calibration tables need calibrated INT8 weights".into(),
            )),
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn installed(
        &self,
        layer: usize,
        kind: Kind,
        precision: CpuPrecision,
        gains: Option<&[f32]>,
        tables: &mut &[u8],
    ) -> Result<Self, EncoderError> {
        let Self::Float(w) = self else {
            return Err(EncoderError::Unsupported(
                "calibration tables install on FP32 weights".into(),
            ));
        };
        PreparedWeight::from_tables(
            w.n(),
            w.k(),
            &signs(precision, layer, kind),
            gains,
            tables,
            operand_digits(precision, layer, kind),
        )
        .map(|w| Self::Int8(Box::new(w)))
        .map_err(|e| {
            EncoderError::Input(format!(
                "layer {layer} {} calibration tables: {e}",
                kind.name()
            ))
        })
    }

    pub(crate) fn n(&self) -> usize {
        match self {
            Self::Float(w) => w.n(),
            #[cfg(target_arch = "x86_64")]
            Self::Int8(w) => w.n(),
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn k(&self) -> usize {
        match self {
            Self::Float(w) => w.k(),
            Self::Int8(w) => w.k(),
        }
    }

    pub(crate) fn defers(&self) -> bool {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Int8(w) => w.defers(),
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_deferred(
        &self,
        kernel: Kernel,
        input: &[f32],
        m: usize,
        row_classes: &[u32],
        mean: &[f32],
        rstd: &[f32],
        gelu: bool,
        segment: usize,
        output: &mut [f32],
    ) -> Result<(), EncoderError> {
        #[cfg(target_arch = "x86_64")]
        if let Self::Int8(w) = self {
            if kernel != Kernel::Avx2 {
                return Err(EncoderError::Unsupported(
                    "deferred INT8 projections require Kernel::Avx2".into(),
                ));
            }
            return w
                .project_deferred_into(input, m, row_classes, mean, rstd, gelu, segment, output)
                .map_err(|e| {
                    EncoderError::Input(format!(
                        "deferred INT8 projection could not represent these activations: {e}"
                    ))
                });
        }
        let _ = (
            kernel,
            input,
            m,
            row_classes,
            mean,
            rstd,
            gelu,
            segment,
            output,
        );
        Err(EncoderError::Unsupported(
            "deferred projections require INT8 weights".into(),
        ))
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn export(
        &self,
        bias: &[f32],
    ) -> Result<kodamaembed_model::repro::ProjectionExport, EncoderError> {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Int8(w) => w
                .export(bias)
                .map_err(|e| EncoderError::Unsupported(format!("projection export: {e}"))),
            _ => {
                let _ = bias;
                Err(EncoderError::Unsupported(
                    "exports need centered INT8 weights".into(),
                ))
            }
        }
    }

    pub(crate) fn apply(
        &self,
        kernel: Kernel,
        input: &[f32],
        m: usize,
        epilogue: Epilogue<'_>,
        output: &mut [f32],
    ) -> Result<(), EncoderError> {
        match self {
            Self::Float(w) => {
                gemm_current_pool(kernel, input, m, w, epilogue, output);
                Ok(())
            }
            #[cfg(target_arch = "x86_64")]
            Self::Int8(_) => Err(EncoderError::Unsupported(
                "INT8 projections run on their restored or deferred paths".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_restored(
        &self,
        kernel: Kernel,
        input: &[f32],
        m: usize,
        row_classes: &[u32],
        segment: usize,
        epilogue: Epilogue<'_>,
        output: &mut [f32],
    ) -> Result<(), EncoderError> {
        #[cfg(target_arch = "x86_64")]
        if let Self::Int8(w) = self {
            if kernel != Kernel::Avx2 {
                return Err(EncoderError::Unsupported(
                    "restored INT8 projections require Kernel::Avx2".into(),
                ));
            }
            return w
                .project_restored_into(input, m, row_classes, segment, epilogue, output)
                .map_err(|e| {
                    EncoderError::Input(format!(
                        "restored INT8 projection could not represent these activations: {e}"
                    ))
                });
        }
        let _ = (kernel, input, m, row_classes, segment, epilogue, output);
        Err(EncoderError::Unsupported(
            "restored projections require INT8 weights".into(),
        ))
    }
}

#[cfg(test)]
mod route_tests {
    use super::*;

    #[test]
    fn restored_route_rejects_fp32_weights_before_writes() {
        let weights = [0.5, -0.25, 0.125, -0.5, 0.25, -0.125];
        let projection = Projection::new(&weights, 2, 3);
        let input = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let bias = [0.25, -0.25];
        let mut untouched = [42.0; 4];
        for kernel in [Kernel::Portable, Kernel::detect()] {
            let error = projection
                .apply_restored(
                    kernel,
                    &input,
                    2,
                    &[0, 1],
                    3,
                    Epilogue::Bias(&bias),
                    &mut untouched,
                )
                .unwrap_err();
            assert!(matches!(error, EncoderError::Unsupported(ref message)
                if message == "restored projections require INT8 weights"));
            assert_eq!(untouched, [42.0; 4]);
        }
    }
}

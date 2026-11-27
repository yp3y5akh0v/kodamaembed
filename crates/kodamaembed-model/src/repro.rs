//! Calibrated centered INT8 parameters every backend runs bit-identically, each row on its own.

use crate::Shape;

/// Static classes in center-table order: first token, last token, interior.
pub const STATIC_CLASSES: usize = 3;

/// A layer's four projections.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Qkv,
    Out,
    Ffn1,
    Ffn2,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Qkv => "qkv",
            Self::Out => "out",
            Self::Ffn1 => "ffn1",
            Self::Ffn2 => "ffn2",
        }
    }
}

/// 0 for the first token, 1 for the last, 2 for interior ones; `t < len`.
pub fn restored_class(t: usize, len: usize) -> usize {
    if t == 0 {
        0
    } else if t + 1 == len {
        1
    } else {
        2
    }
}

/// One centered INT8 projection `y = W x + b`, `W` of shape `[n, k]`.
#[derive(Clone)]
pub struct ProjectionExport {
    pub n: usize,
    pub k: usize,
    /// INT8 digits of the rotated weights, row-major `[n][k]`.
    pub q: Vec<i8>,
    /// Each output row's scale `[n]`, its INT8 digits' step.
    pub scales: Vec<f32>,
    /// The rotation's signs `[k]`, each +1 or -1.
    pub signs: Vec<f32>,
    /// The digits' operand's static centers `[STATIC_CLASSES][k]`, FP16
    /// values: of the input, or of `gamma z` when the LayerNorm is deferred.
    pub centers: Vec<f32>,
    /// Each static center's product on the exact weights, no bias, `[classes][n]`.
    pub center_shared: Vec<f32>,
    pub bias: Vec<f32>,
    /// The operand's INT8 digits: 1, or 2 where a second digit on the step over 254 follows
    /// the first and the dot walks the weights again in the same chain.
    pub operand_digits: usize,
    /// Parameters for a projection whose LayerNorm is deferred to its
    /// epilogue (the QKV projection past layer 0, the up projection).
    pub deferred: Option<DeferredExport>,
}

/// A projection whose LayerNorm runs in its epilogue,
/// `fma(fma(-mean, sums, u), rstd, constants)` on the restored dot `u`.
#[derive(Clone)]
pub struct DeferredExport {
    /// `W gamma` and `W beta + b` `[n]`.
    pub sums: Vec<f32>,
    pub constants: Vec<f32>,
}

/// A LayerNorm's gain, bias and epsilon.
#[derive(Clone)]
pub struct NormExport {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub eps: f32,
}

#[derive(Clone)]
pub struct LayerExport {
    pub qkv: ProjectionExport,
    /// Integer attention's static key centers `[heads][d]`: keys are centered
    /// on these, and values rounded to FP16.
    pub key_centers: Vec<f32>,
    pub out: ProjectionExport,
    pub norm1: NormExport,
    pub up: ProjectionExport,
    pub down: ProjectionExport,
    pub norm2: NormExport,
}

/// A mean-pooling encoder.
#[derive(Clone)]
pub struct EncoderExport {
    pub shape: Shape,
    /// `[vocab][hidden]`.
    pub word: Vec<f32>,
    /// `[max_positions][hidden]`.
    pub position: Vec<f32>,
    /// `[type_vocab][hidden]`.
    pub token_type: Vec<f32>,
    pub embedding_norm: NormExport,
    pub layers: Vec<LayerExport>,
    /// The CPU's f32 constants in order: 1 / sqrt(128), GELU's six Abramowitz
    /// and Stegun 7.1.28 coefficients and its cap, 1 / sqrt 2.
    pub constants_f32: Vec<f32>,
    /// Integer attention's weight of each exponent -u in units of 2^-8,
    /// RNE(16383 2^(-u / 256)) for u in [0, 3840], the last entry zero.
    pub attention_weights: Vec<u16>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restored_classes_are_first_last_and_between() {
        assert_eq!(restored_class(0, 1), 0);
        assert_eq!([0, 1].map(|t| restored_class(t, 2)), [0, 1]);
        assert_eq!([0, 1, 2, 3].map(|t| restored_class(t, 4)), [0, 2, 2, 1]);
    }
}

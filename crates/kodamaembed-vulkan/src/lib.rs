//! Vulkan 1.3 backend for KodamaEmbed, its shaders compiled offline to embedded SPIR-V.

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/crest.svg",
    html_favicon_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/favicon.ico"
)]

pub mod device;
pub mod encoder;
pub mod pipeline;

pub use device::{Adapter, Binding, Buffer, Commands, Device, DeviceRepr, Submission, VulkanError};
pub use encoder::{Encoder, EncoderError};
pub use pipeline::{Descriptors, Kernel, KernelSpec, Push, Recorder};

/// Hashed into fingerprints: every shader and the host code driving them, so
/// even a source-only edit changes the fingerprint.
pub const NUMERICAL_SOURCES: &[&[u8]] = &[
    shaders::LAYERNORM,
    shaders::GEMM_MATRIX,
    shaders::GEMM_ROWS,
    shaders::GEMM_PORTABLE,
    shaders::GEMM_MATRIX_PAIRED,
    shaders::GEMM_ROWS_PAIRED,
    shaders::GEMM_PORTABLE_PAIRED,
    shaders::ATTENTION,
    shaders::ATTENTION_MATRIX,
    shaders::ATTENTION_BIAS,
    shaders::ATTENTION_MATRIX_BIAS,
    shaders::ATTENTION_LOCAL,
    shaders::ATTENTION_MATRIX_LOCAL,
    shaders::EMBED,
    shaders::POOL,
    shaders::RERANK,
    shaders::RERANK_PREDICTION,
    include_bytes!("encoder.rs"),
    include_bytes!("pipeline.rs"),
    include_bytes!("device.rs"),
    include_bytes!("lib.rs"),
];

/// The row shaders' specialization constant 0 is the workgroup size, a
/// quarter of the model width.
pub mod shaders {
    /// Bindings x, gamma, beta, out16, out32; push rows, write32, eps.
    pub const LAYERNORM: &[u8] = include_bytes!("../spv/layernorm.spv");
    /// Bindings A, W, bias, residual, out16, out32; push M, N, K; specialization
    /// 0 the epilogue, 1 the bias epilogue's low columns.
    pub const GEMM_MATRIX: &[u8] = include_bytes!("../spv/gemm_matrix.spv");
    /// GEMM_MATRIX's bindings and bits in 16-row tiles, for rows too few to fill
    /// the GPU with its tiles; specialization 2 the K stage.
    pub const GEMM_ROWS: &[u8] = include_bytes!("../spv/gemm_rows.spv");
    /// The same GEMM on plain FP32 arithmetic, for GPUs without cooperative matrices.
    pub const GEMM_PORTABLE: &[u8] = include_bytes!("../spv/gemm_portable.spv");
    /// GEMM_MATRIX over column pairs: epilogue 3 gates, 4 is the bias epilogue with its low columns
    /// turned; bindings 6 positions, 7 the turns by position and pair; specialization 3 the head.
    pub const GEMM_MATRIX_PAIRED: &[u8] = include_bytes!("../spv/gemm_matrix_paired.spv");
    /// GEMM_ROWS with GEMM_MATRIX_PAIRED's epilogues, bindings and specialization.
    pub const GEMM_ROWS_PAIRED: &[u8] = include_bytes!("../spv/gemm_rows_paired.spv");
    /// GEMM_PORTABLE with GEMM_MATRIX_PAIRED's epilogues, bindings and specialization.
    pub const GEMM_PORTABLE_PAIRED: &[u8] = include_bytes!("../spv/gemm_portable_paired.spv");
    /// Bindings qkv, Q and K low parts (FP32), cu_seqlens, out; push scale;
    /// specialization head dimension, heads.
    pub const ATTENTION: &[u8] = include_bytes!("../spv/attention.spv");
    /// The same attention on cooperative matrices, in subgroups of 32.
    pub const ATTENTION_MATRIX: &[u8] = include_bytes!("../spv/attention_matrix.spv");
    /// ATTENTION plus MPNet's relative-position bias: binding 4 the table, push scale, longest.
    pub const ATTENTION_BIAS: &[u8] = include_bytes!("../spv/attention_bias.spv");
    /// ATTENTION_MATRIX plus the bias, bound and pushed as in ATTENTION_BIAS.
    pub const ATTENTION_MATRIX_BIAS: &[u8] = include_bytes!("../spv/attention_matrix_bias.spv");
    /// ATTENTION over the keys within `reach` rows of each query: push scale, reach.
    pub const ATTENTION_LOCAL: &[u8] = include_bytes!("../spv/attention_local.spv");
    /// ATTENTION_MATRIX inside the same window, pushed as in ATTENTION_LOCAL.
    pub const ATTENTION_MATRIX_LOCAL: &[u8] = include_bytes!("../spv/attention_matrix_local.spv");
    /// Bindings ids, types, positions, word, position, token type, gamma, beta,
    /// out16, out32; push tokens, eps.
    pub const EMBED: &[u8] = include_bytes!("../spv/embed.spv");
    /// Bindings hidden (FP32), cu_seqlens, out.
    pub const POOL: &[u8] = include_bytes!("../spv/pool.spv");
    /// Bindings hidden (FP32), cu_seqlens, pooler weight, pooler bias,
    /// classifier weight, classifier bias, scores; specialization width.
    pub const RERANK: &[u8] = include_bytes!("../spv/rerank.spv");
    /// A prediction head. RERANK's bindings with the dense weight and the norm's scale for the
    /// pooler's; push eps; specialization width, whether the rows' mean is pooled, whether SiLU.
    pub const RERANK_PREDICTION: &[u8] = include_bytes!("../spv/rerank_prediction.spv");
}

#[cfg(test)]
mod tests {
    use super::NUMERICAL_SOURCES;

    /// The manifest stays out of the fingerprint: `cargo publish` rewrites it.
    #[test]
    fn fingerprint_inputs_exclude_the_manifest() {
        let manifest: &[u8] = include_bytes!("../Cargo.toml");
        assert!(!NUMERICAL_SOURCES.contains(&manifest));
    }

    /// Every shipped shader is hashed, so editing any of them changes the fingerprint.
    #[test]
    fn every_shader_is_a_fingerprint_input() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("spv");
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let shader = std::fs::read(&path).unwrap();
            assert!(
                NUMERICAL_SOURCES.contains(&shader.as_slice()),
                "{} is not hashed",
                path.display()
            );
        }
    }
}

# kodamaembed-cpu

CPU kernels and encoders for transformer inference. FP32 is the default, with
AVX2/FMA kernels on x86-64 and NEON kernels on 64-bit ARM when supported and
portable Rust otherwise; the NEON kernels repeat the AVX2 kernels' operations
in the same order and give the same FP32 bits. The encoder provides
attention, mean and `[CLS]` pooling, normalization and one-logit relevance
heads. It runs BERT's layers and, in FP32, ModernBERT's: rotary positions,
local attention windows and gated feed-forward layers.

[`kodamaembed`](https://crates.io/crates/kodamaembed) loads checkpoints and
tokenizes text; this crate's tensor-level APIs validate shapes and arithmetic
boundaries.

`Encoder::new_with_precision` selects E5-large's centered INT8 preset, which
needs x86-64 with AVX-VNNI, AVX2, FMA and F16C; reranking runs in FP32.
`Encoder::calibrate` prepares its tables, and `Encoder::export_reproducible`
exports the calibrated encoder for the CUDA backend, which reproduces its
vector bits.
[Numerical modes](https://github.com/yp3y5akh0v/kodamaembed/blob/master/docs/architecture.md#numerical-modes)
describes the arithmetic.

```text
cargo test -p kodamaembed-cpu --locked
cargo test -p kodamaembed-cpu --examples --locked
cargo doc -p kodamaembed-cpu --no-deps --open
```

The examples measure GEMM, attention and encoder execution.

[Apache-2.0](https://github.com/yp3y5akh0v/kodamaembed/blob/master/LICENSE).

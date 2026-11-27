# kodamaembed-cuda

CUDA kernels and encoders for transformer inference, shipped as PTX. They
need an NVIDIA driver and compute capability 8.0 or newer, and take hidden
width/head count 384/12, 768/12 or 1024/16 with a four-times-wider
feed-forward layer, or with a ModernBERT feed-forward layer whose width is a
multiple of 64.

`Encoder` runs FP16 tensor products with FP32 accumulation. Rounding a query
or key to FP16 drops a small remainder, which the encoder stores beside the
FP16 value; attention scores read both in sequences under 256 tokens and in
ModernBERT's windowed layers.

`repro::ReproEncoder` runs a calibrated
`kodamaembed_model::repro::EncoderExport` from the CPU backend: E5-large's
centered INT8 preset, with the CPU encoder's vector bits. A batch of up to 16
encoded tokens in at most four sequences runs as one persistent launch on
GPUs with at least 40 streaming multiprocessors, and other batches run
per-layer kernels;
[CUDA execution](https://github.com/yp3y5akh0v/kodamaembed/blob/master/docs/architecture.md#cuda-execution)
has the details.

[`kodamaembed`](https://crates.io/crates/kodamaembed) loads models, tokenizes
text and selects these encoders. The low-level API documents context
ownership, allocation lifetimes and the unsafe `DeviceRepr` and kernel-launch
contracts.

Run the GPU tests one process at a time. `--include-ignored` adds the
every-length comparisons with the CPU and the exhaustive arithmetic checks;
on a GPU with fewer than 40 multiprocessors, also pass
`--skip e5_stream_engine`.

```text
cargo test --release -p kodamaembed-cuda --locked -- --test-threads=1
cargo test --release -p kodamaembed-cuda --locked -- --test-threads=1 --include-ignored
```

[Apache-2.0](https://github.com/yp3y5akh0v/kodamaembed/blob/master/LICENSE).

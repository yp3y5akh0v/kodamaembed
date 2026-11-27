# kodamaembed-vulkan

Vulkan kernels and encoders for transformer inference, shipped as SPIR-V.
They need Vulkan 1.3 with the features the
[integration guide](https://github.com/yp3y5akh0v/kodamaembed/blob/master/docs/integration.md#device-behavior)
lists, run on cooperative matrices where supported and FP32 shaders
otherwise, and take hidden width/head count 384/12, 768/12 or 1024/16 with a
four-times-wider feed-forward layer, or with a ModernBERT feed-forward layer
whose width is a multiple of 64.
[`kodamaembed`](https://crates.io/crates/kodamaembed) loads checkpoints,
tokenizes text and streams output.

The low-level API documents device ownership and the unsafe shader,
descriptor and command contracts. An unresolved submission failure
quarantines the device and keeps its resources until the process exits, and
later inference on it fails.

Run the GPU tests one process at a time. They open every Vulkan adapter on
the machine, and nine need cooperative matrices; `--skip where_supported`
skips those:

```text
cargo test --release -p kodamaembed-vulkan --locked -- --test-threads=1
cargo doc -p kodamaembed-vulkan --no-deps --open
```

[Apache-2.0](https://github.com/yp3y5akh0v/kodamaembed/blob/master/LICENSE).

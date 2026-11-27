# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-26

### Added

- Text embeddings and reranking on CPUs and on GPUs, through CUDA at compute
  capability 8.0 or newer or through Vulkan 1.3, for 20 embedding models and
  5 rerankers.
- Vectors that do not depend on batch size or input order, and a fingerprint
  of the model files, input handling and numerics behind them.
- Quantized E5-large, with the same vectors on the processor and CUDA.
- The Python package for Linux, macOS and Windows, with a command line and an
  HTTP server for OpenAI-compatible embeddings (`/v1/embeddings`) and Cohere-
  and Jina-compatible reranking (`/v1/rerank`).
- The `kodamaembed` Rust crate and its model, CPU, CUDA and Vulkan crates.
- Model downloads pinned to one commit and checked against SHA-256, in the
  shared Hugging Face cache, with offline use.
- Local checkpoints, loaded from a directory or from bytes in memory.
- Ordered streams for large inputs, which the caller can stop between
  batches.
- Loaded models that threads share without a lock, their calls taking turns
  on the device batch by batch.
- A build for `wasm32-unknown-unknown`, where models load from bytes and run
  on the calling thread.

[0.1.0]: https://github.com/yp3y5akh0v/kodamaembed/releases/tag/v0.1.0

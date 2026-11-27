<h1><img src="https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/logo.svg" alt="KodamaEmbed" width="420"></h1>

Text embeddings and reranking on CPUs and GPUs. This crate loads models by
name or from local checkpoints, tokenizes text, packs variable-length
sequences and delivers ordered output.

```rust,no_run
use kodamaembed::{Embedder, EmbeddingModel};

fn main() -> Result<(), kodamaembed::Error> {
    let embedder = Embedder::new(EmbeddingModel::AllMiniLmL6V2)?;
    let vectors = embedder.embed_passages(&["An example passage."])?;
    assert_eq!(vectors[0].len(), 384);
    Ok(())
}
```

Debug builds compile dependencies unoptimized. To keep the CPU kernels fast
without `--release`, add to the root `Cargo.toml`:

```toml
[profile.dev.package.kodamaembed-cpu]
opt-level = 3
```

Models load by name from the Hugging Face cache, downloading on first use
with the default `download` feature; each is pinned to one commit, and every
file is checked against its SHA-256. `Embedder::builder()` and
`Reranker::builder()` set the device, precision, batch size and worker
counts. Their `build_from_dir` loads a local checkpoint with `config.json`,
`tokenizer.json` and an F32, F16 or BF16 `model.safetensors`, and
`build_from_bytes` loads the same three files from memory.

`Device::Auto` takes CUDA at compute capability 8.0 or newer, else a discrete
Vulkan GPU, else the CPU. `Precision::Quantized` runs E5-large's centered
INT8 preset, with the same vectors and fingerprint on the CPU and CUDA.
`embed_passages_stream`, `embed_queries_stream` and `score_stream` deliver
ordered batches, and their `_until` forms stop between batches on request.
Threads share one loaded model, and their calls take turns batch by batch.
With `default-features = false` the crate builds for
`wasm32-unknown-unknown`, where a model loads from bytes and runs on the
calling thread.

`tokenizers` and `half` are pinned to exact versions, so vectors stay
bit-identical across builds.

The
[integration guide](https://github.com/yp3y5akh0v/kodamaembed/blob/master/docs/integration.md)
covers devices, precision, caches, worker counts and errors, and the
[benchmarks](https://github.com/yp3y5akh0v/kodamaembed/blob/master/docs/benchmarks.md)
cover latency and retrieval quality. The API reference is on
[docs.rs](https://docs.rs/kodamaembed).

[Apache-2.0](https://github.com/yp3y5akh0v/kodamaembed/blob/master/LICENSE);
models and datasets have separate licenses.

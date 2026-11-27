<p align="center">
  <img src="docs/assets/logo.svg" alt="KodamaEmbed" width="420">
</p>

<p align="center">
  Text embeddings and reranking for Python and Rust, on CPUs and GPUs.
</p>

<p align="center">
  <a href="https://crates.io/crates/kodamaembed"><img src="https://badgen.net/crates/v/kodamaembed" alt="crates.io"></a>
  <a href="https://pypi.org/project/kodamaembed/"><img src="https://img.shields.io/pypi/v/kodamaembed?label=pypi" alt="PyPI"></a>
  <br>
  <a href="https://github.com/yp3y5akh0v/kodamaembed/actions/workflows/ci.yml"><img src="https://github.com/yp3y5akh0v/kodamaembed/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="docs/benchmarks.md#embedding-latency"><img src="https://img.shields.io/badge/quantized%20E5--large%20vs%20TensorRT%20FP16-1.07--2.15x%20faster-success" alt="Quantized E5-large: 1.07 to 2.15 times as fast as TensorRT FP16 at 16 to 512 tokens on an RTX 4000 SFF Ada"></a>
  <a href="docs/benchmarks.md#correctness"><img src="https://img.shields.io/badge/CPU%20and%20CUDA%20vectors-bit--identical-success" alt="Quantized E5-large returns the same vector bits on the CPU and on CUDA"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

KodamaEmbed runs on CPUs, on NVIDIA GPUs through CUDA and on Vulkan 1.3 GPUs.
It loads models by name into the Hugging Face cache, or from local
checkpoints, packs variable-length sequences without padding and returns
L2-normalized vectors or relevance logits. Its GPU kernels ship as PTX and
SPIR-V, which the driver compiles at load. Quantized E5-large computes exact
integer dot products and accumulates them in a fixed FP32 order, so the CPU
and CUDA backends return the same vector bits for every input and batch. Its
quantization builds on [QuaRot](https://arxiv.org/abs/2404.00456) and
[GPTQ](https://arxiv.org/abs/2210.17323).

## Quickstart

### Python

```text
pip install kodamaembed
```

```python
from kodamaembed import Embedder

model = Embedder("sentence-transformers/all-MiniLM-L6-v2")
passages = model.embed_passages(["Berlin is the capital of Germany."])
queries = model.embed_queries(["capital of Germany"])
print(queries @ passages.T)
```

The wheel holds the engine and its GPU kernels; numpy is its only dependency.
It also installs the `kodamaembed` command, which embeds texts and serves
an OpenAI-compatible embeddings endpoint and a reranking endpoint that accepts
Cohere and Jina requests. See the [Python guide](docs/python.md).

### Rust

With Rust 1.98 or newer, add the crate. Debug builds compile dependencies
unoptimized; the profile entry keeps the CPU kernels fast without
`--release`:

```toml
[dependencies]
kodamaembed = "0.1"

[profile.dev.package.kodamaembed-cpu]
opt-level = 3
```

```rust
use kodamaembed::{Embedder, EmbeddingModel};

fn main() -> Result<(), kodamaembed::Error> {
    // Downloads on first use and checks every file against a pinned hash.
    let embedder = Embedder::new(EmbeddingModel::AllMiniLmL6V2)?;
    let passages = embedder.embed_passages(&["Berlin is the capital of Germany."])?;
    let queries = embedder.embed_queries(&["capital of Germany"])?;
    println!("{} passages, {} queries; {} dimensions on {}",
        passages.len(), queries.len(), embedder.dim(), embedder.device_name());
    Ok(())
}
```

| To | Use |
|---|---|
| Load a model by name with the defaults | `Embedder::new(EmbeddingModel::AllMiniLmL6V2)` |
| Pick the device | `Embedder::builder().device(Device::Cpu)`, `Device::Cuda(index)` or `Device::Vulkan(index)` |
| Run E5-large quantized (centered INT8) on the CPU or CUDA | `.precision(Precision::Quantized)` |
| Set CPU worker counts | `.workers(CpuWorkers { .. })` |
| Use another cache, or stay offline | `.build_from_hub(&Hub::new(dir).offline(true), model)` |
| Load a local checkpoint | `.build_from_dir(dir, ModelSpec { .. })` |
| Load a checkpoint held in memory | `.build_from_bytes(config, tokenizer, weights, ModelSpec { .. })` |
| Embed or score a large input in ordered batches | `embed_passages_stream`, `embed_queries_stream`, `Reranker::score_stream` |
| Stop such a run between batches | `embed_passages_until`, `embed_queries_until`, `Reranker::score_until` |
| Serve several threads from one loaded model | a shared `&Embedder` or `Arc<Embedder>`; calls take turns batch by batch |
| Build for `wasm32-unknown-unknown` | `default-features = false`, then `build_from_bytes`; the model runs on the calling thread |

`Device::Auto`, the default, takes CUDA at compute capability 8.0 or newer,
else a discrete Vulkan GPU, else the CPU. Models download into the Hugging
Face cache that Python tools share, and `HF_HUB_CACHE`, `HF_HOME`,
`HF_HUB_OFFLINE` and `HF_ENDPOINT` apply. The
[integration guide](docs/integration.md) covers the details.

## Supported models and devices

| Model | Name in the API | Dimensions | Tokens | Download |
|---|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | `EmbeddingModel::BgeBaseEnV1_5` | 768 | 512 | 438.7 MB |
| `BAAI/bge-large-en-v1.5` | `EmbeddingModel::BgeLargeEnV1_5` | 1024 | 512 | 1.3 GB |
| `BAAI/bge-m3` | `EmbeddingModel::BgeM3` | 1024 | 512 | 2.3 GB |
| `BAAI/bge-small-en-v1.5` | `EmbeddingModel::BgeSmallEnV1_5` | 384 | 512 | 134.2 MB |
| `ibm-granite/granite-embedding-english-r2` | `EmbeddingModel::GraniteEmbeddingEnglishR2` | 768 | 8192 | 301.6 MB |
| `intfloat/e5-large` | `EmbeddingModel::E5Large` | 1024 | 512 | 1.3 GB |
| `intfloat/e5-large-v2` | `EmbeddingModel::E5LargeV2` | 1024 | 512 | 1.3 GB |
| `intfloat/e5-small-v2` | `EmbeddingModel::E5SmallV2` | 384 | 512 | 134.2 MB |
| `intfloat/multilingual-e5-base` | `EmbeddingModel::MultilingualE5Base` | 768 | 512 | 1.1 GB |
| `intfloat/multilingual-e5-large` | `EmbeddingModel::MultilingualE5Large` | 1024 | 512 | 2.3 GB |
| `intfloat/multilingual-e5-small` | `EmbeddingModel::MultilingualE5Small` | 384 | 512 | 487.7 MB |
| `mixedbread-ai/mxbai-embed-large-v1` | `EmbeddingModel::MxbaiEmbedLargeV1` | 1024 | 512 | 671.0 MB |
| `nomic-ai/modernbert-embed-base` | `EmbeddingModel::ModernBertEmbedBase` | 768 | 8192 | 599.7 MB |
| `sentence-transformers/all-MiniLM-L12-v2` | `EmbeddingModel::AllMiniLmL12V2` | 384 | 128 | 133.9 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | `EmbeddingModel::AllMiniLmL6V2` | 384 | 256 | 91.3 MB |
| `sentence-transformers/all-mpnet-base-v2` | `EmbeddingModel::AllMpnetBaseV2` | 768 | 384 | 438.4 MB |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `EmbeddingModel::ParaphraseMultilingualMiniLmL12V2` | 384 | 128 | 479.7 MB |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | `EmbeddingModel::ParaphraseMultilingualMpnetBaseV2` | 768 | 128 | 1.1 GB |
| `Snowflake/snowflake-arctic-embed-l` | `EmbeddingModel::SnowflakeArcticEmbedL` | 1024 | 512 | 1.3 GB |
| `Snowflake/snowflake-arctic-embed-m-v1.5` | `EmbeddingModel::SnowflakeArcticEmbedMV1_5` | 768 | 512 | 436.3 MB |
| `Alibaba-NLP/gte-reranker-modernbert-base` | `RerankerModel::GteRerankerModernBertBase` | - | 8192 | 602.0 MB |
| `BAAI/bge-reranker-v2-m3` | `RerankerModel::BgeRerankerV2M3` | - | 512 | 2.3 GB |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | `RerankerModel::MsMarcoMiniLmL12V2` | - | 512 | 134.2 MB |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `RerankerModel::MsMarcoMiniLmL6V2` | - | 512 | 91.6 MB |
| `ibm-granite/granite-embedding-reranker-english-r2` | `RerankerModel::GraniteEmbeddingRerankerEnglishR2` | - | 1124 | 602.0 MB |

Each model is pinned to one upstream commit, and every file is checked
against its SHA-256 before anything parses it. Token counts include special
tokens and prefixes. The multilingual-e5, bge-m3, bge-reranker-v2-m3 and
paraphrase-multilingual models are multilingual; the others are English. Each
model pools as its model card prescribes, the `[CLS]` token for BGE, mxbai,
Arctic and Granite and the mean for E5, modernbert-embed and the
sentence-transformers models, and `embed_queries` and `embed_passages` add
its prefixes. Every vector is L2-normalized, and reranking returns raw logits.
The [reference tests](crates/kodamaembed/tests/registry/reference.rs) compare
all registered models with sentence-transformers' FP32 outputs on one query
and five passages, using L2 distance for embeddings and absolute error for
reranker logits. The tolerances on CPU are 1e-5 for embeddings and 2e-5 for
logits, and 5e-3 for both on CUDA and Vulkan.

| Backend | Requirements | Default arithmetic |
|---|---|---|
| CPU | portable Rust; AVX2/FMA on x86-64 and NEON on 64-bit ARM when available | FP32 |
| CUDA | NVIDIA compute capability 8.0 or newer | FP16 tensor products, FP32 accumulation, corrected attention scores |
| Vulkan | Vulkan 1.3 with the features and limits the shaders need | cooperative matrices where supported, FP32 shaders otherwise |

The GPU kernels take hidden width/head count 384/12, 768/12 or 1024/16 with
a four-times-wider feed-forward layer, or with a ModernBERT feed-forward
layer whose width is a multiple of 64; CPU FP32 accepts other encoder shapes.

Quantized precision runs the exact `intfloat/e5-large` checkpoint; other
models run float precision. An x86-64 processor with AVX-VNNI, AVX2, FMA and
F16C prepares and calibrates it, also for CUDA, and the calibration is cached
for later loads. The CPU and CUDA give the same vector bits and
`fingerprint()`. See [precision selection](docs/integration.md#precision-selection).

## Performance

Single-query latency on an NVIDIA RTX 4000 SFF Ada Generation under its 70 W
power limit, from text to a normalized FP32 vector in host memory, in
milliseconds.
Each value is the median of three process medians; lengths include special
tokens and the query prefix of E5 and modernbert-embed. KodamaEmbed is
measured against ONNX Runtime CUDA, TensorRT and PyTorch SDPA, all FP16. In
every row KodamaEmbed is the fastest engine and TensorRT the second fastest,
so the table shows TensorRT of the three. The last two columns divide
TensorRT's latency by KodamaEmbed's: above 1.00x, KodamaEmbed is faster. Only
float E5-large at 512 tokens is below 1.00x, and quantized E5-large is ahead
of TensorRT there.

| Model | Tokens | KodamaEmbed quantized | KodamaEmbed float | TensorRT | TensorRT / quantized | TensorRT / float |
|---|---:|---:|---:|---:|---:|---:|
| E5-large-v1 | 16 | 1.325 | 2.708 | 2.846 | 2.15x | 1.05x |
| E5-large-v1 | 33 | 1.594 | 2.857 | 2.932 | 1.84x | 1.03x |
| E5-large-v1 | 64 | 1.814 | 2.972 | 3.088 | 1.70x | 1.04x |
| E5-large-v1 | 65 | 2.044 | 3.157 | 3.165 | 1.55x | 1.00x |
| E5-large-v1 | 128 | 2.595 | 3.409 | 3.476 | 1.34x | 1.02x |
| E5-large-v1 | 256 | 4.643 | 4.920 | 4.968 | 1.07x | 1.01x |
| E5-large-v1 | 512 | 7.777 | 8.463 | 8.415 | 1.08x | 0.99x |
| all-MiniLM-L6-v2 | 16 | | 0.109 | 0.218 | | 2.01x |
| all-MiniLM-L6-v2 | 33 | | 0.135 | 0.276 | | 2.05x |
| all-MiniLM-L6-v2 | 64 | | 0.161 | 0.319 | | 1.98x |
| all-MiniLM-L6-v2 | 65 | | 0.194 | 0.337 | | 1.74x |
| all-MiniLM-L6-v2 | 128 | | 0.222 | 0.483 | | 2.17x |
| all-MiniLM-L6-v2 | 256 | | 0.360 | 0.879 | | 2.44x |
| modernbert-embed-base | 16 | | 1.186 | 1.343 | | 1.13x |
| modernbert-embed-base | 33 | | 1.351 | 1.435 | | 1.06x |
| modernbert-embed-base | 64 | | 1.441 | 1.521 | | 1.06x |
| modernbert-embed-base | 65 | | 1.535 | 1.597 | | 1.04x |
| modernbert-embed-base | 128 | | 1.687 | 1.832 | | 1.09x |
| modernbert-embed-base | 256 | | 2.554 | 2.566 | | 1.00x |
| modernbert-embed-base | 512 | | 4.130 | 4.377 | | 1.06x |
| modernbert-embed-base | 1,024 | | 8.280 | 8.451 | | 1.02x |
| modernbert-embed-base | 2,048 | | 17.380 | 19.983 | | 1.15x |
| modernbert-embed-base | 4,096 | | 41.568 | 59.379 | | 1.43x |
| modernbert-embed-base | 8,192 | | 110.816 | 194.187 | | 1.75x |

With a different length on every call from 16 to 512 tokens, KodamaEmbed was
again the fastest: by mean latency on passes two and three, quantized
E5-large was 1.27 times as fast and float E5-large 1.12 times as fast as the
second-fastest engine, ONNX Runtime CUDA. On six efficiency cores of an Intel
Core i5-13500, KodamaEmbed was the fastest at every fixed length tested:
against the second-fastest engine it was 1.80 to 2.38 times as fast with
quantized E5-large, 1.07 to 1.18 times with float E5-large, 1.22 to 1.54
times with float MiniLM and 1.15 to 1.60 times with float
modernbert-embed-base, which ran there up to 2,048 tokens.

The [benchmark guide](docs/benchmarks.md#embedding-latency) has every
engine's results, the protocol, the measurement's
[limits](docs/benchmarks.md#limits) and the reproduction commands.

## Accuracy

Quantized E5-large passed the retrieval-quality gates on all four datasets
and all 13 individual tasks. Compared with FP32, its largest dataset-average
loss in binary-relevance nDCG@10 was 0.00194 on the 10,000-passage FiQA
subset. Its largest individual-task loss was 0.00401 on a LoCoMo conversation,
within the 0.005 limit.
See [retrieval quality](docs/benchmarks.md#retrieval-quality).

## Documentation

- [Quickstart notebook](notebooks/quickstart.ipynb): embedding, search and
  reranking in a few cells.
- [Python](docs/python.md): the Python package, the command line and the HTTP
  server.
- [Migration](docs/migration.md): moving from sentence-transformers, FastEmbed
  or an OpenAI-compatible server.
- [Integration](docs/integration.md): loading, precision, worker counts,
  streaming, errors and fingerprints.
- [Architecture](docs/architecture.md): crate boundaries, execution and
  numerical contracts.
- [Benchmarks and validation](docs/benchmarks.md): latency, correctness,
  retrieval quality and reproduction commands.
- [Technical report](paper/kodamaembed.pdf): the numerical contract, its
  bit-identical implementations and their evaluation.
- [Contributing](CONTRIBUTING.md): development checks and GPU artifact builds.
- API reference: [docs.rs/kodamaembed](https://docs.rs/kodamaembed).

KodamaEmbed is licensed under [Apache-2.0](LICENSE). Model weights and datasets
have their own licenses.

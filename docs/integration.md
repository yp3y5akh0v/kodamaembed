# Integration

## Loading a model

```rust
use kodamaembed::{Embedder, EmbeddingModel, Reranker, RerankerModel};

let embedder = Embedder::new(EmbeddingModel::E5Large)?;
let reranker = Reranker::new(RerankerModel::MsMarcoMiniLmL6V2)?;
```

`Embedder::builder()` and `Reranker::builder()` take options, and a `build`
method chooses where the files come from.

| Method | Files |
|---|---|
| `build(model)` | the Hugging Face cache from the environment, downloading on first use |
| `build_from_hub(&hub, model)` | the cache a `Hub` names, online or offline |
| `build_from_dir(dir, spec)` | a local directory with `config.json`, `tokenizer.json` and `model.safetensors` |
| `build_from_bytes(config, tokenizer, weights, spec)` | the contents of those three files, held in memory |

| Option | Default |
|---|---|
| `device(Device)` | `Device::Auto` |
| `precision(Precision)`, embedders only | `Precision::Float` |
| `batch_tokens(usize)` | 16,384 |
| `workers(CpuWorkers)` | the default pools |

An embedder's `build_from_dir` and `build_from_bytes` take a `ModelSpec` with
the prefixes, truncation and pooling of any BERT, XLM-R or MPNet checkpoint
with GELU and absolute positions, or of a ModernBERT checkpoint with GELU, no
biases and unscaled rotary positions, its configuration as transformers 4 or
5 writes it; `EmbeddingModel::spec` gives a registered model's. A reranker's
take the pairs' maximum length. A ModernBERT reranker scores through the head
its configuration names: the `[CLS]` token or the mean, GELU or SiLU, and a
dense layer without a bias. Weights are F32, or F16 or BF16, which widen to
F32 exactly. The same files give the same vectors and fingerprint from a
directory and from memory.

Embeddings are L2-normalized FP32 rows, pooled as the `ModelSpec` says: the
mean of every token or the `[CLS]` token alone. Reranking returns raw
relevance logits. Outputs keep the input order. A loaded `Embedder` or
`Reranker` is `Send` and `Sync`, and inference takes it by shared reference,
so [threads share one model](#sharing-a-model).

## Models by name

| Model | Name in the API | Pinned commit |
|---|---|---|
| `BAAI/bge-base-en-v1.5` | `EmbeddingModel::BgeBaseEnV1_5` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| `BAAI/bge-large-en-v1.5` | `EmbeddingModel::BgeLargeEnV1_5` | `d4aa6901d3a41ba39fb536a557fa166f842b0e09` |
| `BAAI/bge-m3` | `EmbeddingModel::BgeM3` | `9a0624b896d81da7492a910ffa53731274b6cf3d` |
| `BAAI/bge-small-en-v1.5` | `EmbeddingModel::BgeSmallEnV1_5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| `ibm-granite/granite-embedding-english-r2` | `EmbeddingModel::GraniteEmbeddingEnglishR2` | `47ea694b257b703fee9253d75c2b1f2985180498` |
| `intfloat/e5-large` | `EmbeddingModel::E5Large` | `4dc6d853a804b9c8886ede6dda8a073b7dc08a81` |
| `intfloat/e5-large-v2` | `EmbeddingModel::E5LargeV2` | `f169b11e22de13617baa190a028a32f3493550b6` |
| `intfloat/e5-small-v2` | `EmbeddingModel::E5SmallV2` | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
| `intfloat/multilingual-e5-base` | `EmbeddingModel::MultilingualE5Base` | `d128750597153bb5987e10b1c3493a34e5a4502a` |
| `intfloat/multilingual-e5-large` | `EmbeddingModel::MultilingualE5Large` | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` |
| `intfloat/multilingual-e5-small` | `EmbeddingModel::MultilingualE5Small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| `mixedbread-ai/mxbai-embed-large-v1` | `EmbeddingModel::MxbaiEmbedLargeV1` | `b33106f585b9ce46904ad7443a3b52b7a63e231c` |
| `nomic-ai/modernbert-embed-base` | `EmbeddingModel::ModernBertEmbedBase` | `d556a88e332558790b210f7bdbe87da2fa94a8d8` |
| `sentence-transformers/all-MiniLM-L12-v2` | `EmbeddingModel::AllMiniLmL12V2` | `a50ef00143b4d5391434df20ae11632588ac25be` |
| `sentence-transformers/all-MiniLM-L6-v2` | `EmbeddingModel::AllMiniLmL6V2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `sentence-transformers/all-mpnet-base-v2` | `EmbeddingModel::AllMpnetBaseV2` | `e8c3b32edf5434bc2275fc9bab85f82640a19130` |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `EmbeddingModel::ParaphraseMultilingualMiniLmL12V2` | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | `EmbeddingModel::ParaphraseMultilingualMpnetBaseV2` | `4328cf26390c98c5e3c738b4460a05b95f4911f5` |
| `Snowflake/snowflake-arctic-embed-l` | `EmbeddingModel::SnowflakeArcticEmbedL` | `d8fb21ca8d905d2832ee8b96c894d3298964346b` |
| `Snowflake/snowflake-arctic-embed-m-v1.5` | `EmbeddingModel::SnowflakeArcticEmbedMV1_5` | `e58a8f756156a1293d763f17e3aae643474e9b8a` |
| `Alibaba-NLP/gte-reranker-modernbert-base` | `RerankerModel::GteRerankerModernBertBase` | `f7481e6055501a30fb19d090657df9ec1f79ab2c` |
| `BAAI/bge-reranker-v2-m3` | `RerankerModel::BgeRerankerV2M3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | `RerankerModel::MsMarcoMiniLmL12V2` | `7b0235231ca2674cb8ca8f022859a6eba2b1c968` |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `RerankerModel::MsMarcoMiniLmL6V2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| `ibm-granite/granite-embedding-reranker-english-r2` | `RerankerModel::GraniteEmbeddingRerankerEnglishR2` | `d09d3d6971b689bf9c23839e45a470874d46e13a` |

Files live in the Hugging Face cache layout,
`models--{owner}--{name}/snapshots/{commit}/{file}`, shared with Python's
`huggingface_hub`. `Hub::from_env` finds the cache as `huggingface_hub` does:
`HF_HUB_CACHE`, else `HF_HOME/hub`, else `XDG_CACHE_HOME/huggingface/hub`,
else `~/.cache/huggingface/hub`. `HF_HUB_OFFLINE` set to `1`, `ON`, `YES` or
`TRUE` forbids downloads, and `HF_ENDPOINT` names a mirror. `Hub::new(dir)`
names a cache directly, and `offline(true)` forbids its downloads.

A cached file of the pinned size loads without a network request; a missing
or differently sized file downloads again, or is an error offline. A download
moves into place only once its size and SHA-256 match, and an interrupted
transfer is retried up to three times. Every load checks each file's SHA-256
before parsing it, and a mismatch names the file. Without the default
`download` feature, the build has no HTTP client and loads from the cache.

## Precision selection

| Mode | Selection | Scope |
|---|---|---|
| CPU FP32 | `Precision::Float` on `Device::Cpu` | encoder shapes |
| CUDA FP16 | `Precision::Float` on `Device::Cuda(index)` | hidden width/head count 384/12, 768/12 or 1024/16 |
| Vulkan | `Precision::Float` on `Device::Vulkan(index)` | hidden width/head count 384/12, 768/12 or 1024/16, with the features under [Device behavior](#device-behavior) |
| Quantized (centered INT8) | `Precision::Quantized` on `Device::Cpu`, `Device::Cuda(index)` or `Device::Auto` | the exact E5-large checkpoint below |

`Device::Auto` tries only the devices that support the requested settings:
`Precision::Float` tries CUDA at compute capability 8.0 or newer, a discrete
Vulkan GPU, then the CPU; `Precision::Quantized` tries CUDA, then the CPU;
explicit workers run on the CPU. It passes over a GPU that lacks a feature or
shape the model needs and returns any other failure. Explicit GPU selection
with CPU workers, or Vulkan with quantized precision, fails before model files
are read. The loader checks GPU
availability and model-specific hardware requirements when it opens the
backend.

On CUDA, `Precision::Float` uses FP16 projection operands with FP32 sums.

### Exact INT8 checkpoints

`Precision::Quantized` runs the centered INT8 preset of E5-large; other models
return `Error::Unsupported`. An x86-64 processor with AVX-VNNI, AVX2, FMA and
F16C prepares and calibrates the preset, also when it runs on CUDA, where it
needs compute capability 8.0 or newer. It accepts only the 24-layer
E5-large-v1 model with mean pooling, maximum length 512 and the prefixes
`passage: ` and `query: `, and verifies the bytes it loaded:

| File | SHA-256 |
|---|---|
| `config.json` | `d960925459889922fd6b3078e05d89c5b0fadd0018d26421699066cdd98bfb3a` |
| `tokenizer.json` | `5fd1c882abbd30517dced455a2c9768945ec726b96727927e4959348d9de550b` |
| `model.safetensors` | `93e04bd3ec4911982905395e3c48ee739ddb8aa88380b8ce27f1fd20bfa5fa8e` |

`EmbeddingModel::E5Large` downloads exactly these files, from upstream
revision
[`4dc6d853a804b9c8886ede6dda8a073b7dc08a81`](https://huggingface.co/intfloat/e5-large/tree/4dc6d853a804b9c8886ede6dda8a073b7dc08a81).
A local copy must keep the original bytes.

```rust
use kodamaembed::{Embedder, EmbeddingModel, Precision};

let embedder = Embedder::builder()
    .precision(Precision::Quantized)
    .build(EmbeddingModel::E5Large)?;
```

With `Device::Auto`, the preset runs on CUDA when it can and on the processor
otherwise. CUDA runs the processor's exact arithmetic, so every vector and
the fingerprint are the same on both. A batch of up to 16 encoded tokens in
at most four sequences, special tokens and prefixes included, runs as one
persistent launch on GPUs with at least 40 streaming multiprocessors; other
batches and GPUs run per-layer kernels with the same arithmetic.
[CUDA execution](architecture.md#cuda-execution) has the details.

### Calibration cache

Loading calibrates the preset with an FP32 forward pass and a GPTQ solve per
projection, over 128 passages shipped with the engine, each passage's first
16 words as a query, and 1,024 queries of one to three vocabulary pieces from
a fixed generator; [numerical modes](architecture.md#numerical-modes)
describes the arithmetic. The first load writes the tables, about 309 MB, to
`kodamaembed-sha256-<fingerprint>.calibration` in the `calibration` directory
of KodamaEmbed's cache: `KODAMAEMBED_CACHE` when set, else
`%LOCALAPPDATA%\kodamaembed` on Windows, `~/Library/Caches/kodamaembed` on
macOS, and `XDG_CACHE_HOME/kodamaembed` or `~/.cache/kodamaembed` elsewhere.

Later loads reuse a valid file, with the same vectors and fingerprint, and
rebuild a missing or invalid one; other read and write errors are returned.
Writing needs a writable cache directory, while a valid file also loads from
a read-only one. Files move into place whole, so concurrent loads never read
a partial file. Model directories are never written.

## CPU worker counts

```rust
use std::num::NonZeroUsize;
use kodamaembed::{CpuWorkers, Device, Embedder, EmbeddingModel};

let workers = CpuWorkers {
    inference_workers: NonZeroUsize::new(4).unwrap(),
    tokenizer_workers: NonZeroUsize::new(1).unwrap(),
};
let embedder = Embedder::builder()
    .device(Device::Cpu)
    .batch_tokens(4_096)
    .workers(workers)
    .build(EmbeddingModel::E5Large)?;
assert_eq!(embedder.cpu_workers(), Some(workers));
```

`workers` sets one CPU model's inference and tokenizer worker counts. The
example runs four inference workers and one tokenizer worker, plus the
caller and, when tokenizing ahead, one producer thread. Calls with up to 64
texts tokenize before inference, on the caller and the tokenizer pool;
larger calls, streams and reranking tokenize ahead on the producer thread.
Checkpoint loading and weight preparation run on the caller, and calibration
also uses the inference pool.

Counts must be positive and within Rayon's limit. Explicit pools ignore
`RAYON_NUM_THREADS`, and `cpu_workers()` reports them; builders without
`workers` keep the default pools. `batch_tokens` caps the tokens in one
encoder batch, which sizes the activation memory, and must be at least the
maximum sequence length. `fingerprint()` includes explicit worker counts and
excludes `batch_tokens`.

## Streaming

```rust
use std::convert::Infallible;
use kodamaembed::Embedder;

fn consume(embedder: &Embedder, texts: impl Iterator<Item = String> + Send)
    -> Result<usize, kodamaembed::StreamError<Infallible>>
{
    embedder.embed_passages_stream(texts, |batch| {
        for (offset, vector) in batch.values.chunks_exact(batch.dim).enumerate() {
            let input_index = batch.start + offset;
            // Store or process this row before returning from the callback.
            let _ = (input_index, vector);
        }
        Ok(())
    })
}
```

`Batch.start` is the first input index, `Batch.dim` the row width, and
`Batch.values` is borrowed until the callback returns. `score_stream` has the
same contract with `dim == 1`. An empty input makes no callbacks and returns
zero. Tokenization holds at most four chunks of 64 texts: two queued, one in
inference and one being tokenized. Encoder batches are bounded by token and
sequence limits.

`embed_passages_until`, `embed_queries_until` and `score_until` also take a
`stop` closure, asked before each batch is encoded; `true` ends the run with
`Error::Interrupted`. A smaller `batch_tokens` asks more often.

```rust
use std::time::{Duration, Instant};

let deadline = Instant::now() + Duration::from_secs(2);
let embedded = embedder.embed_passages_until(texts, || Instant::now() >= deadline, sink);
```

## Sharing a model

```rust
use std::thread;

thread::scope(|scope| {
    scope.spawn(|| embedder.embed_passages_stream(corpus, sink));
    scope.spawn(|| embedder.embed_queries(&["capital of Germany"]));
});
```

Threads share one loaded model through a reference or an `Arc`, with no lock
around it. The device runs one encoder batch at a time, and calls take turns
batch by batch in arrival order: a query waits for at most one batch of each
call ahead of it, never for a whole corpus. Tokenization and the callbacks
run outside the turn, so a slow callback delays only its own call. A call's
batches hold only its own texts, so sharing changes no vector. A smaller
`batch_tokens` shortens the wait.

## Errors

`StreamError::Inference` reports tokenizer and backend errors and a stop
request, and `StreamError::Sink` returns the callback's error; the callbacks
before an error have already run. Callback and input errors and stop requests
leave the model reusable. A panic inside an encoder batch leaves the device
in an unknown state, and later calls on that model return `Error::Poisoned`.

An input that tokenizes to no tokens is a tokenizer error; empty text is
valid when the tokenizer adds special tokens. Malformed shapes, unsupported
configurations, invalid tensor dimensions, non-finite weights or outputs,
overflowing capacities and GPU matrix weights outside FP16's range are
errors, and so are normalization statistics and a reranking head's
preactivations that overflow during inference.

## Fingerprints

`fingerprint()` hashes the model, config and tokenizer files, prefixes,
truncation, output mode, the numerical sources and artifacts, the platform
and the recorded device environment; batch capacity and input order are
excluded. Quantized precision has the same fingerprint on the CPU and CUDA,
and every other configuration has its own. CUDA records the driver release
when NVML is available and an explicit unavailable value otherwise; Vulkan
records the reported driver version.

## Device behavior

Vulkan requires version 1.3 with shader FP16, 16-bit storage, subgroup size
control and sufficient buffer and workgroup limits; the cooperative-matrix
shaders also require `VK_KHR_cooperative_matrix` with 16 x 16 x 16 FP16
tiles, subgroup shuffles and 32-wide subgroups. `Device::Auto` takes only
discrete adapters with conformant drivers; `Device::Vulkan(index)` also takes
integrated adapters and portability drivers such as MoltenVK.

An unresolved Vulkan submission failure quarantines the device and keeps its
resources until the process exits; later inference on it fails, and host
access to its mapped memory panics.

The NVIDIA driver compiles the shipped kernels on a machine's first CUDA
load, which takes tens of seconds, and keeps the result in its compute cache
(`CUDA_CACHE_PATH`). Later loads skip the compilation until the driver is
upgraded or a KodamaEmbed release changes the kernels.

A CUDA encoder keeps the recorded launches of up to 32 recent batch shapes
in float precision and 64 in quantized, and replays them when a shape
returns. A kept shape holds about 2 MiB of GPU memory with a 24-layer model.

## WebAssembly

```toml
[dependencies]
kodamaembed = { version = "0.1", default-features = false }
```

Without the `download` feature the crate builds for `wasm32-unknown-unknown`.
A wasm program opens no GPU driver and starts no threads: `Device::Auto` and
`Device::Cpu` run the portable kernels and the tokenizer on the calling
thread, and `Device::Cuda` and `Device::Vulkan` return an error. Models load
with `build_from_bytes`. The tokenizer's regular expressions run on a Rust
engine there instead of Oniguruma. Worker counts other than one return
`Error::Unsupported`, and so does `enable_profiling`, because the target has
no clock.

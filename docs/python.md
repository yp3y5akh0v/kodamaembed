# Python

The `kodamaembed` package runs the KodamaEmbed engine from Python, from the
command line and as an HTTP server.

```text
pip install kodamaembed
```

It needs Python 3.11 or newer. One wheel per platform holds the engine, the
tokenizer and the CUDA and Vulkan kernels; numpy is its only dependency. A GPU
needs only its driver.

## Embeddings

```python
from kodamaembed import Embedder

model = Embedder("intfloat/e5-large")
passages = model.embed_passages(["Berlin is the capital of Germany.", "Bananas are yellow."])
queries = model.embed_queries(["What is the capital of Germany?"])
scores = queries @ passages.T
```

`embed_passages` and `embed_queries` return a writable float32 array of shape
(number of texts, `model.dim`). Rows are L2-normalized. A model with prefixes
adds them: E5 embeds
`passage: Berlin is ...` and `query: What is ...`, modernbert-embed embeds
`search_document: Berlin is ...` and `search_query: What is ...`, and BGE,
mxbai and Arctic put a retrieval instruction before queries only. MiniLM has
none, and both methods return the same vectors. One string is one text; a list, tuple,
generator or numpy array of strings is many.

| Property | Value |
|---|---|
| `name` | The Hugging Face repository |
| `dim` | Values per vector |
| `device` | The device and backend, such as `NVIDIA GeForce RTX 4090` or `CPU (AVX2 kernels, 16 threads)` |
| `fingerprint` | SHA-256 of the model files, prefixes, truncation and numerics |

Quantized precision has the same fingerprint on the processor and on CUDA.

## Large collections

```python
for vectors in model.embed_passages_stream(open("corpus.txt", encoding="utf-8"), batch_size=1024):
    index.add(vectors)
```

`embed_passages_stream` and `embed_queries_stream` take any iterable and yield
arrays of up to `batch_size` vectors, in order. One batch of texts and vectors
is in memory at a time.

## Reranking

```python
from kodamaembed import Reranker

reranker = Reranker("cross-encoder/ms-marco-MiniLM-L6-v2")
documents = ["Bananas are yellow.", "Berlin is the capital of Germany."]
reranker.score("capital of Germany", documents)          # one logit per document, in order
reranker.rerank("capital of Germany", documents, top_k=1)  # [(1, logit of document 1)]
```

`score` returns one float32 relevance logit per document; higher is more
relevant. `rerank` returns `(index, score)` pairs, best first, for the `top_k`
best documents or for all of them. Pairs truncate to the model's maximum
length.

## Models and downloads

The [package README](../python/README.md#models) lists every model that loads
by name. `kodamaembed models` prints them, and `kodamaembed.list_models()`
returns them as `ModelInfo` objects with `name`, `kind`, `dim`, `max_length`,
`download_size` and `revision`. Names match in any letter case. Token counts
include special tokens and prefixes; longer texts are truncated.

A model downloads on first use into the Hugging Face cache, at a pinned
commit, and every file is checked against its pinned SHA-256 before it loads.
Files other tools already downloaded at that commit are reused.
`kodamaembed.download(name)` fetches and checks a model ahead of time and
returns its directory.

| Setting | Effect |
|---|---|
| `cache_dir=` | The hub cache directory |
| `HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME` | The cache when `cache_dir` is not given: `HF_HUB_CACHE`, else `HF_HOME/hub`, else `XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub` |
| `offline=True` | Load from the cache only |
| `offline=False` | Download when needed, even with `HF_HUB_OFFLINE` set |
| `HF_HUB_OFFLINE=1` | Load from the cache only when `offline` is not given |
| `HF_ENDPOINT` | Download from a mirror |
| `progress=False` | Show nothing |
| `progress=callable` | Call it with `(repository, file, received_bytes, total_bytes)` |

By default downloads show on stderr: redrawn in place on a terminal, and
otherwise one line as each file starts and one as it finishes. A progress
callable that raises is reported through `sys.unraisablehook` and the download
continues.

A container image that downloads during its build and runs offline:

```text
RUN pip install kodamaembed && kodamaembed download sentence-transformers/all-MiniLM-L6-v2
ENV HF_HUB_OFFLINE=1
```

## Devices and precision

| `device` | Runs on |
|---|---|
| `"auto"` (default) | CUDA 0 on compute capability 8.0 or newer, else the first discrete Vulkan GPU, else the processor |
| `"cpu"` | The processor, on every core |
| `"cuda"`, `"cuda:N"` | An NVIDIA GPU of compute capability 8.0 or newer |
| `"vulkan"`, `"vulkan:N"` | The first discrete Vulkan 1.3 GPU, or adapter N |

With `"auto"`, a GPU that lacks something the model needs is passed over, and
any other failure on a GPU raises its error.

| `precision` | Numerics |
|---|---|
| `"float"` (default) | FP32 on the processor; FP16 products with FP32 sums on CUDA; cooperative matrices or FP32 shaders on Vulkan |
| `"quantized"` | Centered INT8 of `intfloat/e5-large` on the processor or CUDA, with the same vector bits on both |

`"quantized"` runs `intfloat/e5-large` only; other models run `"float"`. An
x86-64 processor with AVX-VNNI, AVX2, FMA and F16C prepares and calibrates
it, also when it runs on CUDA. The first load caches the calibration, about
309 MB, for later loads; the [integration guide](integration.md#calibration-cache)
gives its location. With `"auto"`, quantized precision runs on CUDA when it
can and on the processor otherwise; the vectors are the same.

`threads=N` runs the model on the processor with N inference threads and a
quarter as many tokenizer threads. `batch_tokens` sets the tokens per encoder
batch (16,384 by default), which sizes the activation memory. Passing
`threads` with an explicit GPU device raises `UnsupportedError` before any
download. The loader checks the selected GPU when it opens the backend.

## Threads

One model can serve many threads. Calls release the GIL while they compute
and take turns on the device batch by batch, in arrival order, so a short
call does not wait for a long one to finish. The server answers its requests
this way.

## Errors

| Exception | Raised when |
|---|---|
| `DownloadError` | A file is missing offline, or a download fails |
| `ModelError` | A cached file does not match its pinned hash, or the tokenizer rejects an input |
| `UnsupportedError` | The model name is unknown, or the device cannot run the settings |
| `KodamaEmbedError` | The base class of the above, and device failures |
| `ValueError` | `device`, `precision`, `threads`, `batch_size` or `top_k` is malformed |
| `TypeError` | A text is not a string, or `progress` is not a bool or callable |

The error for a file that fails its hash check names the file and both hashes.

## Command line

```text
kodamaembed models
kodamaembed download intfloat/e5-large
kodamaembed embed sentence-transformers/all-MiniLM-L6-v2 "first text" "second text"
kodamaembed embed intfloat/e5-large --query --output queries.npy < queries.txt
kodamaembed serve sentence-transformers/all-MiniLM-L6-v2 cross-encoder/ms-marco-MiniLM-L6-v2 --port 8000
```

`embed` prints one JSON object per text, `{"index": 0, "embedding": [...]}`,
or with `--output` saves a float32 `.npy` array. Without texts it embeds each
line of stdin. `--device`, `--precision` and `--threads` work as in Python.
`python -m kodamaembed` runs the same command. Failures print the reason to
stderr and exit with status 1; usage errors exit with status 2.

## HTTP server

`kodamaembed serve` listens on `127.0.0.1:8000` by default; `--host` and
`--port` change that. One server holds every model it was given.

| Route | Format |
|---|---|
| `POST /v1/embeddings` | OpenAI embeddings |
| `POST /v1/rerank` | Cohere and Jina reranking |
| `GET /v1/models` | OpenAI model list |
| `GET /health` | `{"status": "ok"}` |

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.embeddings.create(
    model="sentence-transformers/all-MiniLM-L6-v2", input=["first text", "second text"]
)
vectors = [item.embedding for item in response.data]
```

Embedding requests take `input` as a string or a list of strings; token-ID
arrays are refused. `encoding_format` is `float` or `base64` (little-endian
float32). `dimensions` must equal the model's width when given. The optional
`input_type` picks the side of a search: `query` or `search_query` embeds
queries, and `passage`, `document` or `search_document` (the default) embeds
passages. `usage` counts the tokens encoded, special tokens and prefixes
included.

```text
curl http://127.0.0.1:8000/v1/rerank -H "Content-Type: application/json" -d "{\"query\": \"capital of Germany\", \"documents\": [\"Bananas are yellow.\", \"Berlin is the capital of Germany.\"], \"top_n\": 1, \"return_documents\": true}"
```

Rerank requests take `query`, `documents` as strings or `{"text": ...}`
objects, and the optional `top_n` and `return_documents`. Results are sorted
best first; `relevance_score` is the sigmoid of the model's logit.

`model` may be omitted when the server holds one model of the requested kind.
Errors are OpenAI error objects with status 400 for invalid requests, 404 for
unknown routes and models, and 500 when the model fails.

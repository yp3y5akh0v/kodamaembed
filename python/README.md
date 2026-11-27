<h1><img src="https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/logo.svg" alt="KodamaEmbed" width="420"></h1>

Text embeddings and reranking on CPUs and GPUs.

```text
pip install kodamaembed
```

```python
from kodamaembed import Embedder

model = Embedder("sentence-transformers/all-MiniLM-L6-v2")
passages = model.embed_passages([
    "Berlin is the capital of Germany.",
    "Bananas are yellow.",
])
query = model.embed_queries("What is the capital of Germany?")
print(passages @ query[0])  # cosine similarities
```

The wheel holds the engine and its GPU kernels, and numpy is its only
dependency. CUDA runs on NVIDIA GPUs of compute capability 8.0 or newer and
Vulkan on Vulkan 1.3 GPUs, each with its driver. Every model downloads at a
pinned Hugging Face commit into the Hugging Face cache, and every file is
checked against its pinned SHA-256 before it loads. `model.fingerprint`
identifies the model files and numerics.

## Models

| Name | Kind | Dimensions | Tokens | Download |
|---|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | embedding | 768 | 512 | 438.7 MB |
| `BAAI/bge-large-en-v1.5` | embedding | 1024 | 512 | 1.3 GB |
| `BAAI/bge-m3` | embedding | 1024 | 512 | 2.3 GB |
| `BAAI/bge-small-en-v1.5` | embedding | 384 | 512 | 134.2 MB |
| `ibm-granite/granite-embedding-english-r2` | embedding | 768 | 8192 | 301.6 MB |
| `intfloat/e5-large` | embedding | 1024 | 512 | 1.3 GB |
| `intfloat/e5-large-v2` | embedding | 1024 | 512 | 1.3 GB |
| `intfloat/e5-small-v2` | embedding | 384 | 512 | 134.2 MB |
| `intfloat/multilingual-e5-base` | embedding | 768 | 512 | 1.1 GB |
| `intfloat/multilingual-e5-large` | embedding | 1024 | 512 | 2.3 GB |
| `intfloat/multilingual-e5-small` | embedding | 384 | 512 | 487.7 MB |
| `mixedbread-ai/mxbai-embed-large-v1` | embedding | 1024 | 512 | 671.0 MB |
| `nomic-ai/modernbert-embed-base` | embedding | 768 | 8192 | 599.7 MB |
| `sentence-transformers/all-MiniLM-L12-v2` | embedding | 384 | 128 | 133.9 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | embedding | 384 | 256 | 91.3 MB |
| `sentence-transformers/all-mpnet-base-v2` | embedding | 768 | 384 | 438.4 MB |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | embedding | 384 | 128 | 479.7 MB |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | embedding | 768 | 128 | 1.1 GB |
| `Snowflake/snowflake-arctic-embed-l` | embedding | 1024 | 512 | 1.3 GB |
| `Snowflake/snowflake-arctic-embed-m-v1.5` | embedding | 768 | 512 | 436.3 MB |
| `Alibaba-NLP/gte-reranker-modernbert-base` | reranker | - | 8192 | 602.0 MB |
| `BAAI/bge-reranker-v2-m3` | reranker | - | 512 | 2.3 GB |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | reranker | - | 512 | 134.2 MB |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | reranker | - | 512 | 91.6 MB |
| `ibm-granite/granite-embedding-reranker-english-r2` | reranker | - | 1124 | 602.0 MB |

The multilingual-e5, bge-m3, bge-reranker-v2-m3 and paraphrase-multilingual
models are multilingual; the others are English. `kodamaembed.list_models()`
returns the same list. A model downloads on first use;
`kodamaembed.download(name)` fetches it ahead of time.

## Reranking

```python
from kodamaembed import Reranker

reranker = Reranker("cross-encoder/ms-marco-MiniLM-L6-v2")
documents = ["Bananas are yellow.", "Berlin is the capital of Germany."]
for index, score in reranker.rerank("capital of Germany", documents, top_k=1):
    print(documents[index], score)
```

## Large collections

`embed_passages_stream` embeds any iterable, yielding arrays of up to
`batch_size` vectors in order:

```python
for vectors in model.embed_passages_stream(lines_of_a_large_file, batch_size=1024):
    index.add(vectors)
```

## Devices and precision

```python
Embedder(name, device="auto")       # CUDA, else a discrete Vulkan GPU, else the CPU
Embedder(name, device="cuda:1")     # also "cpu", "cuda", "vulkan", "vulkan:N"
Embedder("intfloat/e5-large", precision="quantized")  # the same INT8 vectors on the CPU and CUDA
Embedder(name, threads=4)           # the CPU, with four threads
```

One model can be shared between threads; calls release the GIL while they
compute.

## Command line and server

```text
kodamaembed models
kodamaembed embed sentence-transformers/all-MiniLM-L6-v2 "first text" "second text"
kodamaembed download intfloat/e5-large
kodamaembed serve sentence-transformers/all-MiniLM-L6-v2 cross-encoder/ms-marco-MiniLM-L6-v2
```

`kodamaembed serve` answers `POST /v1/embeddings` in the OpenAI format and
`POST /v1/rerank` in the Cohere and Jina format:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.embeddings.create(
    model="sentence-transformers/all-MiniLM-L6-v2", input=["Berlin is the capital of Germany."]
)
```

## Offline use

`HF_HUB_OFFLINE=1` or `offline=True` loads from the cache. `HF_HOME`,
`HF_HUB_CACHE` and `HF_ENDPOINT` work as they do for other Hugging Face
tools.

The [Python guide](https://github.com/yp3y5akh0v/kodamaembed/blob/master/docs/python.md)
covers every option, and the
[quickstart notebook](https://github.com/yp3y5akh0v/kodamaembed/blob/master/notebooks/quickstart.ipynb)
embeds, searches and reranks in a few cells. The
[migration guide](https://github.com/yp3y5akh0v/kodamaembed/blob/master/docs/migration.md)
covers moving from sentence-transformers, FastEmbed or an OpenAI-compatible
server, and the
[benchmarks](https://github.com/yp3y5akh0v/kodamaembed/blob/master/docs/benchmarks.md)
cover latency and accuracy.

Licensed under [Apache-2.0](https://github.com/yp3y5akh0v/kodamaembed/blob/master/LICENSE).

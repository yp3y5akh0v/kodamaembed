# Migrating to KodamaEmbed

Checked against sentence-transformers 6.2.0 and FastEmbed 0.9.1. Models load
by their Hugging Face names in any letter case; `kodamaembed models` lists
them.

## From sentence-transformers

| sentence-transformers | KodamaEmbed |
|---|---|
| `SentenceTransformer(name)` | `Embedder(name)` |
| `encode(passages)`, `encode_document(passages)` | `embed_passages(passages)` |
| `encode(queries)`, `encode_query(queries)` | `embed_queries(queries)` |
| `normalize_embeddings=True` | Always: rows are L2-normalized float32 |
| `batch_size=` | `embed_passages_stream(texts, batch_size=)` for large inputs |
| `device="cpu"`, `"cuda"`, `"cuda:N"` | The same strings, plus `"vulkan"` and `"auto"`, the default |
| `similarity(a, b)` | `a @ b.T` |
| `CrossEncoder(name).predict(pairs)` | `Reranker(name).score(query, documents)` |
| `CrossEncoder(name).rank(query, documents)` | `Reranker(name).rerank(query, documents)`, as `(index, score)` pairs |
| `cache_folder=` | `cache_dir=` |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large")
passages = model.encode(["passage: Berlin is the capital of Germany."], normalize_embeddings=True)
queries = model.encode(["query: What is the capital of Germany?"], normalize_embeddings=True)
```

```python
from kodamaembed import Embedder

model = Embedder("intfloat/e5-large")
passages = model.embed_passages(["Berlin is the capital of Germany."])
queries = model.embed_queries(["What is the capital of Germany?"])
```

## From FastEmbed

| FastEmbed | KodamaEmbed |
|---|---|
| `TextEmbedding(name)` | `Embedder(name)` |
| `embed(texts)`, `passage_embed(texts)`, generators | `embed_passages(texts)`, an array, or `embed_passages_stream(texts)` |
| `query_embed(texts)` | `embed_queries(texts)` |
| `TextCrossEncoder(name).rerank(query, documents)` | `Reranker(name).score(query, documents)`; both return logits |
| `fastembed-gpu`, `cuda=True`, `providers=` | `device=` |
| `threads=`, `parallel=` | `threads=` |
| `cache_dir=` | `cache_dir=`, the Hugging Face hub cache |

| FastEmbed reranker | KodamaEmbed name |
|---|---|
| `Xenova/ms-marco-MiniLM-L-6-v2` | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `Xenova/ms-marco-MiniLM-L-12-v2` | `cross-encoder/ms-marco-MiniLM-L12-v2` |

```python
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
passages = list(model.passage_embed(["Berlin is the capital of Germany."]))
```

```python
from kodamaembed import Embedder

model = Embedder("BAAI/bge-small-en-v1.5")
passages = model.embed_passages(["Berlin is the capital of Germany."])
```

## From an OpenAI-compatible server

`kodamaembed serve` answers at the same `base_url`:

```text
kodamaembed serve intfloat/e5-large cross-encoder/ms-marco-MiniLM-L6-v2
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
passages = client.embeddings.create(model="intfloat/e5-large", input=["Berlin is the capital of Germany."])
queries = client.embeddings.create(
    model="intfloat/e5-large",
    input=["What is the capital of Germany?"],
    extra_body={"input_type": "query"},
)
```

A text without `input_type` embeds as a passage, so queries need `query` or
`search_query` to get the model's query prefix. `POST /v1/rerank` takes the
Cohere and Jina format. From TEI's `/rerank`:

| TEI | KodamaEmbed |
|---|---|
| `texts` | `documents` |
| `return_text` | `return_documents` |
| `score` | `relevance_score`, the sigmoid of the logit |
| `raw_scores=true` | `Reranker.score` in Python, which returns logits |
| `truncate` | Always on: pairs truncate to the model's maximum length |

The server has no authentication; see [SECURITY.md](../SECURITY.md).

## Existing indexes

KodamaEmbed adds each model card's prefixes: `query: ` and `passage: ` for E5,
`search_query: ` and `search_document: ` for modernbert-embed, and a retrieval
instruction before BGE, mxbai and Arctic queries. Delete prefixes typed into
texts, or they apply twice. Neither library adds E5's, BGE's or
modernbert-embed's prefixes; sentence-transformers adds mxbai's and Arctic's
only in `encode_query`.

| Index built with | Model | Action |
|---|---|---|
| Either library | E5 models, passages without `passage: ` | Re-embed |
| sentence-transformers | bge-m3 | Re-embed texts over 512 tokens; sentence-transformers reads 8,192 |
| sentence-transformers | granite-embedding-english-r2 | Normalize; re-embed if it ran in BF16, which sentence-transformers 6 uses by default for this checkpoint |
| sentence-transformers | mxbai-embed-large-v1 | Normalize; re-embed if it ran in FP16, which sentence-transformers 6 uses by default for this checkpoint |
| sentence-transformers | paraphrase-multilingual-MiniLM-L12-v2, paraphrase-multilingual-mpnet-base-v2 | Normalize |
| FastEmbed | BAAI/bge-base-en-v1.5, BAAI/bge-small-en-v1.5, paraphrase-multilingual-MiniLM-L12-v2 | Re-embed: FastEmbed runs `-onnx-Q` exports of these, up to L2 4e-3 away |
| FastEmbed | multilingual-e5-large | Normalize |
| FastEmbed | paraphrase-multilingual-mpnet-base-v2 | Normalize; re-embed texts over 128 tokens, where FastEmbed reads 512 |

Normalizing needs no model:

```python
import numpy as np

vectors = np.array([[3.0, 4.0]], dtype=np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
```

Every other index stays: on the processor, vectors are within L2 1e-5 of
sentence-transformers' FP32 and of FastEmbed's full-precision exports, and
within 5e-3 on CUDA and Vulkan. `precision="quantized"` approximates FP32, so
an FP32 index of E5-large needs re-embedding first.

Reranker logits match both libraries within 2e-5 on the processor and 5e-3 on
CUDA and Vulkan. sentence-transformers applies a sigmoid by default to the
logits of bge-reranker-v2-m3, gte-reranker-modernbert-base and
granite-embedding-reranker-english-r2. KodamaEmbed truncates
bge-reranker-v2-m3 pairs at 512 tokens, where both libraries read up to 8,192.

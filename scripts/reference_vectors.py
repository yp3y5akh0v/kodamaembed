# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy==2.4.3",
#     "sentence-transformers==6.0.0",
#     "torch==2.13.0",
#     "transformers==5.15.1",
# ]
# ///
"""Writes crates/kodamaembed/tests/registry/vectors.json, sentence-transformers' FP32 CPU vectors
of every registered model at its pinned revision, prefixed as its model card says.
Usage: uv run scripts/reference_vectors.py"""

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling

SEARCH_QUERY = "Represent this sentence for searching relevant passages: "

# Repository, pinned revision, query prefix and passage prefix.
EMBEDDERS = [
    ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a", SEARCH_QUERY, ""),
    ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09", SEARCH_QUERY, ""),
    ("BAAI/bge-m3", "9a0624b896d81da7492a910ffa53731274b6cf3d", "", ""),
    ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", SEARCH_QUERY, ""),
    ("ibm-granite/granite-embedding-english-r2", "47ea694b257b703fee9253d75c2b1f2985180498", "", ""),
    ("intfloat/e5-large", "4dc6d853a804b9c8886ede6dda8a073b7dc08a81", "query: ", "passage: "),
    ("intfloat/e5-large-v2", "f169b11e22de13617baa190a028a32f3493550b6", "query: ", "passage: "),
    ("intfloat/e5-small-v2", "ffb93f3bd4047442299a41ebb6fa998a38507c52", "query: ", "passage: "),
    ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a", "query: ", "passage: "),
    ("intfloat/multilingual-e5-large", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3", "query: ", "passage: "),
    ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", "query: ", "passage: "),
    ("mixedbread-ai/mxbai-embed-large-v1", "b33106f585b9ce46904ad7443a3b52b7a63e231c", SEARCH_QUERY, ""),
    (
        "nomic-ai/modernbert-embed-base",
        "d556a88e332558790b210f7bdbe87da2fa94a8d8",
        "search_query: ",
        "search_document: ",
    ),
    ("sentence-transformers/all-MiniLM-L12-v2", "a50ef00143b4d5391434df20ae11632588ac25be", "", ""),
    ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "", ""),
    ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130", "", ""),
    (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "",
        "",
    ),
    (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "4328cf26390c98c5e3c738b4460a05b95f4911f5",
        "",
        "",
    ),
    ("Snowflake/snowflake-arctic-embed-l", "d8fb21ca8d905d2832ee8b96c894d3298964346b", SEARCH_QUERY, ""),
    ("Snowflake/snowflake-arctic-embed-m-v1.5", "e58a8f756156a1293d763f17e3aae643474e9b8a", SEARCH_QUERY, ""),
]
RERANKERS = [
    ("Alibaba-NLP/gte-reranker-modernbert-base", "f7481e6055501a30fb19d090657df9ec1f79ab2c"),
    ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    ("cross-encoder/ms-marco-MiniLM-L12-v2", "7b0235231ca2674cb8ca8f022859a6eba2b1c968"),
    ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
    ("ibm-granite/granite-embedding-reranker-english-r2", "d09d3d6971b689bf9c23839e45a470874d46e13a"),
]
# Both accept 8,192 tokens; KodamaEmbed truncates them to 512.
TRUNCATED = {"BAAI/bge-m3": 512, "BAAI/bge-reranker-v2-m3": 512}

QUERY = "how do solar panels turn sunlight into electricity?"
PASSAGES = [
    # Accents, a symbol and CJK exercise the tokenizer's normalization.
    "Solar panels hold photovoltaic cells: absorbed photons free electrons, and the cells' "
    "electric field drives them as direct current. Café owners in Zürich and "
    "東京 run espresso machines on rooftop arrays (≈ 4 kW).",
    # Six scripts, full-width forms and an ideographic space exercise the
    # multilingual tokenizers' normalization and segmentation.
    "Solarzellen wandeln Sonnenlicht in Strom um, größere Anlagen speisen ins Netz ein. "
    "Солнечные панели превращают свет в электричество. 太阳能电池板把阳光转化为电能。"
    "تحول الألواح الشمسية ضوء الشمس إلى كهرباء. सौर पैनल धूप को बिजली में बदलते हैं। "
    "ＰＶ　１００％",
    # Unrelated, and longer than every model's truncation, so tokens past it must not count.
    " ".join(f"Entry {i} of the log notes that the ferry left the harbor on time." for i in range(700)),
    # Literal padding tokens, which tokenizers keep as padding ids and XLM-R positions apart.
    "Tokenizers pad short batches with <pad> or [PAD]; neither is a word.",
    # A tab, space runs and a trailing newline, which tokenizers drop rather than keep as tokens.
    "Line one.\nLine two.\n\n\tIndented   line three.\r\n",
]

FIXTURE = Path(__file__).resolve().parents[1] / "crates/kodamaembed/tests/registry/vectors.json"
FP32 = {"dtype": torch.float32}


def shortest(values):
    """Each value as the shortest decimal that reads back as the same float32."""
    return [float(np.format_float_positional(value, unique=True)) for value in values.astype(np.float32)]


def dumps(value, depth=0):
    """Indented JSON that keeps each list of numbers on one line."""
    inner = "\n" + " " * (depth + 1)
    if isinstance(value, dict):
        items = [f"{json.dumps(key)}: {dumps(item, depth + 1)}" for key, item in value.items()]
    elif isinstance(value, list) and not all(isinstance(item, float) for item in value):
        items = [dumps(item, depth + 1) for item in value]
    else:
        return json.dumps(value, separators=(",", ":"))
    brackets = "{}" if isinstance(value, dict) else "[]"
    return brackets[0] + inner + ("," + inner).join(items) + "\n" + " " * depth + brackets[1]


def check_fp32(model):
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if dtypes != {torch.float32}:
        raise SystemExit(f"expected FP32 parameters, found {dtypes}")


def embedder(repository, revision, query_prefix, passage_prefix):
    model = SentenceTransformer(repository, revision=revision, device="cpu", model_kwargs=FP32)
    model.max_seq_length = TRUNCATED.get(repository, model.max_seq_length)
    check_fp32(model)
    pooling = next(module for module in model if isinstance(module, Pooling))
    if pooling.pooling_mode not in ("cls", "mean"):
        raise SystemExit(f"{repository}: {pooling.pooling_mode} pooling")
    if pooling.pooling_mode == "mean" and not pooling.include_prompt:
        raise SystemExit(f"{repository}: mean pooling leaves out the prefix tokens")
    if model.default_prompt_name is not None:
        raise SystemExit(f"{repository}: a default prompt would prefix every text")
    prompts = {name: text for name, text in model.prompts.items() if text}
    documented = {"query": query_prefix, "passage": passage_prefix, "document": passage_prefix}
    if any(documented.get(name) != text for name, text in prompts.items()):
        raise SystemExit(f"{repository}: prompts {prompts} differ from the model card's prefixes")

    def encode(prefix, texts):
        # The prefix is concatenated as KodamaEmbed does; prompt="" stops
        # sentence-transformers from adding one of its own.
        vectors = model.encode(
            [prefix + text for text in texts],
            batch_size=1,
            prompt="",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [shortest(vector) for vector in vectors]

    return {
        "repository": repository,
        "revision": revision,
        "pooling": pooling.pooling_mode,
        "max_length": model.max_seq_length,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "query_vectors": encode(query_prefix, [QUERY]),
        "passage_vectors": encode(passage_prefix, PASSAGES),
    }


def reranker(repository, revision):
    model = CrossEncoder(repository, revision=revision, device="cpu", model_kwargs=FP32)
    model.max_seq_length = TRUNCATED.get(repository, model.max_seq_length)
    check_fp32(model)
    scores = model.predict(
        [(QUERY, passage) for passage in PASSAGES],
        batch_size=1,
        activation_fn=torch.nn.Identity(),
        convert_to_numpy=True,
    )
    return {
        "repository": repository,
        "revision": revision,
        "max_length": model.max_seq_length,
        "scores": shortest(scores),
    }


def main():
    fixture = {
        "generator": "scripts/reference_vectors.py",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("sentence-transformers", "transformers", "torch", "numpy")
        },
        "query": QUERY,
        "passages": PASSAGES,
        "embedders": [embedder(*model) for model in EMBEDDERS],
        "rerankers": [reranker(*model) for model in RERANKERS],
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(dumps(fixture) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()

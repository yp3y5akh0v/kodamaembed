"""Text embeddings and reranking on CPUs and GPUs.

    >>> from kodamaembed import Embedder
    >>> model = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    >>> model.embed_passages(["Berlin is the capital of Germany."]).shape
    (1, 384)
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np

from . import _kodamaembed
from ._kodamaembed import (
    DownloadError,
    KodamaEmbedError,
    ModelError,
    UnsupportedError,
    __version__,
)
from ._progress import ProgressDisplay

__all__ = [
    "DownloadError",
    "Embedder",
    "KodamaEmbedError",
    "ModelError",
    "ModelInfo",
    "ProgressCallback",
    "Reranker",
    "UnsupportedError",
    "__version__",
    "download",
    "list_models",
]

ProgressCallback = Callable[[str, str, int, int], object]
"""Receives (repository, file, received bytes, total bytes) as a download runs."""


@dataclass(frozen=True)
class ModelInfo:
    """A model that loads by name."""

    name: str
    """The Hugging Face repository, such as ``intfloat/e5-large``."""
    kind: Literal["embedding", "reranker"]
    dim: int | None
    """Values per vector; None for a reranker, which scores pairs."""
    max_length: int
    """Tokens an input truncates to, special tokens included."""
    download_size: int
    """Bytes of the model's files, which its first use downloads."""
    revision: str
    """The pinned Hugging Face commit."""


def list_models() -> list[ModelInfo]:
    """Every model that loads by name."""
    return [ModelInfo(*row) for row in _kodamaembed.models()]


def download(
    model: str,
    *,
    cache_dir: str | PathLike[str] | None = None,
    offline: bool | None = None,
    progress: bool | ProgressCallback = True,
) -> Path:
    """Downloads `model` into the Hugging Face cache unless it is there, checks its files'
    SHA-256 and returns their directory; the options are :class:`Embedder`'s."""
    return _kodamaembed.download(model, _source(cache_dir, offline, progress))


class _Loaded:
    """What every loaded model reports."""

    _native: _kodamaembed.Embedder | _kodamaembed.Reranker

    @property
    def name(self) -> str:
        """The model's Hugging Face repository."""
        return self._native.name

    @property
    def device(self) -> str:
        """The device and backend the model runs on."""
        return self._native.device

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the model files, input handling and numerics."""
        return self._native.fingerprint

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, device={self.device!r})"


class Embedder(_Loaded):
    """A sentence embedder: one L2-normalized float32 vector per text.

    Args:
        model: A name from :func:`list_models`, such as
            ``"sentence-transformers/all-MiniLM-L6-v2"``.
        device: ``"auto"`` picks CUDA on compute capability 8.0 or newer, else
            a discrete Vulkan GPU, else the processor. ``"cpu"``, ``"cuda"``,
            ``"cuda:N"``, ``"vulkan"`` (the first discrete GPU) and
            ``"vulkan:N"`` choose one.
        precision: ``"float"`` is FP32 on the processor and FP16 products with
            FP32 sums on CUDA. ``"quantized"`` runs ``intfloat/e5-large``'s
            centered INT8 preset, with the same vectors on the processor and CUDA.
        batch_tokens: Tokens per encoder batch, which sizes the activation
            memory.
        threads: Runs the model on the processor with this many threads.
        cache_dir: The Hugging Face hub cache. By default ``HF_HUB_CACHE``,
            else ``HF_HOME/hub``, else ``XDG_CACHE_HOME/huggingface/hub``,
            else ``~/.cache/huggingface/hub``.
        offline: True loads only from the cache, False may download even when
            ``HF_HUB_OFFLINE`` is set, and None follows ``HF_HUB_OFFLINE``.
        progress: True shows downloads on stderr, False shows nothing, and a
            :data:`ProgressCallback` receives each step.

    One instance can be shared between threads: calls release the GIL while
    they compute and take turns batch by batch.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        precision: Literal["float", "quantized"] = "float",
        batch_tokens: int | None = None,
        threads: int | None = None,
        cache_dir: str | PathLike[str] | None = None,
        offline: bool | None = None,
        progress: bool | ProgressCallback = True,
    ) -> None:
        self._native = _kodamaembed.Embedder(
            model,
            precision,
            _run(device, batch_tokens, threads),
            _source(cache_dir, offline, progress),
        )

    @property
    def dim(self) -> int:
        """Values per vector."""
        return self._native.dim

    def embed_passages(self, texts: str | Iterable[str]) -> np.ndarray:
        """Vectors of the documents to search as a (number of texts, dim) array, in order,
        after the model's passage prefix (``"passage: "`` for E5); one string is one text."""
        return self._encode(texts, query=False)[0]

    def embed_queries(self, texts: str | Iterable[str]) -> np.ndarray:
        """Vectors of search queries, as :meth:`embed_passages` returns, after
        the model's query prefix."""
        return self._encode(texts, query=True)[0]

    def embed_passages_stream(
        self, texts: str | Iterable[str], batch_size: int = 1024
    ) -> Iterator[np.ndarray]:
        """Embeds a large or lazy iterable of passages, yielding arrays of up
        to `batch_size` vectors in order."""
        return self._stream(texts, batch_size, query=False)

    def embed_queries_stream(
        self, texts: str | Iterable[str], batch_size: int = 1024
    ) -> Iterator[np.ndarray]:
        """Embeds queries as :meth:`embed_passages_stream` embeds passages."""
        return self._stream(texts, batch_size, query=True)

    def _encode(self, texts: str | Iterable[str], *, query: bool) -> tuple[np.ndarray, int]:
        """The vectors and the tokens they encoded, prefixes and special
        tokens included."""
        native = self._native.embed_queries if query else self._native.embed_passages
        values, tokens = native(_texts(texts))
        return np.frombuffer(values, dtype=np.float32).reshape(-1, self.dim), tokens

    def _stream(
        self, texts: str | Iterable[str], batch_size: int, *, query: bool
    ) -> Iterator[np.ndarray]:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, not {batch_size}")
        return (self._encode(chunk, query=query)[0] for chunk in _chunks(texts, batch_size))


class Reranker(_Loaded):
    """A cross-encoder: how relevant each document is to a query.

    It takes the loading options of :class:`Embedder` except `precision`.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        batch_tokens: int | None = None,
        threads: int | None = None,
        cache_dir: str | PathLike[str] | None = None,
        offline: bool | None = None,
        progress: bool | ProgressCallback = True,
    ) -> None:
        self._native = _kodamaembed.Reranker(
            model,
            _run(device, batch_tokens, threads),
            _source(cache_dir, offline, progress),
        )

    def score(self, query: str, documents: str | Iterable[str]) -> np.ndarray:
        """The relevance logit of each document to `query`, in the documents'
        order: higher is more relevant."""
        return self._score(query, documents)[0]

    def rerank(
        self, query: str, documents: str | Iterable[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """(index, score) of the `top_k` most relevant documents, best first;
        every document when `top_k` is None."""
        if top_k is not None and top_k < 0:
            raise ValueError(f"top_k must not be negative, not {top_k}")
        scores = self.score(query, documents)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(int(index), float(scores[index])) for index in order]

    def _score(self, query: str, documents: str | Iterable[str]) -> tuple[np.ndarray, int]:
        """The logits and the tokens their pairs encoded."""
        values, tokens = self._native.score(query, _texts(documents))
        return np.frombuffer(values, dtype=np.float32), tokens


def _texts(texts: str | Iterable[str]) -> list[str]:
    """One string is one text; any other iterable holds the texts."""
    if isinstance(texts, str):
        return [texts]
    return texts if isinstance(texts, list) else list(texts)


def _chunks(texts: str | Iterable[str], size: int) -> Iterator[list[str]]:
    iterator = iter([texts] if isinstance(texts, str) else texts)
    while chunk := list(islice(iterator, size)):
        yield chunk


def _run(device: str, batch_tokens: int | None, threads: int | None) -> _kodamaembed.Run:
    return {"device": device, "batch_tokens": batch_tokens, "threads": threads}


def _source(
    cache_dir: str | PathLike[str] | None,
    offline: bool | None,
    progress: bool | ProgressCallback,
) -> _kodamaembed.Source:
    return {"cache_dir": cache_dir, "offline": offline, "progress": _observer(progress)}


def _observer(progress: bool | ProgressCallback) -> ProgressCallback | None:
    if progress is True:
        # Without a console, as under pythonw, there is nowhere to show it.
        return None if sys.stderr is None else ProgressDisplay(sys.stderr)
    if progress is False:
        return None
    if callable(progress):
        return progress
    raise TypeError(f"progress must be True, False or a callable, not {progress!r}")

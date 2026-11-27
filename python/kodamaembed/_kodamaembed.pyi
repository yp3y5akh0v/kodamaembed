"""The native module; ``kodamaembed`` wraps it in its public API."""

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import TypedDict

__version__: str

class KodamaEmbedError(Exception): ...
class DownloadError(KodamaEmbedError): ...
class ModelError(KodamaEmbedError): ...
class UnsupportedError(KodamaEmbedError): ...

class Run(TypedDict):
    """Where and how a model runs; exists only for type checking."""

    device: str
    batch_tokens: int | None
    threads: int | None

class Source(TypedDict):
    """Where a model's files come from; exists only for type checking."""

    cache_dir: str | PathLike[str] | None
    offline: bool | None
    progress: Callable[[str, str, int, int], object] | None

class Embedder:
    def __init__(self, model: str, precision: str, run: Run, source: Source) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def dim(self) -> int: ...
    @property
    def device(self) -> str: ...
    @property
    def fingerprint(self) -> str: ...
    def embed_passages(self, texts: list[str]) -> tuple[bytearray, int]:
        """Native-endian float32 rows and the tokens they encoded."""
    def embed_queries(self, texts: list[str]) -> tuple[bytearray, int]:
        """As embed_passages, after the model's query prefix."""

class Reranker:
    def __init__(self, model: str, run: Run, source: Source) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def fingerprint(self) -> str: ...
    def score(self, query: str, documents: list[str]) -> tuple[bytearray, int]:
        """Native-endian float32 logits and the tokens their pairs encoded."""

def models() -> list[tuple[str, str, int | None, int, int, str]]:
    """(name, kind, dim, max_length, download_size, revision) of every model."""

def download(model: str, source: Source) -> Path: ...

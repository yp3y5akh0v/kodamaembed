import pytest
from names import MINILM, RERANKER

import kodamaembed


@pytest.fixture(scope="session")
def embedder() -> kodamaembed.Embedder:
    return kodamaembed.Embedder(MINILM, device="cpu", progress=False)


@pytest.fixture(scope="session")
def reranker() -> kodamaembed.Reranker:
    return kodamaembed.Reranker(RERANKER, device="cpu", progress=False)

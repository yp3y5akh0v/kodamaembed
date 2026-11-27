"""Runs with ``pytest -m gpu``: needs CUDA at compute capability 8.0 or newer, a discrete
Vulkan GPU and, for quantized precision, a processor with AVX-VNNI."""

import threading

import numpy as np
import pytest
from names import E5, MINILM, RERANKER

import kodamaembed

pytestmark = pytest.mark.gpu

TEXTS = [f"passage {index} about {topic}" for index, topic in enumerate(["Berlin", "bananas"] * 40)]


def cosines(left, right):
    return np.sum(left * right, axis=1)


def on_gpu(model):
    """Processor models report CPU first; GPUs report the driver's name."""
    return not model.device.startswith("CPU")


def test_cuda_answers_every_thread_with_the_same_vectors(embedder):
    gpu = kodamaembed.Embedder(MINILM, device="cuda", progress=False)
    assert on_gpu(gpu)
    expected = gpu.embed_passages(TEXTS)
    elsewhere = []
    worker = threading.Thread(target=lambda: elsewhere.append(gpu.embed_passages(TEXTS)))
    worker.start()
    worker.join()
    np.testing.assert_array_equal(elsewhere[0], expected)
    assert np.all(cosines(expected, embedder.embed_passages(TEXTS)) > 0.999)


def test_quantized_vectors_are_the_same_on_cuda_and_the_processor_and_close_to_float():
    gpu = kodamaembed.Embedder(E5, device="cuda", precision="quantized", progress=False)
    cpu = kodamaembed.Embedder(E5, device="cpu", precision="quantized", progress=False)
    assert on_gpu(gpu)
    quantized = gpu.embed_passages(TEXTS)
    np.testing.assert_array_equal(quantized, cpu.embed_passages(TEXTS))
    assert gpu.fingerprint == cpu.fingerprint
    float_gpu = kodamaembed.Embedder(E5, device="cuda", progress=False)
    assert np.all(cosines(quantized, float_gpu.embed_passages(TEXTS)) > 0.99)
    assert float_gpu.fingerprint != gpu.fingerprint


def test_vulkan_vectors_match_the_processor(embedder):
    gpu = kodamaembed.Embedder(MINILM, device="vulkan", progress=False)
    assert "Vulkan" in gpu.device
    assert np.all(cosines(gpu.embed_passages(TEXTS), embedder.embed_passages(TEXTS)) > 0.999)


def test_the_reranker_on_cuda_ranks_as_the_processor_does(reranker):
    gpu = kodamaembed.Reranker(RERANKER, device="cuda", progress=False)
    assert on_gpu(gpu)
    query = "the capital of Germany"
    np.testing.assert_allclose(gpu.score(query, TEXTS), reranker.score(query, TEXTS), atol=0.05)
    assert [index for index, _ in gpu.rerank(query, TEXTS, top_k=5)] == [
        index for index, _ in reranker.rerank(query, TEXTS, top_k=5)
    ]

import http.server
import io
import shutil
import sys
import threading

import numpy as np
import pytest
from names import MINILM, RERANKER, UNREGISTERED

import kodamaembed
from kodamaembed import DownloadError, KodamaEmbedError, ModelError, UnsupportedError
from kodamaembed._progress import ProgressDisplay

TEXTS = ["Berlin is the capital of Germany.", "Bananas are yellow."]


@pytest.fixture
def hub_mirror(monkeypatch):
    """Serves the cached all-MiniLM-L6-v2 files at the Hub's URLs, so a
    download runs without the network."""
    snapshot = kodamaembed.download(MINILM, progress=False)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = snapshot / self.path.rsplit("/", 1)[-1]
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as file:
                shutil.copyfileobj(file, self.wfile)

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{httpd.server_address[1]}")
        yield
        httpd.shutdown()


def test_models_list_names_kinds_widths_lengths_sizes_and_pins():
    models = {model.name: model for model in kodamaembed.list_models()}
    minilm = models[MINILM]
    assert (minilm.kind, minilm.dim, minilm.max_length) == ("embedding", 384, 256)
    assert minilm.download_size == 612 + 466_247 + 90_868_376
    assert minilm.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert models["intfloat/e5-large"].dim == 1024
    bge = models["BAAI/bge-small-en-v1.5"]
    assert (bge.kind, bge.dim, bge.max_length) == ("embedding", 384, 512)
    reranker = models[RERANKER]
    assert (reranker.kind, reranker.dim, reranker.max_length) == ("reranker", None, 512)


def test_vectors_are_writable_normalized_float32_rows(embedder):
    vectors = embedder.embed_passages(TEXTS)
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert vectors.flags.writeable
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_a_related_passage_scores_above_an_unrelated_one(embedder):
    query = embedder.embed_queries(["capital of Germany"])[0]
    scores = embedder.embed_passages(TEXTS) @ query
    assert scores[0] > scores[1]


def test_one_string_is_one_text_and_no_texts_are_no_rows(embedder):
    assert embedder.embed_queries("capital of Germany").shape == (1, 384)
    assert embedder.embed_passages([]).shape == (0, 384)


def test_any_iterable_of_strings_embeds(embedder):
    expected = embedder.embed_passages(TEXTS)
    np.testing.assert_array_equal(embedder.embed_passages(tuple(TEXTS)), expected)
    np.testing.assert_array_equal(embedder.embed_passages(text for text in TEXTS), expected)
    np.testing.assert_array_equal(embedder.embed_passages(np.array(TEXTS)), expected)


def test_streams_match_one_call_in_bounded_chunks(embedder):
    texts = [f"passage number {index}" for index in range(10)]
    chunks = list(embedder.embed_passages_stream(iter(texts), batch_size=4))
    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate(chunks), embedder.embed_passages(texts))
    queries = list(embedder.embed_queries_stream(texts, batch_size=10))
    np.testing.assert_array_equal(queries[0], embedder.embed_queries(texts))
    with pytest.raises(ValueError, match="batch_size"):
        embedder.embed_passages_stream(texts, batch_size=0)


def test_texts_must_be_strings(embedder):
    with pytest.raises(TypeError, match="str"):
        embedder.embed_passages(["text", 3])


def test_properties_describe_the_loaded_model(embedder):
    assert embedder.name == MINILM
    assert embedder.dim == 384
    assert "CPU" in embedder.device
    assert embedder.fingerprint.startswith("sha256:")
    assert repr(embedder) == f"Embedder({MINILM!r}, device={embedder.device!r})"


def test_threads_share_one_model_and_get_the_same_vectors(embedder):
    expected = embedder.embed_passages(TEXTS)
    results = [None] * 4

    def work(slot):
        results[slot] = embedder.embed_passages(TEXTS)

    threads = [threading.Thread(target=work, args=(slot,)) for slot in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_threads_run_the_model_on_the_processor():
    model = kodamaembed.Embedder(MINILM, device="auto", threads=2, progress=False)
    assert "2 threads" in model.device


def test_unknown_models_name_the_ones_that_load():
    with pytest.raises(UnsupportedError, match="all-MiniLM-L6-v2"):
        kodamaembed.Embedder(UNREGISTERED, progress=False)
    with pytest.raises(UnsupportedError, match="ms-marco"):
        kodamaembed.download(UNREGISTERED, progress=False)


@pytest.mark.parametrize(
    ("settings", "error", "message"),
    [
        ({"device": "tpu"}, ValueError, "device"),
        ({"device": "cuda:x"}, ValueError, "device"),
        ({"precision": "fp8"}, ValueError, "precision"),
        ({"threads": 0}, ValueError, "threads"),
        ({"progress": "yes"}, TypeError, "progress"),
        ({"device": "vulkan:0", "precision": "quantized"}, UnsupportedError, "quantized precision"),
        ({"device": "cpu", "precision": "quantized"}, UnsupportedError, "intfloat/e5-large"),
        ({"device": "cuda", "threads": 2}, UnsupportedError, "processor"),
    ],
)
def test_settings_are_checked_before_loading(tmp_path, settings, error, message):
    # Offline with an empty cache: reaching the files would be a DownloadError.
    with pytest.raises(error, match=message):
        kodamaembed.Embedder(MINILM, cache_dir=tmp_path, offline=True, **settings)


def test_offline_with_an_empty_cache_is_a_download_error(tmp_path):
    with pytest.raises(DownloadError, match="offline"):
        kodamaembed.Embedder(MINILM, cache_dir=tmp_path, offline=True, progress=False)


def test_download_returns_the_checked_snapshot():
    path = kodamaembed.download(MINILM, progress=False)
    assert (path / "model.safetensors").stat().st_size == 90_868_376
    assert path.name == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_a_progress_callable_hears_every_downloading_file(tmp_path, hub_mirror):
    events = []
    kodamaembed.download(MINILM, cache_dir=tmp_path, progress=lambda *event: events.append(event))
    assert (MINILM, "config.json", 612, 612) in events
    assert (MINILM, "model.safetensors", 90_868_376, 90_868_376) in events
    assert all(received <= total for _, _, received, total in events)


def test_a_failing_progress_callable_does_not_stop_the_download(tmp_path, hub_mirror, monkeypatch):
    def fail(*_):
        raise RuntimeError("the display broke")

    reported = []
    monkeypatch.setattr(sys, "unraisablehook", reported.append)
    path = kodamaembed.download(MINILM, cache_dir=tmp_path, progress=fail)
    assert (path / "model.safetensors").stat().st_size == 90_868_376
    assert reported
    assert all(isinstance(report.exc_value, RuntimeError) for report in reported)


def test_the_progress_display_announces_and_finishes_each_file():
    stream = io.StringIO()
    show = ProgressDisplay(stream)
    show(MINILM, "model.safetensors", 45_000_000, 90_868_376)
    show(MINILM, "model.safetensors", 60_000_000, 90_868_376)
    show(MINILM, "model.safetensors", 90_868_376, 90_868_376)
    show(MINILM, "config.json", 612, 612)
    assert stream.getvalue().splitlines() == [
        f"Downloading {MINILM}/model.safetensors (90.9 MB)",
        f"Downloaded {MINILM}/model.safetensors (90.9 MB)",
        f"Downloaded {MINILM}/config.json (612 B)",
    ]


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_on_a_terminal_the_progress_display_redraws_one_line():
    stream = Terminal()
    show = ProgressDisplay(stream)
    show(MINILM, "model.safetensors", 45_000_000, 90_868_376)
    show(MINILM, "model.safetensors", 45_100_000, 90_868_376)
    show(MINILM, "model.safetensors", 90_868_376, 90_868_376)
    drawn = f"Downloading {MINILM}/model.safetensors (90.9 MB) 49%"
    done = f"Downloaded {MINILM}/model.safetensors (90.9 MB)"
    assert stream.getvalue() == f"\r{drawn}\r{done.ljust(len(drawn))}\n"


def test_reranker_scores_in_input_order_and_ranks_best_first(reranker):
    documents = ["Bananas are yellow.", "Berlin is the capital of Germany."]
    scores = reranker.score("capital of Germany", documents)
    assert scores.shape == (2,)
    assert scores.dtype == np.float32
    assert scores[1] > scores[0]
    best = [(1, float(scores[1])), (0, float(scores[0]))]
    assert reranker.rerank("capital of Germany", documents) == best
    assert reranker.rerank("capital of Germany", documents, top_k=1) == best[:1]
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("capital of Germany", documents, top_k=-1)
    assert reranker.name == RERANKER
    assert repr(reranker) == f"Reranker({RERANKER!r}, device={reranker.device!r})"


def test_every_error_shares_one_base_class():
    for error in (DownloadError, ModelError, UnsupportedError):
        assert issubclass(error, KodamaEmbedError)
    assert DownloadError.__module__ == "kodamaembed"

import base64
import io
import json
import threading
import urllib.error
import urllib.request
from email.message import Message
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from names import MINILM, RERANKER
from openai import OpenAI

from kodamaembed import server

TEXTS = ["Berlin is the capital of Germany.", "Bananas are yellow."]


@pytest.fixture(scope="module")
def endpoint(embedder, reranker):
    httpd = server.make_server([embedder, reranker], "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def request(endpoint, path, body=None, method=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    headers = {"Content-Type": "application/json"} if data is not None else {}
    call = urllib.request.Request(endpoint + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(call) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_embeddings_follow_the_openai_schema(endpoint, embedder):
    status, body = request(endpoint, "/v1/embeddings", {"model": MINILM, "input": TEXTS})
    assert status == 200
    assert (body["object"], body["model"]) == ("list", MINILM)
    assert [(item["object"], item["index"]) for item in body["data"]] == [
        ("embedding", 0),
        ("embedding", 1),
    ]
    vectors = np.array([item["embedding"] for item in body["data"]], np.float32)
    np.testing.assert_array_equal(vectors, embedder.embed_passages(TEXTS))
    usage = body["usage"]
    assert usage["prompt_tokens"] == usage["total_tokens"] > len(TEXTS)


def test_the_openai_client_reads_the_embeddings(endpoint, embedder):
    client = OpenAI(base_url=f"{endpoint}/v1", api_key="unused")
    response = client.embeddings.create(model=MINILM, input=TEXTS)
    vectors = np.array([item.embedding for item in response.data], np.float32)
    np.testing.assert_array_equal(vectors, embedder.embed_passages(TEXTS))
    assert response.model == MINILM
    assert response.usage.total_tokens > len(TEXTS)


def test_base64_embeddings_decode_to_the_same_floats(endpoint, embedder):
    status, body = request(
        endpoint, "/v1/embeddings", {"input": "capital of Germany", "encoding_format": "base64"}
    )
    assert status == 200
    vector = np.frombuffer(base64.b64decode(body["data"][0]["embedding"]), dtype="<f4")
    np.testing.assert_array_equal(vector, embedder.embed_passages(["capital of Germany"])[0])


@pytest.mark.parametrize("input_type", ["query", "search_query", "passage", "document"])
def test_input_type_names_the_side_of_the_search(endpoint, embedder, input_type):
    status, body = request(endpoint, "/v1/embeddings", {"input": TEXTS, "input_type": input_type})
    assert status == 200
    expected = (
        embedder.embed_queries(TEXTS)
        if "query" in input_type
        else embedder.embed_passages(TEXTS)
    )
    vectors = np.array([item["embedding"] for item in body["data"]], np.float32)
    np.testing.assert_array_equal(vectors, expected)


@pytest.mark.parametrize(
    ("body", "status", "words"),
    [
        ({"model": "BAAI/bge-small-en-v1.5", "input": "x"}, 404, "BAAI/bge-small-en-v1.5"),
        ({"model": RERANKER, "input": "x"}, 404, "not an embedding model"),
        ({"input": [1, 2, 3]}, 400, "input"),
        ({}, 400, "input"),
        ({"input": "x", "encoding_format": "float16"}, 400, "encoding_format"),
        ({"input": "x", "input_type": "image"}, 400, "input_type"),
        ({"input": "x", "dimensions": 128}, 400, "384"),
        ({"input": "x", "dimensions": 384.0}, 400, "dimensions"),
        ({"input": "\ud800"}, 400, "Unicode"),
    ],
)
def test_bad_embedding_requests_get_openai_shaped_errors(endpoint, body, status, words):
    code, reply = request(endpoint, "/v1/embeddings", body)
    assert code == status
    assert words in reply["error"]["message"]
    assert reply["error"]["type"] in ("invalid_request_error", "not_found_error")


def test_invalid_json_is_a_bad_request(endpoint):
    code, reply = request(endpoint, "/v1/embeddings", raw=b"{not json")
    assert code == 400
    assert "JSON" in reply["error"]["message"]


@pytest.mark.parametrize(
    ("fields", "status"),
    [
        ([], 411),
        ([("Content-Length", "-1")], 400),
        ([("Content-Length", "2"), ("Content-Length", "2")], 400),
        ([("Content-Length", str(server._MAX_BODY_BYTES + 1))], 413),
        ([("Transfer-Encoding", "chunked"), ("Content-Length", "2")], 400),
    ],
)
def test_invalid_framing_is_rejected_before_reading(fields, status):
    headers = Message()
    for name, value in fields:
        headers[name] = value
    source = Mock()
    with pytest.raises(server._RequestError) as caught:
        server._Handler._body(SimpleNamespace(headers=headers, rfile=source))
    assert caught.value.status == status
    source.read.assert_not_called()


def test_invalid_request_target_gets_a_bad_request():
    with pytest.raises(server._RequestError) as caught:
        server._Handler._dispatch(SimpleNamespace(path="http://[invalid"), "GET")
    assert caught.value.status == 400


def test_content_length_counts_bytes():
    headers = Message()
    headers["Content-Length"] = "0002"
    assert server._Handler._body(SimpleNamespace(headers=headers, rfile=io.BytesIO(b"{}more"))) == {}
    with pytest.raises(server._RequestError) as caught:
        server._Handler._body(SimpleNamespace(headers=headers, rfile=io.BytesIO(b"{")))
    assert caught.value.status == 400


def test_rerank_preserves_logit_order_when_relevance_rounds_to_one():
    model = Mock(spec=server.Reranker)
    model.name = RERANKER
    model._score.return_value = np.array([40, 50, 50, -710], dtype=np.float32), 4
    service = server._Service([model])
    with np.errstate(all="raise"):
        reply = service.rerank({"query": "q", "documents": ["a", "b", "c", "d"]})
    assert [row["index"] for row in reply["results"]] == [1, 2, 0, 3]
    assert [row["relevance_score"] for row in reply["results"]] == [1.0, 1.0, 1.0, np.exp(-710.0)]


def test_rerank_returns_the_best_documents_first(endpoint, reranker):
    documents = ["Bananas are yellow.", {"text": "Berlin is the capital of Germany."}]
    status, body = request(
        endpoint,
        "/v1/rerank",
        {"query": "capital of Germany", "documents": documents, "top_n": 1, "return_documents": True},
    )
    assert status == 200
    assert body["model"] == RERANKER
    (best,) = body["results"]
    assert best["index"] == 1
    assert best["document"] == {"text": "Berlin is the capital of Germany."}
    logit = reranker.score("capital of Germany", ["Berlin is the capital of Germany."])[0]
    assert best["relevance_score"] == pytest.approx(1 / (1 + np.exp(-float(logit))))
    assert body["usage"]["total_tokens"] > 0


def test_rerank_without_top_n_scores_every_document(endpoint):
    status, body = request(
        endpoint, "/v1/rerank", {"query": "capital of Germany", "documents": TEXTS}
    )
    assert status == 200
    assert [result["index"] for result in body["results"]] == [0, 1]
    assert all("document" not in result for result in body["results"])


def test_models_list_what_is_loaded(endpoint):
    status, body = request(endpoint, "/v1/models")
    assert status == 200
    assert {model["id"] for model in body["data"]} == {MINILM, RERANKER}


def test_health_and_unknown_routes(endpoint):
    assert request(endpoint, "/health") == (200, {"status": "ok"})
    assert request(endpoint, "/v1/chat/completions", {"x": 1})[0] == 404
    assert request(endpoint, "/v1/embeddings", method="GET")[0] == 405

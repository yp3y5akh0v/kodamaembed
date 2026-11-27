"""An HTTP server for loaded models: OpenAI embeddings and Cohere and Jina reranking."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Iterable
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TypeVar
from urllib.parse import urlsplit

import numpy as np

from . import Embedder, KodamaEmbedError, Reranker

__all__ = ["make_server"]

_Body = dict[str, Any]
_Model = TypeVar("_Model", Embedder, Reranker)

_MAX_BODY_BYTES = 64 * 1024 * 1024
_QUERY_TYPES = ("query", "search_query")
_PASSAGE_TYPES = ("passage", "document", "search_document")


def make_server(models: Iterable[Embedder | Reranker], host: str, port: int) -> ThreadingHTTPServer:
    """A server for `models` on `host` and `port`; port 0 picks a free one.
    Call ``serve_forever`` to run it."""
    service = _Service(models)
    return ThreadingHTTPServer((host, port), partial(_Handler, service=service))


class _RequestError(Exception):
    """A request the server refuses, answered as an OpenAI error object."""

    def __init__(
        self,
        status: HTTPStatus,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.param = param
        self.code = code
        self.headers = headers or {}

    def body(self) -> _Body:
        kind = "server_error" if self.status >= 500 else "invalid_request_error"
        return {"error": {"message": str(self), "type": kind, "param": self.param, "code": self.code}}


def _bad_request(message: str, param: str) -> _RequestError:
    return _RequestError(HTTPStatus.BAD_REQUEST, message, param=param)


class _Service:
    """Answers requests as JSON objects, independent of HTTP."""

    def __init__(self, models: Iterable[Embedder | Reranker]) -> None:
        self.models: dict[str, Embedder | Reranker] = {}
        for model in models:
            key = model.name.lower()
            if key in self.models:
                raise ValueError(f"{model.name} is loaded twice; serve each model once")
            self.models[key] = model
        if not self.models:
            raise ValueError("a server needs at least one model")
        self.embedders = {key: m for key, m in self.models.items() if isinstance(m, Embedder)}
        self.rerankers = {key: m for key, m in self.models.items() if isinstance(m, Reranker)}
        self.started = int(time.time())
        self.gets: dict[str, Callable[[], _Body]] = {
            "/health": self.health,
            "/v1/models": self.list_models,
        }
        self.posts: dict[str, Callable[[_Body], _Body]] = {
            "/v1/embeddings": self.embeddings,
            "/v1/rerank": self.rerank,
        }

    def health(self) -> _Body:
        return {"status": "ok"}

    def list_models(self) -> _Body:
        return {
            "object": "list",
            "data": [
                {"id": model.name, "object": "model", "created": self.started, "owned_by": "kodamaembed"}
                for model in self.models.values()
            ],
        }

    def embeddings(self, body: _Body) -> _Body:
        model = self._choose(body, self.embedders, "an embedding model")
        texts = body.get("input")
        if isinstance(texts, str):
            texts = [texts]
        if not (isinstance(texts, list) and all(isinstance(text, str) for text in texts)):
            raise _bad_request("input must be a string or a list of strings", "input")
        encoding = _option(body, "encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise _bad_request("encoding_format must be float or base64", "encoding_format")
        dimensions = body.get("dimensions")
        if dimensions is not None and (type(dimensions) is not int or dimensions != model.dim):
            raise _bad_request(
                f"{model.name} returns {model.dim} dimensions and cannot shorten them", "dimensions"
            )
        input_type = _option(body, "input_type", "passage")
        if input_type not in _QUERY_TYPES + _PASSAGE_TYPES:
            choices = ", ".join(_QUERY_TYPES + _PASSAGE_TYPES)
            raise _bad_request(f"input_type must be one of {choices}", "input_type")
        vectors, tokens = model._encode(texts, query=input_type in _QUERY_TYPES)
        if encoding == "base64":
            rows = [base64.b64encode(row.astype("<f4").tobytes()).decode("ascii") for row in vectors]
        else:
            rows = vectors.tolist()
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": row} for index, row in enumerate(rows)
            ],
            "model": model.name,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    def rerank(self, body: _Body) -> _Body:
        model = self._choose(body, self.rerankers, "a reranker")
        query = body.get("query")
        if not isinstance(query, str):
            raise _bad_request("query must be a string", "query")
        documents = body.get("documents")
        if not isinstance(documents, list):
            raise _bad_request("documents must be a list", "documents")
        texts = [_document_text(document) for document in documents]
        top_n = body.get("top_n")
        if top_n is not None and (type(top_n) is not int or top_n < 1):
            raise _bad_request("top_n must be a positive integer", "top_n")
        return_documents = _option(body, "return_documents", False)
        if not isinstance(return_documents, bool):
            raise _bad_request("return_documents must be true or false", "return_documents")
        logits, tokens = model._score(query, texts)
        scores = logits.astype(np.float64)
        with np.errstate(over="ignore", under="ignore"):
            relevance = 1 / (1 + np.exp(-scores))
            # exp(score) preserves subnormal probabilities when exp(-score) overflows.
            underflow = relevance == 0
            relevance[underflow] = np.exp(scores[underflow])
        results = []
        for index in np.argsort(-logits, kind="stable")[:top_n]:
            result: _Body = {"index": int(index), "relevance_score": float(relevance[index])}
            if return_documents:
                result["document"] = {"text": texts[index]}
            results.append(result)
        return {"model": model.name, "results": results, "usage": {"total_tokens": tokens}}

    def _choose(self, body: _Body, models: dict[str, _Model], kind: str) -> _Model:
        """The model a request names, or the only model of its kind."""
        name = body.get("model")
        if name is None and len(models) == 1:
            return next(iter(models.values()))
        served = ", ".join(model.name for model in models.values()) or "none"
        if name is None:
            raise _bad_request(f"model is required; served: {served}", "model")
        if not isinstance(name, str):
            raise _bad_request("model must be a string", "model")
        model = models.get(name.lower())
        if model is not None:
            return model
        if name.lower() in self.models:
            reason = f"{name} is not {kind}; served: {served}"
        else:
            reason = f"the model {name} is not served; served: {served}"
        raise _RequestError(HTTPStatus.NOT_FOUND, reason, param="model", code="model_not_found")


def _option(body: _Body, key: str, default: Any) -> Any:
    """A field's value, or `default` when it is missing or null."""
    value = body.get(key)
    return default if value is None else value


def _document_text(document: Any) -> str:
    if isinstance(document, str):
        return document
    if isinstance(document, dict) and isinstance(document.get("text"), str):
        return document["text"]
    raise _bad_request("each document must be a string or an object with a text string", "documents")


class _Handler(BaseHTTPRequestHandler):
    """Frames the service's JSON objects as HTTP."""

    def __init__(self, *args: Any, service: _Service, **kwargs: Any) -> None:
        self.service = service
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        self._answer("GET")

    def do_POST(self) -> None:
        self._answer("POST")

    def _answer(self, method: str) -> None:
        try:
            reply = self._dispatch(method)
        except _RequestError as error:
            self._send(error.status, error.body(), error.headers)
        else:
            self._send(HTTPStatus.OK, reply, {})

    def _dispatch(self, method: str) -> _Body:
        try:
            path = urlsplit(self.path).path
        except ValueError:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "invalid request target") from None
        gets, posts = self.service.gets, self.service.posts
        if method == "GET" and path in gets:
            action = gets[path]
        elif method == "POST" and path in posts:
            action = partial(posts[path], self._body())
        elif path in gets or path in posts:
            allow = "GET" if path in gets else "POST"
            raise _RequestError(
                HTTPStatus.METHOD_NOT_ALLOWED, f"{path} accepts {allow} requests", headers={"Allow": allow}
            )
        else:
            raise _RequestError(HTTPStatus.NOT_FOUND, f"no route for {path}")
        try:
            return action()
        except UnicodeEncodeError:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "input text must be valid Unicode") from None
        except KodamaEmbedError as error:
            raise _RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, str(error)) from error

    def _body(self) -> _Body:
        if self.headers.get("Transfer-Encoding") is not None:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Transfer-Encoding is unsupported")
        lengths = self.headers.get_all("Content-Length")
        if not lengths:
            raise _RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        length = lengths[0].strip(" \t")
        if len(lengths) != 1 or not length.isascii() or not length.isdecimal():
            raise _RequestError(HTTPStatus.BAD_REQUEST, "expected one decimal Content-Length")
        digits = length.lstrip("0") or "0"
        if len(digits) > len(str(_MAX_BODY_BYTES)) or int(digits) > _MAX_BODY_BYTES:
            raise _RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"the body exceeds {_MAX_BODY_BYTES} bytes"
            )
        size = int(digits)
        payload = self.rfile.read(size)
        if len(payload) != size:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "body is shorter than Content-Length")
        try:
            body = json.loads(payload)
        except (ValueError, RecursionError) as error:
            raise _RequestError(HTTPStatus.BAD_REQUEST, f"the body is not valid JSON: {error}") from None
        if not isinstance(body, dict):
            raise _RequestError(HTTPStatus.BAD_REQUEST, "the body must be a JSON object")
        return body

    def _send(self, status: HTTPStatus, reply: _Body, headers: dict[str, str]) -> None:
        payload = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

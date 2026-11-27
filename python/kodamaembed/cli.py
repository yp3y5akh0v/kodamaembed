"""The ``kodamaembed`` command: list, download and run models, or serve them."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from itertools import chain
from typing import NamedTuple

import numpy as np

from . import Embedder, KodamaEmbedError, Reranker, __version__, download, list_models
from ._progress import format_size


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command in `argv` (by default the process arguments) and
    returns its exit status."""
    request = _parser().parse_args(argv)
    command = _COMMANDS[request.command]
    # parse_intermixed_args requires a parser without subcommands.
    args = command.parser(request.command).parse_intermixed_args(request.arguments)
    try:
        return command.run(args)
    except (KodamaEmbedError, ValueError, OSError) as error:
        print(f"kodamaembed: error: {error}", file=sys.stderr)
        return 1


class _Command(NamedTuple):
    summary: str
    parser: Callable[[str], argparse.ArgumentParser]
    run: Callable[[argparse.Namespace], int]


def _parser() -> argparse.ArgumentParser:
    commands = "\n".join(f"  {name:<10}{command.summary}" for name, command in _COMMANDS.items())
    parser = argparse.ArgumentParser(
        prog="kodamaembed",
        description="Text embeddings and reranking.",
        epilog=f"commands:\n{commands}\n\n'kodamaembed COMMAND --help' shows a command's options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"kodamaembed {__version__}")
    parser.add_argument("command", choices=_COMMANDS, metavar="COMMAND")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def _command_parser(name: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"kodamaembed {name}", description=_COMMANDS[name].summary)


def _worker_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= sys.maxsize:
        raise argparse.ArgumentTypeError("threads must be a positive, addressable count")
    return count


def _port(value: str) -> int:
    port = int(value)
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def _add_device_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:N, vulkan or vulkan:N (default: %(default)s)"
    )
    parser.add_argument(
        "--precision",
        choices=("float", "quantized"),
        default="float",
        help="numerics of embedding models (default: %(default)s)",
    )
    parser.add_argument("--threads", type=_worker_count, help="run on the processor with this many threads")


def _models(args: argparse.Namespace) -> int:
    rows = [("NAME", "KIND", "DIM", "TOKENS", "DOWNLOAD")]
    rows += [
        (
            model.name,
            model.kind,
            "-" if model.dim is None else str(model.dim),
            str(model.max_length),
            format_size(model.download_size),
        )
        for model in list_models()
    ]
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    return 0


def _download_parser(name: str) -> argparse.ArgumentParser:
    parser = _command_parser(name)
    parser.add_argument("model")
    return parser


def _download(args: argparse.Namespace) -> int:
    print(download(args.model))
    return 0


def _embed_parser(name: str) -> argparse.ArgumentParser:
    parser = _command_parser(name)
    parser.add_argument("model")
    parser.add_argument("texts", nargs="*", metavar="text", help="texts; without any, each line of stdin")
    parser.add_argument("--query", action="store_true", help="embed the texts as search queries")
    parser.add_argument(
        "--output", metavar="FILE", help="save the vectors to a .npy file instead of printing JSON lines"
    )
    _add_device_options(parser)
    return parser


def _embed(args: argparse.Namespace) -> int:
    texts = args.texts or (line.rstrip("\r\n") for line in sys.stdin)
    model = Embedder(args.model, device=args.device, precision=args.precision, threads=args.threads)
    stream = model.embed_queries_stream if args.query else model.embed_passages_stream
    batches = stream(texts)
    if args.output:
        np.save(args.output, np.concatenate([np.empty((0, model.dim), np.float32), *batches]))
        return 0
    for index, vector in enumerate(chain.from_iterable(batches)):
        print(json.dumps({"index": index, "embedding": vector.tolist()}))
    return 0


def _serve_parser(name: str) -> argparse.ArgumentParser:
    parser = _command_parser(name)
    parser.add_argument("models", nargs="+", metavar="model")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: %(default)s)")
    parser.add_argument("--port", type=_port, default=8000, help="port to listen on (default: %(default)s)")
    _add_device_options(parser)
    return parser


def _serve(args: argparse.Namespace) -> int:
    from .server import make_server

    rerankers = {model.name.lower() for model in list_models() if model.kind == "reranker"}
    models = [
        Reranker(name, device=args.device, threads=args.threads)
        if name.lower() in rerankers
        else Embedder(name, device=args.device, precision=args.precision, threads=args.threads)
        for name in args.models
    ]
    with make_server(models, args.host, args.port) as httpd:
        host, port = httpd.server_address[:2]
        names = ", ".join(model.name for model in models)
        print(f"Serving {names} on http://{host}:{port}", file=sys.stderr)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


_COMMANDS = {
    "models": _Command("list the models that load by name", _command_parser, _models),
    "download": _Command("download a model and print its directory", _download_parser, _download),
    "embed": _Command("print or save the vectors of texts", _embed_parser, _embed),
    "serve": _Command(
        "serve models over HTTP: OpenAI embeddings, Cohere and Jina reranking", _serve_parser, _serve
    ),
}

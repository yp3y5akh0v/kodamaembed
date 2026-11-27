"""Export a local BERT or ModernBERT checkpoint for the CUDA and TensorRT query benchmarks."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


# What the exporter names every node of a rotary position module. FP16 holds every whole number
# only up to 2,048, so a module's positions and angles stay FP32 in the FP16 graph.
ROTARY = "rotary_emb"


def package_versions():
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "transformers", "optimum", "optimum-onnx", "onnx")}
    try:
        packages["onnxruntime-gpu"] = importlib.metadata.version("onnxruntime-gpu")
    except importlib.metadata.PackageNotFoundError:
        packages["onnxruntime"] = importlib.metadata.version("onnxruntime")
    return packages


def drop_repeated_nodes(graph):
    """Keeps one of each node the FP16 converter wrote more than once: among the nodes it leaves in
    FP32, every reader of a value gets a cast of its own, all under one name."""
    seen = set()
    for node in list(graph.node):
        written = node.SerializeToString(deterministic=True)
        if written in seen:
            graph.node.remove(node)
        seen.add(written)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pooled", action="store_true", help="also export normalized FP32 [batch, hidden] outputs")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir must be a local checkpoint directory")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from optimum.exporters.onnx import main_export
    from onnxruntime.transformers.optimizer import optimize_model

    input_hashes = {}
    for path in sorted(args.model_dir.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".txt", ".model"):
            with path.open("rb") as source:
                input_hashes[path.relative_to(args.model_dir).as_posix()] = hashlib.file_digest(source, "sha256").hexdigest()
    main_export(str(args.model_dir), output=str(args.out_dir), task="feature-extraction",
                device="cpu", local_files_only=True, trust_remote_code=False)
    model = optimize_model(str(args.out_dir / "model.onnx"), model_type="bert",
                           use_gpu=True, opt_level=0)
    rotary = [node.name for node in model.model.graph.node if ROTARY in node.name]
    model.convert_float_to_float16(keep_io_types=True, node_block_list=rotary)
    drop_repeated_nodes(model.model.graph)
    model.save_model_to_file(str(args.out_dir / "model-gpu-fp16.onnx"))
    if args.pooled:
        from types import SimpleNamespace
        from _matched import prepare
        for source, target in (("model.onnx", "model-pooled.onnx"),
                               ("model-gpu-fp16.onnx", "model-gpu-fp16-pooled.onnx")):
            prepare(SimpleNamespace(onnx=args.out_dir / source, out=args.out_dir / target))
    hashes = {}
    for path in sorted(args.out_dir.iterdir()):
        if path.is_file():
            with path.open("rb") as source:
                hashes[path.name] = hashlib.file_digest(source, "sha256").hexdigest()
    fp16 = {"model_type": "bert", "opt_level": 0, "keep_io_types": True}
    # Only a graph with rotary positions leaves nodes in FP32.
    if rotary:
        fp16["fp32_rotary_nodes"] = len(rotary)
    receipt = {
        "schema_version": 1,
        "packages": package_versions(),
        "task": "feature-extraction",
        "pooled": args.pooled,
        "input_files_sha256": input_hashes,
        "fp16": fp16,
        "files_sha256": hashes,
    }
    with (args.out_dir / "export.json").open("x", encoding="utf-8") as destination:
        json.dump(receipt, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()

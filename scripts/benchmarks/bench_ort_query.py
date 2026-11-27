"""Queries through ONNX Runtime's CUDA, TensorRT or CPU provider: a fixture's text or a sequence of
changing lengths to normalized FP32 host vectors, or without either one fixed-shape query on the GPU."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time

from _matched import add_scheduling_arguments


def nonnegative_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("seconds must be finite and nonnegative")
    return seconds


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def sha256(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "tensorrt", "cpu"), required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=positive_int, default=16)
    parser.add_argument("--hidden", type=positive_int, default=1024)
    parser.add_argument("--vocab", type=positive_int, default=30522)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--level", type=int, choices=range(6), default=3)
    parser.add_argument("--settle", type=nonnegative_seconds, default=30.0)
    parser.add_argument("--warm", type=nonnegative_seconds, default=1.0)
    parser.add_argument("--fixture", type=Path, help="matched text-to-normalized-host-vector fixture")
    parser.add_argument("--sequence", type=Path,
                        help="a fixture of texts and their call order: each text timed alone, three passes on a "
                             "fresh session that takes any length")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--export-receipt", type=Path)
    parser.add_argument("--threads", type=positive_int, default=8)
    add_scheduling_arguments(parser)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.fixture and args.sequence:
        parser.error("--fixture and --sequence are separate benchmarks")
    if args.fixture:
        if args.model_dir is None or args.export_receipt is None:
            parser.error("--fixture requires --model-dir and --export-receipt")
        from _matched import bench
        args.scope = "end-to-end"
        bench(args)
        return
    if args.sequence:
        if args.model_dir is None or args.export_receipt is None:
            parser.error("--sequence requires --model-dir and --export-receipt")
        if args.warm != parser.get_default("warm"):
            parser.error("--sequence starts on a fresh session and takes no --warm")
        from _sequence import bench
        bench(args)
        return
    if args.model_dir or args.export_receipt or args.prepare_only or args.provider == "cpu":
        parser.error("--model-dir, --export-receipt, --prepare-only and --provider cpu require --fixture or "
                     "--sequence")
    if not args.onnx.is_file():
        parser.error("--onnx must be an existing file")
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.provider == "tensorrt" and args.cache is None:
        parser.error("--cache is required for TensorRT")
    if args.out.exists():
        parser.error("--out already exists; use a new receipt path")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from _matched import configure_scheduling, scheduling_snapshot
    scheduling = configure_scheduling(args.cpus, args.priority)
    import numpy as np
    import onnx
    import onnxruntime as ort

    model_hashes = {args.onnx.name: sha256(args.onnx)}
    model = onnx.load(str(args.onnx), load_external_data=False)
    initializer_dtypes = dict(sorted(Counter(
        onnx.TensorProto.DataType.Name(tensor.data_type)
        for tensor in model.graph.initializer).items()))
    external_locations = {entry.value for tensor in model.graph.initializer
                          for entry in tensor.external_data if entry.key == "location"}
    for index, location in enumerate(sorted(external_locations)):
        model_hashes[f"external_{index}:{Path(location).name}"] = sha256(args.onnx.parent / location)
    del model
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "onnx", "onnxruntime-gpu")}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name and name.lower().startswith("nvidia-"):
            packages[name] = distribution.version
    dll_handles = []
    if args.provider == "tensorrt":
        import tensorrt_libs

        packages["tensorrt-cu12-libs"] = importlib.metadata.version("tensorrt-cu12-libs")
        library_dir = next(iter(tensorrt_libs.__path__))
        if os.name == "nt":
            dll_handles.append(os.add_dll_directory(library_dir))
        os.environ["PATH"] = library_dir + os.pathsep + os.environ.get("PATH", "")
    ort.preload_dlls()
    input_names = ("input_ids", "attention_mask", "token_type_ids")
    if args.provider == "cuda":
        provider = "CUDAExecutionProvider"
        options = {"device_id": args.device, "enable_cuda_graph": "1"}
    else:
        provider = "TensorrtExecutionProvider"
        args.cache.mkdir(parents=True, exist_ok=True)
        shape = ",".join(f"{name}:1x{args.tokens}" for name in input_names)
        options = {
            "device_id": args.device,
            "trt_fp16_enable": True,
            "trt_builder_optimization_level": args.level,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(args.cache.resolve()),
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(args.cache.resolve()),
            "trt_profile_min_shapes": shape,
            "trt_profile_opt_shapes": shape,
            "trt_profile_max_shapes": shape,
            "trt_cuda_graph_enable": True,
        }
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"{provider} is unavailable")
    session = ort.InferenceSession(str(args.onnx), providers=[(provider, options)])
    active_providers = session.get_providers()
    if not active_providers or active_providers[0] != provider:
        raise RuntimeError(f"requested provider was not selected: {active_providers}")
    session.disable_fallback()
    names = {value.name for value in session.get_inputs()}
    if not names.issubset(input_names) or "input_ids" not in names:
        raise RuntimeError(f"unsupported graph inputs: {sorted(names)}")
    ids = np.array([[(index * 7919) % args.vocab for index in range(args.tokens)]], np.int64)
    feed = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
    binding = session.io_binding()
    inputs = []
    for name, value in feed.items():
        if name in names:
            inputs.append(ort.OrtValue.ortvalue_from_numpy(value, "cuda", args.device))
            binding.bind_ortvalue_input(name, inputs[-1])
    output_info = session.get_outputs()[0]
    if output_info.type != "tensor(float)" or len(output_info.shape) != 3:
        raise RuntimeError("expected an FP32 last_hidden_state output")
    if isinstance(output_info.shape[2], int) and output_info.shape[2] != args.hidden:
        raise RuntimeError("--hidden differs from the model's output width")
    output = ort.OrtValue.ortvalue_from_shape_and_type([1, args.tokens, args.hidden], np.float32, "cuda", args.device)
    binding.bind_ortvalue_output(output_info.name, output)

    def run():
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()

    time.sleep(args.settle)
    started = time.perf_counter()
    warmed = 0
    while warmed < 20 or time.perf_counter() - started < args.warm:
        run()
        warmed += 1
    samples = []
    scheduling_before_timing = scheduling_snapshot(scheduling)
    timing_start = time.time_ns() / 1e6
    for _ in range(51):
        started = time.perf_counter()
        for _ in range(3):
            run()
        samples.append((time.perf_counter() - started) * 1e3 / 3)
    timing_end = time.time_ns() / 1e6
    scheduling_after_timing = scheduling_snapshot(scheduling)
    receipt = {
        "schema_version": 1,
        "samples_ms": samples,
        "timing_start_unix_ms": timing_start,
        "timing_end_unix_ms": timing_end,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "metadata": {
            "scheduling": scheduling, "scheduling_before_timing": scheduling_before_timing,
            "scheduling_after_timing": scheduling_after_timing,
            "engine": args.provider,
            "precision": "onnx-graph" if args.provider == "cuda" else "fp16-enabled",
            "onnx_initializer_dtype_counts": initializer_dtypes,
            "initializer_dtype_scope": "Top-level ONNX initializers; this does not establish runtime arithmetic precision.",
            "trt_fp16_enable": True if args.provider == "tensorrt" else None,
            "scope": "fixed token IDs and unpooled last_hidden_state on device; CUDA graph replay",
            "packages": packages,
            "onnx_sha256": model_hashes,
            "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "providers": active_providers,
            "device": args.device,
            "tokens": args.tokens,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "trt_builder_level": args.level if args.provider == "tensorrt" else None,
            "settle_seconds": args.settle,
            "warm_seconds": args.warm,
            "warm_calls": warmed,
            "calls_per_sample": 3,
        },
    }
    if args.provider == "tensorrt":
        receipt["metadata"]["trt_cache_sha256"] = {
            path.name: sha256(path) for path in sorted(args.cache.iterdir())
            if path.is_file() and path.suffix in (".engine", ".profile", ".timing")
        }
    with args.out.open("x", encoding="utf-8") as destination:
        json.dump(receipt, destination, indent=2, allow_nan=False)
        destination.write("\n")
    precision_label = "ONNX graph" if args.provider == "cuda" else "FP16 enabled"
    print(f"onnxruntime {ort.__version__} {args.provider} {precision_label}: one {args.tokens}-token sequence "
          f"median {receipt['median_ms']:.3f} ms (min {receipt['min_ms']:.3f}, max {receipt['max_ms']:.3f})")


if __name__ == "__main__":
    main()

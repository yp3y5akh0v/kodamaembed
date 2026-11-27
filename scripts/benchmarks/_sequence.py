"""Queries whose length changes every call: the sequence fixture, the timed passes and the leg
receipt of the ONNX Runtime and torch engines, the form bench_query_api --sequence also writes."""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import time

from _matched import (ENCODERS, FAMILIES, configure_scheduling, graph_metadata, read_fixture, save_json,
                      scheduling_snapshot, sha256, validate_embeddings, verify_export_receipt)

# The first pass meets every length on a fresh session; the later ones repeat it.
PASSES = 3
RETENTION = "a pass's returned outputs retained until its clock stops; validation and release outside timing"
CHECKPOINT_FILES = ("config.json", "tokenizer.json", "model.safetensors")
PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "tensorrt": "TensorrtExecutionProvider"}


def read_sequence(path):
    """The fixture, whose `order` names every text once: the call order."""
    fixture = read_fixture(path)
    order = fixture.get("order")
    if (not isinstance(order, list) or not all(type(index) is int for index in order)
            or sorted(order) != list(range(len(fixture["texts"])))):
        raise ValueError("sequence fixture order must name every text once")
    return fixture


def percentile(values, share):
    """Nearest rank: the smallest value with at least `share` of the values at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def pass_statistics(passes):
    """The first pass's statistics, then the later passes' together. The mean is a pass's time over
    its calls; the maximum holds a fresh session's first calls, which lie above the 99th percentile."""
    later = [sample for samples in passes[1:] for sample in samples]
    return {name: {"mean_ms": statistics.fmean(samples), "median_ms": statistics.median(samples),
                   "p99_ms": percentile(samples, 0.99), "max_ms": max(samples)}
            for name, samples in (("first_pass", passes[0]), ("later_passes", later))}


def bands(lengths, longest):
    """Token ranges to a quarter, to a half and to the whole of the model's longest input."""
    bounds = (min(lengths) - 1, longest // 4, longest // 2, longest)
    return [[low + 1, high] for low, high in zip(bounds, bounds[1:])]


def band_medians(passes, call_lengths, longest):
    """The later passes' median in each band; an engine's time against the length may be a
    staircase, which one median over all lengths hides."""
    result = []
    for low, high in bands(call_lengths, longest):
        samples = [sample for samples in passes[1:] for sample, length in zip(samples, call_lengths)
                   if low <= length <= high]
        if not samples:
            raise ValueError(f"the sequence holds no text of {low} to {high} tokens")
        result.append({"tokens": [low, high], "median_ms": statistics.median(samples)})
    return result


def profile_shapes(lengths, inputs):
    """One TensorRT engine for every length: the shortest, the middle and the longest of each of the
    graph's `inputs` as its profile."""
    ordered = sorted(lengths)
    return {kind: ",".join(f"{name}:1x{tokens}" for name in inputs)
            for kind, tokens in (("min", ordered[0]), ("opt", ordered[len(ordered) // 2]), ("max", ordered[-1]))}


def load_tokenizer(model_dir, family):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=family.max_length, stride=0, strategy="longest_first", direction="right")
    return tokenizer


def audit_tokens(tokenizer, fixture):
    """The prefixed texts, once their fresh tokens equal the fixture's, and the token digest as
    bench_query_api forms it."""
    prefix = FAMILIES[fixture["family"]].prefix(fixture["role"])
    texts = [prefix + text for text in fixture["texts"]]
    digest = hashlib.sha256()
    for text, ids, types in zip(texts, fixture["expected_ids"], fixture["expected_types"]):
        row = tokenizer.encode(text, add_special_tokens=True)
        if row.ids != ids or row.type_ids != types:
            raise ValueError("fixture IDs or types differ from tokenization")
        digest.update(struct.pack("<Q", len(ids)))
        digest.update(struct.pack(f"<{len(ids)}I", *ids))
        digest.update(struct.pack(f"<{len(ids)}I", *types))
    return texts, digest.hexdigest()


def feed(tokenizer, text):
    """One text's model inputs from a fresh tokenization, inside the timed call."""
    import numpy as np

    row = tokenizer.encode(text, add_special_tokens=True)
    ids = np.array([row.ids], dtype=np.int64)
    return {"input_ids": ids, "token_type_ids": np.array([row.type_ids], dtype=np.int64),
            "attention_mask": np.ones_like(ids)}


def timed_passes(order, run, hidden):
    """PASSES passes over `order`, a call a text: each call's milliseconds, each pass's wall-clock
    interval, the first pass's vectors in text order and the extreme norms."""
    import numpy as np

    passes, intervals, reference = [], [], [None] * len(order)
    low, high = math.inf, 0.0
    for _ in range(PASSES):
        samples, results = [], []
        pass_start = time.time_ns() / 1e6
        for index in order:
            started = time.perf_counter()
            result = run(index)
            samples.append((time.perf_counter() - started) * 1e3)
            results.append(result)
        intervals.append([pass_start, time.time_ns() / 1e6])
        for index, result in zip(order, results):
            norms = validate_embeddings(result, 1, hidden)
            low, high = min(low, float(norms.min())), max(high, float(norms.max()))
            if reference[index] is None:
                reference[index] = result
        passes.append(samples)
    return passes, intervals, np.concatenate(reference), low, high


def prepare_leg(args, extra_outputs=()):
    """The fixture, its family's geometry, the tokenizer, the audited texts and their digest, once
    no output of this leg exists."""
    if any(path.exists() for path in (args.out, args.out.with_suffix(".outputs.npy"), *extra_outputs)):
        raise ValueError("output artifacts already exist")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fixture = read_sequence(args.sequence)
    family = FAMILIES[fixture["family"]]
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if int(config["hidden_size"]) != family.hidden:
        raise ValueError("checkpoint width does not match fixture family")
    tokenizer = load_tokenizer(args.model_dir, family)
    texts, digest = audit_tokens(tokenizer, fixture)
    return fixture, family.hidden, tokenizer, texts, digest


def write_leg(args, fixture, digest, hidden, engine, scheduling, run, metadata):
    """Settles, times the passes under an unchanged scheduling and writes the leg's receipt and
    its first-pass vectors; `metadata` gives the engine's own fields once timing is over."""
    import numpy as np

    time.sleep(args.settle)
    before = scheduling_snapshot(scheduling)
    passes, intervals, vectors, low, high = timed_passes(fixture["order"], run, hidden)
    after = scheduling_snapshot(scheduling)
    output_path = args.out.with_suffix(".outputs.npy")
    with output_path.open("xb") as destination:
        np.save(destination, vectors, allow_pickle=False)
    save_json(args.out, {
        "schema_version": 1, "status": "complete", "passes_ms": passes, "pass_intervals_unix_ms": intervals,
        "timing_start_unix_ms": intervals[0][0], "timing_end_unix_ms": intervals[-1][1],
        "metadata": {
            "scheduling": scheduling, "scheduling_before_timing": before, "scheduling_after_timing": after,
            "engine": engine, "scope": "end-to-end",
            "family": fixture["family"], "role": fixture["role"], "label": fixture["label"],
            "fixture_sha256": sha256(args.sequence), "lengths": fixture["expected_lengths"],
            "calls_per_pass": len(fixture["order"]), "passes": PASSES, "input_ids_types_sha256": digest,
            "max_length": FAMILIES[fixture["family"]].max_length, "hidden": hidden,
            "checkpoint_sha256": {name: sha256(args.model_dir / name) for name in CHECKPOINT_FILES},
            "python_executable": sys.executable,
            "threads": args.threads, "settle_seconds": args.settle, "warm_calls": 0,
            "tokenizer_method": "encode", "caller_retention": RETENTION,
            "outputs": output_path.name, "outputs_sha256": sha256(output_path),
            "outputs_payload_sha256": hashlib.sha256(vectors.astype("<f4", copy=False).tobytes()).hexdigest(),
            "output_norm_min": low, "output_norm_max": high, **metadata(),
        },
    })
    later = pass_statistics(passes)["later_passes"]["mean_ms"]
    print(f"{fixture['label']} {engine} sequence: later passes' mean {later:.6f} ms", flush=True)


def bench(args):
    """bench_ort_query.py --sequence: a session that takes any length, so no graph replay and,
    for TensorRT, one engine over the fixture's range of lengths."""
    scheduling = configure_scheduling(args.cpus, args.priority)
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    import onnxruntime as ort

    if args.provider == "tensorrt" and args.cache is None:
        raise ValueError("TensorRT requires a separate --cache directory")
    if args.device != 0:
        raise ValueError("the sequence benchmark requires device 0 with one visible GPU")
    placement_path = args.out.with_suffix(".placement.json")
    fixture, hidden, tokenizer, texts, digest = prepare_leg(args, [placement_path])
    graph, graph_hashes = graph_metadata(args.onnx)
    export_lineage = verify_export_receipt(args.export_receipt, args.model_dir, graph_hashes)
    inputs = ENCODERS[FAMILIES[fixture["family"]].encoder].inputs
    if ({value.name for value in graph.graph.input} != set(inputs)
            or any(len(value.type.tensor_type.shape.dim) != 2 for value in graph.graph.input)):
        raise ValueError(f"the graph's inputs are not its encoder's two-dimensional {', '.join(inputs)}")
    del graph
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    # CPU placement is what the CPU provider measures, not a fallback to refuse.
    if args.provider != "cpu":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    wanted = PROVIDERS[args.provider]
    cuda = (PROVIDERS["cuda"], {"device_id": 0, "use_tf32": "0"})
    handles = []
    if args.provider == "cpu":
        providers = [(wanted, {})]
    elif args.provider == "cuda":
        providers = [cuda]
    else:
        import tensorrt_libs

        library = next(iter(tensorrt_libs.__path__))
        if os.name == "nt":
            handles.append(os.add_dll_directory(library))
        os.environ["PATH"] = library + os.pathsep + os.environ.get("PATH", "")
        args.cache.mkdir(parents=True, exist_ok=True)
        shapes = profile_shapes(fixture["expected_lengths"], inputs)
        providers = [(wanted, {
            "device_id": 0, "trt_fp16_enable": True, "trt_builder_optimization_level": args.level,
            "trt_engine_cache_enable": True, "trt_engine_cache_path": str(args.cache.resolve()),
            "trt_timing_cache_enable": True, "trt_timing_cache_path": str(args.cache.resolve()),
            **{f"trt_profile_{kind}_shapes": shape for kind, shape in shapes.items()}}), cuda]
    if args.provider != "cpu":
        ort.preload_dlls()
    if wanted not in ort.get_available_providers():
        raise RuntimeError(f"requested provider unavailable: {wanted}")
    load_started = time.perf_counter()
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=providers)
    load_seconds = time.perf_counter() - load_started
    session.disable_fallback()
    if session.get_providers()[0] != wanted:
        raise RuntimeError("requested provider was not selected")
    names = {value.name for value in session.get_inputs()}
    if names != set(inputs):
        raise RuntimeError("the session does not take the graph's inputs")
    outputs = session.get_outputs()
    if len(outputs) != 1 or outputs[0].type != "tensor(float)" or len(outputs[0].shape) != 2:
        raise RuntimeError("the graph must return only one FP32 [batch, hidden] pooled output")
    assigned = Counter(item.ep_name for item in session.get_provider_graph_assignment_info())
    allowed = {wanted, PROVIDERS["cuda"]} if args.provider == "tensorrt" else {wanted}
    placement = {"provider_counts": dict(sorted(assigned.items())), "unexpected": sorted(set(assigned) - allowed)}
    save_json(placement_path, placement)
    if placement["unexpected"] or wanted not in assigned:
        raise RuntimeError(f"unexpected execution-provider placement; inspect {placement_path}")

    def run(index):
        inputs = feed(tokenizer, texts[index])
        return session.run([outputs[0].name], {name: inputs[name] for name in names})[0]

    if args.prepare_only:
        validate_embeddings(run(fixture["order"][0]), 1, hidden)
        save_json(args.out, {"schema_version": 1, "status": "prepared", "load_seconds": load_seconds,
                             "providers": session.get_providers(), "fixture_sha256": sha256(args.sequence),
                             "onnx_sha256": graph_hashes, "placement_audit_sha256": sha256(placement_path)})
        print(f"Prepared {args.provider} {fixture['label']}", flush=True)
        return
    versions = {}
    for package in ("numpy", "onnx", "onnxruntime", "onnxruntime-gpu", "tokenizers", "tensorrt-cu12-libs"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {
        "scope_description": (
            "fresh text tokenization and prefix, graph forward/pooling/normalization, pooled FP32 vector, "
            "one text a call" if args.provider == "cpu" else
            "fresh text tokenization and prefix, H2D, graph forward/pooling/normalization, D2H pooled FP32 "
            "vector, one text a call"),
        "providers": session.get_providers(), "provider_options": session.get_provider_options(),
        "placement_audit_sha256": sha256(placement_path),
        "pooling_precision": "FLOAT ONNX mean and L2 graph; runtime internal precision not independently established",
        "onnx_sha256": graph_hashes, "export_lineage": export_lineage, "packages": versions,
        "harness_sha256": sha256(__file__), "load_seconds": load_seconds,
    }
    if args.provider == "tensorrt":
        metadata["trt_cache_sha256"] = {
            path.name: sha256(path) for path in sorted(args.cache.iterdir())
            if path.is_file() and path.suffix in (".engine", ".profile", ".timing")}
    write_leg(args, fixture, digest, hidden, args.provider, scheduling, run, lambda: metadata)

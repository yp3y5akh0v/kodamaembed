"""The benchmarked model families and the input, export and ONNX helpers every engine shares."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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

from _process import PRIORITIES, Scheduling, configure_current, current_scheduling, parse_cpus


@dataclass(frozen=True)
class Encoder:
    """What the engines know of one transformer type, named as its checkpoints' `model_type`."""

    # The type's name where a leg describes what it timed.
    title: str
    # The token arrays its graph takes, in the order a TensorRT profile names them.
    inputs: tuple[str, ...]
    # The class through which transformers runs its scaled-dot-product attention.
    attention: str


@dataclass(frozen=True)
class Family:
    """One benchmarked model: its checkpoint's geometry, the text every engine gives it and its cases."""

    model: str
    hidden: int
    # The cases' token counts, special tokens and prefix included; the longest is the model's limit.
    lengths: tuple[int, ...]
    encoder: str = "bert"
    query_prefix: str = ""
    passage_prefix: str = ""
    # Quantized precision, centered INT8, runs e5-large only.
    quantized: bool = False

    @property
    def max_length(self):
        return self.lengths[-1]

    def prefix(self, role):
        return self.query_prefix if role == "query" else self.passage_prefix


ENCODERS = {
    "bert": Encoder("BERT", ("input_ids", "token_type_ids", "attention_mask"), "BertSdpaSelfAttention"),
    # ModernBERT takes no segment IDs.
    "modernbert": Encoder("ModernBERT", ("input_ids", "attention_mask"), "ModernBertAttention"),
}
FAMILIES = {
    "minilm": Family("sentence-transformers/all-MiniLM-L6-v2", 384, (16, 33, 64, 65, 128, 256)),
    "e5": Family("intfloat/e5-large", 1024, (16, 33, 64, 65, 128, 256, 512),
                 query_prefix="query: ", passage_prefix="passage: ", quantized=True),
    "modernbert-embed": Family("nomic-ai/modernbert-embed-base", 768,
                               (16, 33, 64, 65, 128, 256, 512, 1024, 2048, 4096, 8192), "modernbert",
                               "search_query: ", "search_document: "),
}
ROLES = ("query", "passage")


def add_scheduling_arguments(parser):
    parser.add_argument("--cpus", type=parse_cpus, help="logical processors, such as 0-7,16 (Linux and Windows)")
    parser.add_argument("--priority", choices=PRIORITIES, default="inherit")


def configure_scheduling(cpus=None, priority="inherit"):
    """Applies the runner's scheduling inside this process and returns the readback."""
    return configure_current(Scheduling(cpus, priority))


def scheduling_snapshot(expected):
    actual = current_scheduling()
    if actual != expected:
        raise RuntimeError(f"process scheduling changed: {actual}")
    return actual


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
        destination.write("\n")


def validate_embeddings(result, batch, hidden):
    import numpy as np

    if result.shape != (batch, hidden) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise RuntimeError("invalid embeddings")
    norms = np.linalg.norm(result.astype(np.float64), axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-3:
        raise RuntimeError("embeddings are not L2 normalized")
    return norms


def save_preflight_output(path, result, batch, hidden):
    import numpy as np

    norms = validate_embeddings(result, batch, hidden)
    with Path(path).open("xb") as destination:
        np.save(destination, result, allow_pickle=False)
    return {"outputs": Path(path).name, "outputs_sha256": sha256(path),
            "outputs_payload_sha256": hashlib.sha256(result.astype("<f4", copy=False).tobytes()).hexdigest(),
            "output_norm_min": float(norms.min()), "output_norm_max": float(norms.max())}


def preflight_metadata(args, fixture, prepared, lengths, digest, hidden, maximum, result, engine):
    metadata = {
        "engine": engine, "scope": args.scope,
        "family": fixture["family"], "role": fixture["role"], "label": fixture["label"],
        "fixture_sha256": sha256(args.fixture), "shape": list(prepared["input_ids"].shape),
        "lengths": lengths, "real_tokens": sum(lengths),
        "padded_tokens": int(prepared["input_ids"].size), "input_ids_types_sha256": digest,
        "padded_array_sha256": {name: hashlib.sha256(value.astype("<i8", copy=False).tobytes()).hexdigest()
                                for name, value in prepared.items()},
        "max_length": maximum, "hidden": hidden,
        "checkpoint_sha256": {name: sha256(args.model_dir / name)
                              for name in ("config.json", "tokenizer.json", "model.safetensors")},
    }
    metadata.update(save_preflight_output(args.out.with_suffix(".outputs.npy"), result,
                                         prepared["input_ids"].shape[0], hidden))
    return metadata


def finite_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("seconds must be finite and nonnegative")
    return result


def graph_metadata(path):
    import onnx

    graph = onnx.load(str(path), load_external_data=False)
    hashes = {Path(path).name: sha256(path)}
    locations = sorted({entry.value for tensor in graph.graph.initializer
                        for entry in tensor.external_data if entry.key == "location"})
    for location in locations:
        external = (Path(path).parent / location).resolve()
        if not external.is_relative_to(Path(path).parent.resolve()):
            raise ValueError("external ONNX weights must stay inside the graph directory")
        hashes[location] = sha256(external)
    return graph, hashes


def prepare(args):
    import onnx
    from onnx import TensorProto, helper

    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".json").exists():
        raise ValueError("pooled graph or receipt already exists")
    if args.onnx.resolve() == args.out.resolve():
        raise ValueError("pooled graph must be a new artifact")
    graph, hashes = graph_metadata(args.onnx)
    onnx.external_data_helper.load_external_data_for_model(graph, str(args.onnx.parent))
    opset = next(item.version for item in graph.opset_import if item.domain in ("", "ai.onnx"))
    if opset < 13:
        raise ValueError("pooling helper requires ONNX opset 13 or newer")
    if "attention_mask" not in {item.name for item in graph.graph.input}:
        raise ValueError("the graph requires an attention_mask input")
    hidden = graph.graph.output[0]
    dimensions = hidden.type.tensor_type.shape.dim
    if len(dimensions) != 3 or dimensions[2].dim_value < 1:
        raise ValueError("expected [batch, sequence, hidden] first output")
    width = dimensions[2].dim_value
    prefix = "matched_pool_"
    names = {name for node in graph.graph.node for name in (*node.input, *node.output)}
    names.update(item.name for item in graph.graph.initializer)
    if any(name.startswith(prefix) for name in names):
        raise ValueError("graph already contains the benchmark pooling namespace")
    def name(value):
        return prefix + value
    graph.graph.initializer.extend([
        helper.make_tensor(name("expand_axes"), TensorProto.INT64, [1], [-1]),
        helper.make_tensor(name("sum_axes"), TensorProto.INT64, [1], [1]),
        helper.make_tensor(name("minimum_count"), TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor(name("minimum_norm"), TensorProto.FLOAT, [], [1e-12]),
    ])
    nodes = [
        helper.make_node("Cast", [hidden.name], [name("hidden_f32")], name=name("hidden_cast"), to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], [name("mask_f32")], name=name("mask_cast"), to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", [name("mask_f32"), name("expand_axes")], [name("mask")], name=name("expand")),
        helper.make_node("Mul", [name("hidden_f32"), name("mask")], [name("masked")], name=name("apply_mask")),
        helper.make_node("ReduceSum", [name("masked"), name("sum_axes")], [name("sum")], name=name("sum_hidden"), keepdims=0),
        helper.make_node("ReduceSum", [name("mask"), name("sum_axes")], [name("count")], name=name("sum_mask"), keepdims=0),
        helper.make_node("Clip", [name("count"), name("minimum_count")], [name("safe_count")], name=name("count_clip")),
        helper.make_node("Div", [name("sum"), name("safe_count")], [name("mean")], name=name("mean_pool")),
    ]
    norm_arguments = {"axes": [1]} if opset < 18 else {}
    norm_inputs = [name("mean")] if opset < 18 else [name("mean"), name("sum_axes")]
    nodes.extend([
        helper.make_node("ReduceL2", norm_inputs, [name("norm")], name=name("l2"), keepdims=1, **norm_arguments),
        helper.make_node("Clip", [name("norm"), name("minimum_norm")], [name("safe_norm")], name=name("norm_clip")),
        helper.make_node("Div", [name("mean"), name("safe_norm")], [name("embeddings")], name=name("normalize")),
    ])
    graph.graph.node.extend(nodes)
    del graph.graph.output[:]
    graph.graph.output.append(helper.make_tensor_value_info(name("embeddings"), TensorProto.FLOAT, ["batch", width]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(graph, str(args.out))
    onnx.checker.check_model(str(args.out))
    save_json(args.out.with_suffix(args.out.suffix + ".json"), {
        "source_hashes": hashes, "pooled_sha256": sha256(args.out), "hidden": width,
        "opset": opset, "pooling": "masked mean and L2 normalization, FLOAT graph operations",
        "runtime_precision": "Execution provider optimization can select different internal arithmetic; graph types alone do not prove it.",
    })
    print(f"Prepared {args.out}", flush=True)


def read_fixture(path):
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    if fixture.get("family") not in FAMILIES or fixture.get("role") not in ROLES:
        raise ValueError(f"fixture requires family {'|'.join(FAMILIES)} and role {'|'.join(ROLES)}")
    texts = fixture.get("texts")
    if not isinstance(texts, list) or not texts or not all(isinstance(value, str) for value in texts):
        raise ValueError("fixture texts must be a nonempty string list")
    if not isinstance(fixture.get("label"), str) or not fixture["label"]:
        raise ValueError("fixture requires a nonempty label")
    lengths, ids, types = (fixture.get(key) for key in ("expected_lengths", "expected_ids", "expected_types"))
    if not all(isinstance(value, list) and len(value) == len(texts) for value in (lengths, ids, types)):
        raise ValueError("fixture requires one expected token row per text")
    for length, row, kinds in zip(lengths, ids, types):
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise ValueError("fixture lengths must be positive integers")
        if not isinstance(row, list) or not isinstance(kinds, list) or len(row) != length or len(kinds) != length:
            raise ValueError("fixture expected token lengths differ")
        if not all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**31 for value in row + kinds):
            raise ValueError("fixture token IDs must fit nonnegative signed 32-bit integers")
    return fixture


def prepared_inputs(tokenizer, fixture, shape=None, audit=True):
    import numpy as np

    prefix = FAMILIES[fixture["family"]].prefix(fixture["role"])
    encode = getattr(tokenizer, "encode_batch_fast", tokenizer.encode_batch)
    encoded = encode([prefix + text for text in fixture["texts"]], add_special_tokens=True)
    lengths = [len(row.ids) for row in encoded]
    expected = (len(encoded), max(lengths))
    if shape is not None and expected != shape:
        raise RuntimeError("fresh tokenization changed the fixed input shape")
    ids = np.zeros(expected, dtype=np.int64)
    types = np.zeros_like(ids)
    mask = np.zeros_like(ids)
    digest = hashlib.sha256() if audit else None
    for index, row in enumerate(encoded):
        size = lengths[index]
        if not size or len(row.type_ids) != size:
            raise ValueError("tokenizer returned invalid token/type lengths")
        ids[index, :size], types[index, :size], mask[index, :size] = row.ids, row.type_ids, 1
        if digest is not None:
            digest.update(struct.pack("<Q", size))
            digest.update(struct.pack("<" + "I" * size, *row.ids))
            digest.update(struct.pack("<" + "I" * size, *row.type_ids))
    return {"input_ids": ids, "token_type_ids": types, "attention_mask": mask}, lengths, digest.hexdigest() if digest is not None else None


def validate_expected_tokens(fixture, prepared, lengths):
    if fixture.get("expected_lengths") != lengths:
        raise ValueError("fixture expected_lengths differ from tokenization")
    for key, array in (("expected_ids", "input_ids"), ("expected_types", "token_type_ids")):
        actual = [prepared[array][row, :size].tolist() for row, size in enumerate(lengths)]
        if fixture.get(key) != actual:
            raise ValueError(f"fixture {key} differ from tokenization")


def audit_profile(path, allowed):
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    placements = []
    for event in events:
        arguments = event.get("args", {})
        if event.get("cat") == "Node" and arguments.get("provider"):
            placements.append({"name": event.get("name"), "operator": arguments.get("op_name"),
                               "provider": arguments["provider"]})
    if not placements:
        raise RuntimeError("profiling did not establish execution-provider placement")
    unexpected = [item for item in placements if item["provider"] not in allowed]
    report = {"profile": str(path), "profile_sha256": sha256(path), "nodes": placements,
              "provider_counts": dict(Counter(item["provider"] for item in placements)),
              "unexpected": unexpected}
    return report


def assignment_info(session):
    def describe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [describe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): describe(item) for key, item in value.items()}
        result = {}
        for key in dir(value):
            if key.startswith("_"):
                continue
            item = getattr(value, key)
            if not callable(item):
                result[key] = describe(item)
        return result or str(value)
    method = getattr(session, "get_provider_graph_assignment_info", None)
    return describe(method()) if method is not None else {"status": "API unavailable"}


def bench(args):
    scheduling = configure_scheduling(args.cpus, args.priority)
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    if args.out.exists():
        raise ValueError("receipt already exists; use a new run path")
    output_path = args.out.with_suffix(".outputs.npy")
    audit_path = args.out.with_suffix(".placement.json")
    if output_path.exists() or audit_path.exists():
        raise ValueError("output artifacts already exist")
    if args.provider == "tensorrt" and args.cache is None:
        raise ValueError("TensorRT requires a separate --cache directory")
    if args.device != 0:
        raise ValueError("matched benchmarks require device 0 with one visible GPU")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fixture = read_fixture(args.fixture)
    family = FAMILIES[fixture["family"]]
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    hidden, maximum = int(config["hidden_size"]), family.max_length
    if hidden != family.hidden:
        raise ValueError("checkpoint width does not match fixture family")
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=maximum, stride=0, strategy="longest_first", direction="right")
    prepared, lengths, digest = prepared_inputs(tokenizer, fixture)
    validate_expected_tokens(fixture, prepared, lengths)
    shape = prepared["input_ids"].shape
    graph, graph_hashes = graph_metadata(args.onnx)
    export_lineage = verify_export_receipt(args.export_receipt, args.model_dir, graph_hashes)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(args.out.with_suffix(".ort-profile"))
    # CPU placement is what the CPU provider measures, not a fallback to refuse.
    if args.provider != "cpu":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    inputs = ENCODERS[family.encoder].inputs
    if {value.name for value in graph.graph.input} != set(inputs):
        raise ValueError(f"the graph's inputs are not its encoder's: {', '.join(inputs)}")
    dynamic_dimensions = {}
    for value in graph.graph.input:
        if len(value.type.tensor_type.shape.dim) != 2:
            raise ValueError("a graph input must have two dimensions")
        for axis, dim in enumerate(value.type.tensor_type.shape.dim):
            if dim.dim_param:
                previous = dynamic_dimensions.setdefault(dim.dim_param, shape[axis])
                if previous != shape[axis]:
                    raise ValueError("graph shares incompatible symbolic input dimensions")
            elif dim.dim_value and dim.dim_value != shape[axis]:
                raise ValueError("fixed graph input dimension differs from fixture")
    for symbol, size in dynamic_dimensions.items():
        options.add_free_dimension_override_by_name(symbol, size)
    del graph
    handles = []
    if args.provider == "cpu":
        wanted, provider_options = "CPUExecutionProvider", {}
    elif args.provider == "cuda":
        wanted = "CUDAExecutionProvider"
        provider_options = {"device_id": 0, "enable_cuda_graph": "1", "use_tf32": "0"}
    else:
        import tensorrt_libs

        library = next(iter(tensorrt_libs.__path__))
        if os.name == "nt":
            handles.append(os.add_dll_directory(library))
        os.environ["PATH"] = library + os.pathsep + os.environ.get("PATH", "")
        wanted = "TensorrtExecutionProvider"
        args.cache.mkdir(parents=True, exist_ok=True)
        profile = ",".join(f"{name}:{shape[0]}x{shape[1]}" for name in inputs)
        provider_options = {
            "device_id": 0, "trt_fp16_enable": True, "trt_builder_optimization_level": args.level,
            "trt_engine_cache_enable": True, "trt_engine_cache_path": str(args.cache.resolve()),
            "trt_timing_cache_enable": True, "trt_timing_cache_path": str(args.cache.resolve()),
            "trt_profile_min_shapes": profile, "trt_profile_opt_shapes": profile,
            "trt_profile_max_shapes": profile, "trt_cuda_graph_enable": True,
        }
    if args.provider != "cpu":
        ort.preload_dlls()
    if wanted not in ort.get_available_providers():
        raise RuntimeError(f"requested provider unavailable: {wanted}")
    load_started = time.perf_counter()
    providers = [(wanted, provider_options)]
    if args.provider == "tensorrt":
        providers.append(("CUDAExecutionProvider", {"device_id": 0, "use_tf32": "0"}))
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
        raise RuntimeError("matched graph must return only one FP32 [batch, hidden] pooled output")
    if isinstance(outputs[0].shape[1], int) and outputs[0].shape[1] != hidden:
        raise RuntimeError("pooled width differs from checkpoint")
    if args.provider == "cpu":
        def run():
            feed, _, _ = prepared_inputs(tokenizer, fixture, shape, audit=False)
            return session.run([outputs[0].name], {name: feed[name] for name in names})[0]
    else:
        values = {name: ort.OrtValue.ortvalue_from_numpy(prepared[name], "cuda", 0) for name in names}
        binding = session.io_binding()
        for name, value in values.items():
            binding.bind_ortvalue_input(name, value)
        output_value = ort.OrtValue.ortvalue_from_shape_and_type([shape[0], hidden], np.float32, "cuda", 0)
        binding.bind_ortvalue_output(outputs[0].name, output_value)

        def run():
            feed, _, _ = prepared_inputs(tokenizer, fixture, shape, audit=False)
            for name, value in values.items():
                value.update_inplace(feed[name])
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            return binding.copy_outputs_to_cpu()[0]

    prepared_output = run()
    profile_path = session.end_profiling()
    allowed = {wanted, "CUDAExecutionProvider"} if args.provider == "tensorrt" else {wanted}
    placement = audit_profile(profile_path, allowed)
    placement["graph_assignment"] = assignment_info(session)
    save_json(audit_path, placement)
    if placement["unexpected"]:
        raise RuntimeError(f"unexpected execution-provider placement; inspect {audit_path}")
    if wanted not in placement["provider_counts"]:
        raise RuntimeError("requested provider executed no profiled nodes")
    if args.prepare_only:
        metadata = preflight_metadata(args, fixture, prepared, lengths, digest, hidden,
                                      maximum, prepared_output, args.provider)
        metadata.update({"scheduling": scheduling, "providers": session.get_providers(),
                         "provider_options": session.get_provider_options(),
                         "placement_audit_sha256": sha256(audit_path), "onnx_sha256": graph_hashes, "export_lineage": export_lineage,
                         "harness_sha256": sha256(__file__)})
        save_json(args.out, {"schema_version": 1, "status": "prepared", "load_seconds": load_seconds,
                            "shape": list(shape), "providers": session.get_providers(),
                            "placement_audit_sha256": sha256(audit_path),
                            "fixture_sha256": sha256(args.fixture), "onnx_sha256": graph_hashes, "export_lineage": export_lineage,
                            "metadata": metadata})
        print(f"Prepared {args.provider} {fixture['label']}", flush=True)
        return
    time.sleep(args.settle)
    started = time.perf_counter()
    warmed = 0
    while warmed < 20 or time.perf_counter() - started < args.warm:
        run()
        warmed += 1
    samples = []
    intervals = []
    scheduling_before_timing = scheduling_snapshot(scheduling)
    timing_start = time.time_ns() / 1e6
    for _ in range(51):
        results = [None] * 3
        sample_start = time.time_ns() / 1e6
        started = time.perf_counter()
        for index in range(3):
            results[index] = run()
        samples.append((time.perf_counter() - started) * 1e3 / 3)
        intervals.append([sample_start, time.time_ns() / 1e6])
        result = results[-1]
        for candidate in results:
            validate_embeddings(candidate, shape[0], hidden)
    timing_end = time.time_ns() / 1e6
    scheduling_after_timing = scheduling_snapshot(scheduling)
    norms = validate_embeddings(result, shape[0], hidden)
    with output_path.open("xb") as destination:
        np.save(destination, result, allow_pickle=False)
    versions = {}
    for package in ("numpy", "onnx", "onnxruntime", "onnxruntime-gpu", "tokenizers", "tensorrt-cu12-libs"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    receipt = {
        "schema_version": 1, "samples_ms": samples, "sample_intervals_unix_ms": intervals,
        "timing_start_unix_ms": timing_start, "timing_end_unix_ms": timing_end,
        "median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
        "metadata": {
            "scheduling": scheduling, "scheduling_before_timing": scheduling_before_timing,
            "scheduling_after_timing": scheduling_after_timing,
            "engine": args.provider, "scope": args.scope,
            "scope_description": (
                "fresh text tokenization, prefix, padding, graph forward/pooling/normalization, pooled FP32 vector"
                if args.provider == "cpu" else "fresh text tokenization, prefix, padding, H2D, "
                "graph forward/pooling/normalization, D2H pooled FP32 vector"),
            "family": fixture["family"], "role": fixture["role"], "label": fixture["label"],
            "fixture_sha256": sha256(args.fixture), "shape": list(shape), "lengths": lengths,
            "real_tokens": sum(lengths), "padded_tokens": int(prepared["input_ids"].size),
            "input_ids_types_sha256": digest,
            "padded_array_sha256": {name: hashlib.sha256(value.astype("<i8", copy=False).tobytes()).hexdigest() for name, value in prepared.items()},
            "max_length": maximum, "hidden": hidden, "providers": session.get_providers(),
            "provider_options": session.get_provider_options(), "placement_audit_sha256": sha256(audit_path),
            "pooling_precision": "FLOAT ONNX mean and L2 graph; runtime internal precision not independently established",
            "onnx_sha256": graph_hashes, "export_lineage": export_lineage,
            "checkpoint_sha256": {name: sha256(args.model_dir / name) for name in ("config.json", "tokenizer.json", "model.safetensors")},
            "harness_sha256": sha256(__file__), "python_executable": sys.executable, "packages": versions,
            "threads": args.threads, "settle_seconds": args.settle, "warm_seconds": args.warm,
            "tokenizer_method": "encode_batch_fast" if hasattr(tokenizer, "encode_batch_fast") else "encode_batch",
            "warm_calls": warmed, "calls_per_sample": 3, "load_seconds": load_seconds,
            "caller_retention": "three returned outputs retained until the sample clock stops; validation and release outside timing",
            "outputs": output_path.name, "outputs_sha256": sha256(output_path),
            "outputs_payload_sha256": hashlib.sha256(result.astype("<f4", copy=False).tobytes()).hexdigest(),
            "output_norm_min": float(norms.min()), "output_norm_max": float(norms.max()),
        },
    }
    if args.provider == "tensorrt":
        receipt["metadata"]["trt_cache_sha256"] = {
            path.name: sha256(path) for path in sorted(args.cache.iterdir())
            if path.is_file() and path.suffix in (".engine", ".profile", ".timing")}
    validate_receipt(receipt, args.provider, sha256(args.fixture), fixture["family"], fixture["role"])
    save_json(args.out, receipt)
    print(f"{fixture['label']} {args.provider} {args.scope}: median {receipt['median_ms']:.6f} ms", flush=True)


def verify_export_receipt(path, model_dir, graph_hashes):
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    inputs = receipt.get("input_files_sha256", {})
    outputs = receipt.get("files_sha256", {})
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        if inputs.get(name) != sha256(Path(model_dir) / name):
            raise ValueError(f"export receipt does not identify this checkpoint: {name}")
    for name, digest in graph_hashes.items():
        if outputs.get(name) != digest:
            raise ValueError(f"export receipt does not identify this graph artifact: {name}")
    return {"receipt": Path(path).name, "receipt_sha256": sha256(path),
            "checkpoint_and_graph_hashes_match": True}


def validate_receipt(receipt, engine, fixture_hash, family, role):
    values = receipt.get("samples_ms")
    if not isinstance(values, list) or len(values) != 51 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and value > 0 for value in values):
        raise ValueError("receipt requires 51 finite positive samples")
    start, end = (receipt.get(key) for key in ("timing_start_unix_ms", "timing_end_unix_ms"))
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (start, end)) or end <= start:
        raise ValueError("receipt has invalid timing bounds")
    metadata = receipt.get("metadata", {})
    if metadata.get("engine") != engine or metadata.get("fixture_sha256") != fixture_hash:
        raise ValueError("receipt engine or fixture identity differs")
    if metadata.get("family") != family or metadata.get("role") != role:
        raise ValueError("receipt family or role differs")
    if metadata.get("scope") != "end-to-end" or metadata.get("calls_per_sample") != 3:
        raise ValueError("receipt does not measure the matched API boundary")

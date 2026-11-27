"""Text-to-normalized-host-vector benchmark of a BERT or ModernBERT checkpoint through PyTorch SDPA: FP16
in a CUDA graph on the GPU, or FP32 eager on the CPU; a sequence of changing lengths runs eagerly on both."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import NamedTuple

from _matched import (
    ENCODERS, FAMILIES, add_scheduling_arguments, configure_scheduling, scheduling_snapshot,
    finite_seconds, preflight_metadata, prepared_inputs, read_fixture, save_json, sha256,
    validate_expected_tokens, validate_receipt,
)

ENGINES = {"cuda": "torch-sdpa-graph", "cpu": "torch-cpu"}
# A CUDA graph replays one shape, so a sequence of changing lengths runs eagerly.
SEQUENCE_ENGINES = {"cuda": "torch-sdpa", "cpu": "torch-cpu"}


class Loaded(NamedTuple):
    """A checkpoint through transformers and what running it for one fixture takes."""

    model: object
    attention_classes: list
    # (model, inputs, all_tokens_valid) -> the forward for inputs of that shape and mask.
    specialize: object
    # How far a value of the specialized forward may stand from the ordinary one's, beside 0.1% of it.
    atol: float
    # The encoder's own settings, recorded with a leg.
    settings: dict


def bert_embeddings(model, inputs, all_tokens_valid):
    """Specialize BERT's mask control flow for one fixed fixture before capture."""
    import torch

    embeddings = model.embeddings(input_ids=inputs["input_ids"],
                                  token_type_ids=inputs["token_type_ids"])
    attention_mask = None
    if not all_tokens_valid:
        mask = inputs["attention_mask"][:, None, None, :].to(embeddings.dtype)
        inverted = 1.0 - mask
        attention_mask = inverted.masked_fill(inverted.to(torch.bool), torch.finfo(embeddings.dtype).min)
    hidden = model.encoder(embeddings, attention_mask=attention_mask,
                           use_cache=False, output_attentions=False,
                           output_hidden_states=False, return_dict=False)[0]
    return masked_mean_normalize(hidden, inputs["attention_mask"])


def bert_forward(model, _inputs, all_tokens_valid):
    """BERT's forward for inputs of one mask. Its usual mask helper converts torch.all(mask == 1) to
    a Python bool; the fixed choice removes that synchronization before capture."""
    return lambda fresh: bert_embeddings(model, fresh, all_tokens_valid)


def modernbert_forward(model, inputs, all_tokens_valid):
    """ModernBERT's forward for inputs of one shape and mask, both masks and the positions made
    beforehand: the model derives them on the host at every call, which a CUDA graph cannot record."""
    import torch

    with torch.inference_mode():
        everywhere, window = model._update_attention_mask(inputs["attention_mask"], output_attentions=False)
        positions = torch.arange(inputs["input_ids"].shape[1], device=inputs["input_ids"].device).unsqueeze(0)
    # No token is hidden from a layer that sees the whole text, so it takes no mask.
    if all_tokens_valid:
        everywhere = None

    def forward(fresh):
        hidden = model.embeddings(input_ids=fresh["input_ids"])
        for layer in model.layers:
            hidden = layer(hidden, attention_mask=everywhere, sliding_window_mask=window, position_ids=positions)[0]
        return masked_mean_normalize(model.final_norm(hidden), fresh["attention_mask"])

    return forward


def masked_mean_normalize(hidden, mask):
    import torch
    import torch.nn.functional as functional

    weights = mask.unsqueeze(-1).to(dtype=torch.float32)
    mean = (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return functional.normalize(mean, p=2, dim=1, eps=1e-12)


def validate_outputs(result, batch, hidden):
    import numpy as np

    if result.shape != (batch, hidden) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise RuntimeError("invalid final embeddings")
    norms = np.linalg.norm(result.astype(np.float64), axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-3:
        raise RuntimeError("final embeddings are not L2 normalized")
    return norms


def summarize_attention_trace(trace):
    events = trace.get("traceEvents", [])
    operators = Counter(event["name"] for event in events
                        if event.get("cat") == "cpu_op" and "scaled_dot_product" in event.get("name", ""))
    kernels = Counter(event["name"] for event in events if event.get("cat") == "kernel")
    if not operators or not kernels:
        raise RuntimeError("profile did not capture both SDPA operator and CUDA kernel evidence")
    return {"sdpa_operators": dict(operators), "cuda_kernels": dict(kernels)}


def placed(model, device):
    return model.cuda(0) if device == "cuda" else model


def load_bert(model_dir, device, dtype):
    from transformers import BertModel

    model = BertModel.from_pretrained(str(model_dir), local_files_only=True, use_safetensors=True,
                                      add_pooling_layer=False, attn_implementation="sdpa", torch_dtype=dtype).eval()
    if model.attn_implementation != "sdpa":
        raise RuntimeError("BERT SDPA was not selected")
    attention_classes = sorted({type(layer.attention.self).__name__ for layer in model.encoder.layer})
    return Loaded(placed(model, device), attention_classes, bert_forward, 2e-5, {})


def load_modernbert(model_dir, device, dtype):
    from transformers import ModernBertModel

    model = placed(ModernBertModel.from_pretrained(str(model_dir), local_files_only=True, use_safetensors=True,
                                                   attn_implementation="sdpa", torch_dtype=dtype).eval(), device)
    if model.config._attn_implementation != "sdpa":
        raise RuntimeError("ModernBERT SDPA was not selected")
    # As its own forward does first: the model compiles its embeddings and feed-forward blocks where it can.
    model._maybe_set_compile()
    attention_classes = sorted({type(layer.attn).__name__ for layer in model.layers})
    # Without a mask the layers that see the whole text run another attention kernel, whose FP16 sums
    # round differently: up to 2.1e-4 measured, where FP16 itself stands 1e-3 from FP32.
    atol = 1e-3 if device == "cuda" else 2e-5
    return Loaded(model, attention_classes, modernbert_forward, atol,
                  {"reference_compile": bool(model.config.reference_compile)})


LOADERS = {"bert": load_bert, "modernbert": load_modernbert}


def load_model(model_dir, device, encoder):
    """The checkpoint of `encoder`'s type with SDPA attention: FP16 parameters on the GPU, FP32 on the CPU."""
    import torch

    loaded = LOADERS[encoder](model_dir, device, torch.float16 if device == "cuda" else torch.float32)
    if loaded.attention_classes != [ENCODERS[encoder].attention]:
        raise RuntimeError(f"unexpected attention implementation: {loaded.attention_classes}")
    return loaded


def encoder_of(fixture, model_dir):
    """The fixture family's encoder, once the checkpoint is of that type and width; returns it with the width."""
    family = FAMILIES[fixture["family"]]
    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    if int(config["hidden_size"]) != family.hidden:
        raise ValueError("checkpoint width does not match fixture family")
    if config.get("model_type") != family.encoder or config.get("is_decoder", False):
        raise ValueError(f"benchmark requires an encoder-only {family.encoder} checkpoint")
    return family.encoder, family.hidden


def captured_forward(loaded, prepared, all_tokens_valid):
    """The specialized forward captured in one CUDA graph: its replay on fresh
    inputs, a profiler of one replay and one uncaptured forward, and the capture time."""
    import torch

    static_inputs = {name: torch.from_numpy(value).to("cuda:0") for name, value in prepared.items()}
    encode = loaded.specialize(loaded.model, static_inputs, all_tokens_valid)

    @torch.inference_mode()
    def forward():
        return encode(static_inputs)

    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph), torch.inference_mode():
        graph_output = forward()
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - capture_started

    @torch.inference_mode()
    def replay(feed):
        for name, value in static_inputs.items():
            value.copy_(torch.from_numpy(feed[name]))
        graph.replay()
        return graph_output.cpu().numpy()

    def profile(path):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            graph.replay()
            torch.cuda.synchronize()
            forward()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(path))

    return replay, profile, capture_seconds


def eager_forward(loaded, prepared, all_tokens_valid):
    """The specialized forward run eagerly on the CPU for fresh inputs."""
    import torch

    def tensors(feed):
        return {name: torch.from_numpy(value) for name, value in feed.items()}

    encode = loaded.specialize(loaded.model, tensors(prepared), all_tokens_valid)

    @torch.inference_mode()
    def run(feed):
        return encode(tensors(feed)).numpy()

    return run


def device_metadata(device, capture_seconds):
    """The timed device's settings and arithmetic, read after timing."""
    import torch

    if device == "cpu":
        return {"transformer_precision": "FP32 parameters and activations",
                "intra_op_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}
    return {"transformer_precision": "FP16 parameters and activations; PyTorch library accumulation rules",
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "flash_attention_built": torch.backends.cuda.is_flash_attention_available(),
            "cuda_build": torch.version.cuda, "cuda_arch_list": torch.cuda.get_arch_list(),
            "device_name": torch.cuda.get_device_name(0), "device_capability": list(torch.cuda.get_device_capability(0)),
            "capture_seconds": capture_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def bench(args):
    scheduling = configure_scheduling(args.cpus, args.priority)
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    import numpy as np
    import torch
    from tokenizers import Tokenizer

    engine = ENGINES[args.device]
    if args.out.exists():
        raise ValueError("receipt already exists; use a new run path")
    output_path = args.out.with_suffix(".outputs.npy")
    trace_path = args.out.with_suffix(".attention-trace.json")
    if output_path.exists() or trace_path.exists():
        raise ValueError("output artifact already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fixture = read_fixture(args.fixture)
    encoder, hidden = encoder_of(fixture, args.model_dir)
    maximum = FAMILIES[fixture["family"]].max_length
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=maximum, stride=0, strategy="longest_first", direction="right")
    prepared, lengths, digest = prepared_inputs(tokenizer, fixture)
    validate_expected_tokens(fixture, prepared, lengths)
    shape = prepared["input_ids"].shape
    all_tokens_valid = bool(np.all(prepared["attention_mask"] == 1))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA device is required")
        torch.cuda.set_device(0)
    load_started = time.perf_counter()
    loaded = load_model(args.model_dir, args.device, encoder)
    if args.device == "cuda":
        forward, profile, capture_seconds = captured_forward(loaded, prepared, all_tokens_valid)
    else:
        forward, profile, capture_seconds = eager_forward(loaded, prepared, all_tokens_valid), None, None
    load_seconds = time.perf_counter() - load_started

    def run():
        feed, _, _ = prepared_inputs(tokenizer, fixture, shape, audit=False)
        return forward(feed)

    specialized_check = forward(prepared)
    inputs = {name: torch.from_numpy(prepared[name]).to(args.device) for name in ENCODERS[encoder].inputs}
    with torch.inference_mode():
        ordinary_hidden = loaded.model(**inputs, return_dict=False)[0]
        ordinary_check = masked_mean_normalize(ordinary_hidden, inputs["attention_mask"]).cpu().numpy()
    max_reference_difference = float(np.max(np.abs(specialized_check - ordinary_check)))
    if not np.allclose(specialized_check, ordinary_check, rtol=1e-3, atol=loaded.atol):
        raise RuntimeError("specialized forward differs from the checkpoint's ordinary forward")
    validate_outputs(specialized_check, shape[0], hidden)
    profile_report = None
    if args.profile:
        profile(trace_path)
        profile_report = summarize_attention_trace(json.loads(trace_path.read_text(encoding="utf-8")))
        profile_report.update({"trace": trace_path.name, "trace_sha256": sha256(trace_path),
                               "scope": "one captured replay and one uncaptured specialized forward; SDPA operator events identify the uncaptured dispatch, CUDA kernel events cover both"})
    if args.prepare_only:
        metadata = preflight_metadata(args, fixture, prepared, lengths, digest, hidden,
                                      maximum, specialized_check, engine)
        metadata.update({"scheduling": scheduling, "ordinary_forward_max_abs_difference": max_reference_difference,
                         "attention_profile": profile_report, "harness_sha256": sha256(__file__),
                         "helper_sha256": sha256(Path(__file__).with_name("_matched.py"))})
        save_json(args.out, {"schema_version": 1, "status": "prepared", "engine": engine, "shape": list(shape),
                            "load_seconds": load_seconds, "capture_seconds": capture_seconds,
                            "fixture_sha256": sha256(args.fixture),
                            "ordinary_forward_max_abs_difference": max_reference_difference,
                            "attention_profile": profile_report, "metadata": metadata})
        print(f"Prepared {engine} {fixture['label']}", flush=True)
        return
    time.sleep(args.settle)
    warmed = 0
    started = time.perf_counter()
    while warmed < 20 or time.perf_counter() - started < args.warm:
        run()
        warmed += 1
    samples, intervals = [], []
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
            validate_outputs(candidate, shape[0], hidden)
    timing_end = time.time_ns() / 1e6
    scheduling_after_timing = scheduling_snapshot(scheduling)
    norms = validate_outputs(result, shape[0], hidden)
    with output_path.open("xb") as destination:
        np.save(destination, result, allow_pickle=False)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "transformers", "tokenizers", "safetensors")}
    title = ENCODERS[encoder].title
    receipt = {
        "schema_version": 1, "samples_ms": samples, "sample_intervals_unix_ms": intervals,
        "timing_start_unix_ms": timing_start, "timing_end_unix_ms": timing_end,
        "median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
        "metadata": {
            "scheduling": scheduling, "scheduling_before_timing": scheduling_before_timing,
            "scheduling_after_timing": scheduling_after_timing,
            "engine": engine, "scope": args.scope,
            "scope_description": (
                f"fresh text tokenization, prefix, padding, H2D, captured {title} SDPA forward/FP32 "
                "pooling/normalization, D2H pooled FP32 vector" if args.device == "cuda" else
                f"fresh text tokenization, prefix, padding, {title} SDPA forward/FP32 pooling/normalization, "
                "pooled FP32 vector"),
            "family": fixture["family"], "role": fixture["role"], "label": fixture["label"],
            "fixture_sha256": sha256(args.fixture), "shape": list(shape), "lengths": lengths,
            "real_tokens": sum(lengths), "padded_tokens": int(prepared["input_ids"].size),
            "input_ids_types_sha256": digest,
            "padded_array_sha256": {name: hashlib.sha256(value.astype("<i8", copy=False).tobytes()).hexdigest() for name, value in prepared.items()},
            "max_length": maximum, "hidden": hidden, "attention_classes": loaded.attention_classes,
            "attention_dispatch": "PyTorch SDPA automatic backend selection; not a claim of FlashAttention kernel selection",
            "attention_profile": profile_report,
            "all_tokens_valid": all_tokens_valid, "pooling_precision": "FP32 masked mean and L2 normalization",
            "ordinary_forward_max_abs_difference": max_reference_difference,
            **loaded.settings, **device_metadata(args.device, capture_seconds),
            "checkpoint_sha256": {name: sha256(args.model_dir / name) for name in ("config.json", "tokenizer.json", "model.safetensors")},
            "harness_sha256": sha256(__file__), "helper_sha256": sha256(Path(__file__).with_name("_matched.py")),
            "python_executable": sys.executable, "packages": versions,
            "threads": args.threads, "settle_seconds": args.settle, "warm_seconds": args.warm,
            "tokenizer_method": "encode_batch_fast" if hasattr(tokenizer, "encode_batch_fast") else "encode_batch",
            "warm_calls": warmed, "calls_per_sample": 3, "load_seconds": load_seconds,
            "caller_retention": "three returned outputs retained until the sample clock stops; validation and release outside timing",
            "outputs": output_path.name, "outputs_sha256": sha256(output_path),
            "outputs_payload_sha256": hashlib.sha256(result.astype("<f4", copy=False).tobytes()).hexdigest(),
            "output_norm_min": float(norms.min()), "output_norm_max": float(norms.max()),
        },
    }
    validate_receipt(receipt, engine, sha256(args.fixture), fixture["family"], fixture["role"])
    save_json(args.out, receipt)
    print(f"{fixture['label']} {engine} {args.scope}: median {receipt['median_ms']:.6f} ms", flush=True)


def sequence_bench(args):
    """--sequence: each text timed alone through the eager forward, three passes on a fresh model."""
    from _sequence import feed, prepare_leg, write_leg

    scheduling = configure_scheduling(args.cpus, args.priority)
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    import torch

    fixture, hidden, tokenizer, texts, digest = prepare_leg(args)
    encoder, _ = encoder_of(fixture, args.model_dir)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA device is required")
        torch.cuda.set_device(0)
    load_started = time.perf_counter()
    loaded = load_model(args.model_dir, args.device, encoder)
    load_seconds = time.perf_counter() - load_started
    device = "cuda:0" if args.device == "cuda" else "cpu"

    @torch.inference_mode()
    def run(index):
        inputs = {name: torch.from_numpy(value).to(device) for name, value in feed(tokenizer, texts[index]).items()}
        # One text carries no padding, so every token is valid.
        return loaded.specialize(loaded.model, inputs, True)(inputs).cpu().numpy()

    def metadata():
        title = ENCODERS[encoder].title
        return {
            "scope_description": (
                f"fresh text tokenization and prefix, H2D, eager {title} SDPA forward/FP32 pooling/normalization, "
                "D2H pooled FP32 vector, one text a call" if args.device == "cuda" else
                f"fresh text tokenization and prefix, eager {title} SDPA forward/FP32 pooling/normalization, pooled "
                "FP32 vector, one text a call"),
            "attention_classes": loaded.attention_classes,
            "attention_dispatch": "PyTorch SDPA automatic backend selection; not a claim of FlashAttention kernel selection",
            "all_tokens_valid": True, "pooling_precision": "FP32 masked mean and L2 normalization",
            **loaded.settings,
            **{name: value for name, value in device_metadata(args.device, None).items() if name != "capture_seconds"},
            "harness_sha256": sha256(__file__), "helper_sha256": sha256(Path(__file__).with_name("_sequence.py")),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "torch", "transformers", "tokenizers", "safetensors")},
            "load_seconds": load_seconds,
        }

    write_leg(args, fixture, digest, hidden, SEQUENCE_ENGINES[args.device], scheduling, run, metadata)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--sequence", type=Path,
                        help="a fixture of texts and their call order: each text timed alone, three passes on a "
                             "fresh model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=tuple(ENGINES), default="cuda")
    parser.add_argument("--settle", type=finite_seconds, default=30.0)
    parser.add_argument("--warm", type=finite_seconds, default=1.0)
    parser.add_argument("--threads", type=int, default=8)
    add_scheduling_arguments(parser)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--profile", action="store_true", help="record SDPA operators and CUDA kernels outside timing")
    parser.set_defaults(scope="end-to-end")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if (args.fixture is None) == (args.sequence is None):
        parser.error("exactly one of --fixture and --sequence is required")
    if args.sequence:
        if args.prepare_only or args.profile or args.warm != parser.get_default("warm"):
            parser.error("--sequence starts on a fresh model and takes no --warm, --prepare-only or --profile")
        sequence_bench(args)
        return
    if args.profile and args.device != "cuda":
        parser.error("--profile records CUDA kernels and requires --device cuda")
    bench(args)


if __name__ == "__main__":
    main()

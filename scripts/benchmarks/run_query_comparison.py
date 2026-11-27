"""Matched latency comparison: every engine turns the same text into a normalized FP32
host vector, one owned and checked process at a time, in alternating rounds."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import statistics
import sys
import time
import traceback

import _system
from _matched import FAMILIES, ROLES
from _process import PLATFORM, PRIORITIES, Scheduling, Tree, parse_cpus, satisfies
from _telemetry import (GATE_INTERVAL_SECONDS, GATE_SAMPLES, IDLE_CONTEXT_UTILIZATION, MAXIMUM_GAP_MS,
                        MONITOR_INTERVAL_MS, TIMING_MARGIN_MS, GpuWatch, Limits, Smi, check_timing)

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
EXECUTABLE_SUFFIX = ".exe" if PLATFORM == "windows" else ""
SAMPLES, CALLS = 51, 3
MATCHED_SCOPE = "fresh text to normalized FP32 host vectors"
CHECKPOINT_FILES = ("config.json", "tokenizer.json", "model.safetensors")
RETENTION = "three returned outputs retained"
CASE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,79}", re.ASCII)
ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*", re.ASCII)
DIGEST = re.compile(r"[0-9a-f]{64}", re.ASCII)
FINGERPRINT = re.compile(r"sha256:[0-9a-fA-F]{64}", re.ASCII)
# FILE_ATTRIBUTE_REPARSE_POINT: a Windows link or junction.
REPARSE_POINT = 0x400
CONTAINMENT = {"windows": "kill-on-close job object", "linux": "session, child subreaper and parent-death signal",
               "macos": "process group"}


QUANTIZED = tuple(name for name, family in FAMILIES.items() if family.quantized)


@dataclass(frozen=True)
class Engine:
    """Maps a benchmark engine name to its invocation and receipt."""

    name: str
    device: str
    receipt: str
    kind: str
    selector: str
    graph: str | None = None
    families: tuple[str, ...] = tuple(FAMILIES)


ENGINES = {engine.name: engine for engine in (
    Engine("ort-cuda", "gpu", "cuda", "ort", "cuda", "cuda_onnx"),
    Engine("trt-fp16", "gpu", "tensorrt", "ort", "tensorrt", "trt_onnx"),
    Engine("torch-sdpa-graph", "gpu", "torch-sdpa-graph", "torch", "cuda"),
    Engine("kodamaembed-cuda-fp16", "gpu", "kodamaembed", "rust", "cuda-fp16"),
    Engine("kodamaembed-cuda-centered", "gpu", "kodamaembed", "rust", "cuda-centered", families=QUANTIZED),
    Engine("ort-cpu", "cpu", "cpu", "ort", "cpu", "cpu_onnx"),
    Engine("torch-cpu", "cpu", "torch-cpu", "torch", "cpu"),
    Engine("kodamaembed-cpu-fp32", "cpu", "kodamaembed", "rust", "cpu-fp32"),
    Engine("kodamaembed-cpu-centered", "cpu", "kodamaembed", "rust", "cpu-centered", families=QUANTIZED),
)}
GROUPS = {
    "gpu": ("ort-cuda", "trt-fp16", "torch-sdpa-graph", "kodamaembed-cuda-fp16", "kodamaembed-cuda-centered"),
    # Centered INT8 runs on x86-64 processors only.
    "cpu": ("ort-cpu", "torch-cpu", "kodamaembed-cpu-fp32")
           + (("kodamaembed-cpu-centered",) if platform.machine().lower() in ("x86_64", "amd64") else ()),
}
DIAGNOSTIC_ENGINES = ("ort-cuda", "trt-fp16", "kodamaembed-synthetic", "kodamaembed-api")


@dataclass(frozen=True)
class LegSettings:
    """What every leg's command shares: timing, thread counts and scheduling."""

    settle: float
    # None where a leg starts on a fresh session and takes no warm-up.
    warm: float | None
    gpu_threads: int
    cpu_threads: int | None
    trt_level: int
    scheduling: Scheduling

    def scheduling_arguments(self):
        cpus = self.scheduling.cpus
        return (["--cpus", ",".join(map(str, cpus))] if cpus is not None else []) + [
            "--priority", self.scheduling.priority]


@dataclass
class Plan:
    case: str
    engine: str
    key: str
    inputs: dict = field(default_factory=dict)
    family: str | None = None
    role: str | None = None
    gpu: bool = False


def matched_command(engine, inputs, paths, settings, result, fixture="--fixture"):
    """The program and arguments of one matched leg writing its receipt to `result`; `fixture` is
    the option naming the leg's kind of fixture."""
    common = [fixture, inputs["fixture"], "--model-dir", inputs["checkpoint"], "--out", result,
              "--settle", repr(settings.settle)]
    if settings.warm is not None:
        common += ["--warm", repr(settings.warm)]
    threads = str(settings.cpu_threads if engine.device == "cpu" else settings.gpu_threads)
    if engine.kind == "rust":
        cpu = ["--threads", threads] if engine.device == "cpu" else []
        return paths["api_executable"], [*common, "--mode", engine.selector, *cpu]
    python = [*common, "--threads", threads, *settings.scheduling_arguments()]
    if engine.kind == "torch":
        return paths["torch_executable"], [paths["torch_benchmark"], *python, "--device", engine.selector]
    tensorrt = (["--cache", inputs["trt_cache"], "--level", str(settings.trt_level)]
                if engine.selector == "tensorrt" else [])
    return paths["python_executable"], [
        paths["python_benchmark"], *python, "--export-receipt", inputs["export_receipt"],
        "--onnx", inputs[engine.graph], "--provider", engine.selector, *tensorrt]


def sha256(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def json_timing(data, matched=True):
    if not isinstance(data, dict):
        raise ValueError("Timing receipt must be a JSON object.")
    samples, start, end = data.get("samples_ms"), data.get("timing_start_unix_ms"), data.get("timing_end_unix_ms")
    metadata = data.get("metadata")
    intervals = None
    if matched:
        if not finite_number(start) or not finite_number(end):
            raise ValueError("Matched timing requires finite start and end timestamps.")
        intervals = data.get("sample_intervals_unix_ms")
        if not isinstance(intervals, list) or len(intervals) != SAMPLES:
            raise ValueError("Matched timing requires 51 sample intervals.")
        previous = start
        for interval in intervals:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("Matched sample interval must contain two timestamps.")
            if not all(finite_number(value) for value in interval):
                raise ValueError("Matched sample timestamp must be a finite number.")
            if interval[0] < previous or interval[1] <= interval[0] or interval[1] > end:
                raise ValueError("Matched sample intervals overlap or exceed timing bounds.")
            previous = interval[1]
    return timing_summary(samples, start, end, metadata, intervals)


def read_timing(path, matched):
    """The 51 samples and their timed window from a JSON receipt or a CSV log."""
    path = Path(path)
    if path.suffix == ".json":
        return json_timing(json.loads(path.read_text(encoding="utf-8")), matched)
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("CSV timing output is empty.")
    samples = [float(row["latency_ms"]) for row in rows]
    start, end = float(rows[0]["start_unix_ms"]), float(rows[-1]["end_unix_ms"])
    previous = 0.0
    for index, row in enumerate(rows):
        row_start, row_end = float(row["start_unix_ms"]), float(row["end_unix_ms"])
        if (int(row["sample"]) != index or not math.isfinite(row_start) or not math.isfinite(row_end)
                or row_start < previous or row_end <= row_start):
            raise ValueError("CSV sample indices or timestamps are invalid.")
        previous = row_end
    return timing_summary(samples, start, end, None, None)


def timing_summary(samples, start, end, metadata, intervals):
    if (not isinstance(samples, list) or len(samples) != SAMPLES or not finite_number(start)
            or not finite_number(end) or end <= start):
        raise ValueError("Timing output must contain 51 samples and a finite, positive timing interval.")
    if not all(finite_number(sample) and sample > 0 for sample in samples):
        raise ValueError("Timing sample is not finite and positive.")
    return {"samples_ms": samples, "timing_start_unix_ms": start, "timing_end_unix_ms": end,
            "median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
            "metadata": metadata, "sample_intervals_unix_ms": intervals}


def assert_input_hashes(paths, expected):
    """Rehashes every input; the snapshot must name exactly these inputs."""
    if set(paths) != set(expected):
        raise ValueError("Input hash keys differ from the initial snapshot.")
    digests, current = {}, {}
    for name, path in paths.items():
        if not isinstance(expected[name], str) or not DIGEST.fullmatch(expected[name]):
            raise ValueError("Invalid initial input digest.")
        real = os.path.realpath(path)
        if real not in digests:
            digests[real] = sha256(path)
        if digests[real] != expected[name]:
            raise ValueError(f"Input changed during the rotation: {name}")
        current[name] = digests[real]
    return current


@contextmanager
def open_artifact(path):
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & REPARSE_POINT:
        raise ValueError("Artifact is a link or reparse point.")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Artifact must be a regular file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as source:
        if not os.path.samestat(info, os.fstat(source.fileno())):
            raise ValueError("Artifact changed while opening it.")
        yield source


def assert_artifact_hashes(directory, expected):
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Missing artifact digests.")
    for name, digest in expected.items():
        if (not isinstance(name, str) or not ARTIFACT_NAME.fullmatch(name)
                or not isinstance(digest, str) or not DIGEST.fullmatch(digest)):
            raise ValueError("Artifact names and digests must be local filenames and SHA256 strings.")
        with open_artifact(Path(directory) / name) as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError(f"Artifact digest differs: {name}")


def read_matched_result(path, engine, read=json_timing):
    """Derive the timing, by `read`, and the artifact digests from the same receipt bytes."""
    path = Path(path)
    with open_artifact(path) as source:
        payload = source.read()
    receipt = json.loads(payload.decode("utf-8"))
    timing = read(receipt)
    metadata = receipt.get("metadata") or {}
    stem = path.stem
    expected = {}
    if engine.kind == "rust":
        name = f"{stem}.vectors.f32"
        vectors = receipt.get("vectors") or {}
        if vectors.get("path") != name:
            raise ValueError("Rust vector artifact does not belong to this receipt.")
        expected[name] = vectors.get("sha256")
    else:
        name = f"{stem}.outputs.npy"
        if metadata.get("outputs") != name:
            raise ValueError("Python vector artifact does not belong to this receipt.")
        expected[name] = metadata.get("outputs_sha256")
        if engine.kind == "ort":
            expected[f"{stem}.placement.json"] = metadata.get("placement_audit_sha256")
    expected[path.name] = hashlib.sha256(payload).hexdigest()
    assert_artifact_hashes(path.parent, expected)
    return timing, expected


def check_python_scheduling(metadata, requested):
    """A Python engine's readbacks: one process, which kept the runner's scheduling."""
    first = metadata.get("scheduling")
    for name in ("scheduling", "scheduling_before_timing", "scheduling_after_timing"):
        state = metadata.get(name)
        if (not isinstance(state, dict) or not isinstance(first, dict) or type(state.get("process_id")) is not int
                or state["process_id"] <= 0 or state["process_id"] != first.get("process_id")):
            raise ValueError("Matched Python scheduling identity is absent or changed.")
        if not satisfies(state, requested):
            raise ValueError(f"Matched Python scheduling differs: {name}")


def accept_matched(timing, plan, hashes, token_hashes, requested):
    """The matched receipt's timed boundary, identity, inputs and lineage; raises unless all hold."""
    metadata = timing["metadata"]
    if (not isinstance(metadata, dict) or metadata.get("calls_per_sample") != CALLS
            or not str(metadata.get("caller_retention", "")).startswith(RETENTION)):
        raise ValueError("Matched receipt timed boundary differs.")
    accept_lineage(metadata, ENGINES[plan.engine], plan, hashes, token_hashes, requested)


def accept_lineage(metadata, engine, plan, hashes, token_hashes, requested):
    """A leg receipt's engine, fixture, scheduling, checkpoint, graph and tokens against the run's
    inputs; raises unless all hold."""
    if (metadata.get("engine") != engine.receipt or metadata.get("scope") != "end-to-end"
            or metadata.get("family") != plan.family or metadata.get("role") != plan.role
            or metadata.get("fixture_sha256") != hashes[f"{plan.case}/fixture"]):
        raise ValueError("Matched receipt engine or fixture differs.")
    if engine.kind in ("ort", "torch"):
        check_python_scheduling(metadata, requested)
    checkpoint = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint, dict) or any(
            checkpoint.get(name) != hashes[f"{plan.case}/{name}"] for name in CHECKPOINT_FILES):
        raise ValueError("Matched receipt checkpoint differs.")
    if engine.kind == "ort":
        graph_name = Path(plan.inputs[engine.graph]).name
        lineage = metadata.get("export_lineage") or {}
        if ((metadata.get("onnx_sha256") or {}).get(graph_name) != hashes[f"{plan.case}/{engine.graph}"]
                or lineage.get("receipt_sha256") != hashes[f"{plan.case}/export_receipt"]
                or lineage.get("checkpoint_and_graph_hashes_match") is not True):
            raise ValueError("Matched receipt graph or export lineage differs from the initial snapshot.")
    digest = metadata.get("input_ids_types_sha256")
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ValueError("Matched receipt lacks token evidence.")
    if token_hashes.setdefault(plan.case, digest) != digest:
        raise ValueError("Matched engines tokenized different model inputs.")
    if engine.kind == "rust" and metadata.get("mode") != engine.selector:
        raise ValueError("KodamaEmbed matched precision mode differs.")


def aggregate(plans, legs, rounds):
    result = {}
    for plan in plans:
        accepted = [leg for leg in legs if leg["key"] == plan.key and leg["valid"]]
        if len(accepted) != rounds:
            raise ValueError("The rotation is incomplete.")
        medians = [leg["timing"]["median_ms"] for leg in accepted]
        minimums = [leg["timing"]["min_ms"] for leg in accepted]
        result[plan.key] = {"valid_runs": len(accepted), "median_of_medians_ms": statistics.median(medians),
                            "best_median_ms": min(medians), "median_of_minimums_ms": statistics.median(minimums)}
    return result


def executable(path):
    """An absolute path to `path`; symbolic links stay unresolved so a venv interpreter keeps its venv."""
    found = shutil.which(str(path))
    if found is None:
        raise FileNotFoundError(f"executable not found: {path}")
    return os.path.abspath(found)


def interpreter(path, option, engine):
    if path is None:
        raise ValueError(f"{engine} requires {option}")
    return executable(path)


def selected_engines(text, engines=ENGINES, groups=GROUPS):
    """Engine names from a comma-separated list of engines and groups, in order, once each."""
    selected = []
    for item in text.split(","):
        for name in groups.get(item.strip(), (item.strip(),)):
            if name not in engines:
                raise ValueError(f"unknown engine: {name}")
            if name not in selected:
                selected.append(name)
    return selected


def selected_families(text):
    """The families a comma-separated list names; every family if none is asked for."""
    if text is None:
        return set(FAMILIES)
    names = {item.strip() for item in text.split(",")}
    for name in names - set(FAMILIES):
        raise ValueError(f"unknown family: {name}")
    return names


class Comparison:
    # The matched protocol's engines and the option naming its fixtures; a runner of another
    # protocol replaces them and the methods reading a leg's receipt.
    table, groups, fixture_option = ENGINES, GROUPS, "--fixture"

    def __init__(self, args):
        self.args = args
        self.output = Path(args.output).absolute()
        if self.output.exists() and (not self.output.is_dir() or any(self.output.iterdir())):
            raise ValueError("Output must be a new or empty directory.")
        self.output.mkdir(parents=True, exist_ok=True)
        self.matched = args.cases is not None
        self.requested = Scheduling(args.cpus, args.priority)
        self.settings = LegSettings(args.settle, args.warm, args.threads, args.cpu_threads, args.trt_level,
                                    self.requested)
        self.stage = "preparation"
        self.legs = []
        self.engines = []
        self.plans = []
        self.paths = {}
        self.token_hashes = {}
        self.gpu = None
        self.gpu_hardware = None
        self.diagnostic = None
        self.summary = {
            "schema_version": 1, "created_utc": now_utc(), "status": "preparing", "valid": False,
            "settings": None, "platform": {"system": PLATFORM, "process_containment": CONTAINMENT.get(PLATFORM)},
            "hardware": None, "speed_settings": None, "hashes": {}, "background_process_observations": [],
            "legs": self.legs, "aggregates": None, "failure": None,
        }

    def save(self):
        text = json.dumps(self.summary, indent=2, allow_nan=False)
        (self.output / "summary.json").write_text(text + "\n", encoding="utf-8")

    def run(self):
        self.save()
        try:
            self.prepare()
            self.rotate()
            self.complete()
        except Exception as error:
            self.fail(error)
            return 1
        return 0

    def prepare(self):
        self.paths = {"runner": Path(__file__).resolve(), "process_helper": HERE / "_process.py",
                      "telemetry_helper": HERE / "_telemetry.py", "system_helper": HERE / "_system.py",
                      "matched_helper": HERE / "_matched.py",
                      "cargo_lock": REPOSITORY / "Cargo.lock", "rust_toolchain": REPOSITORY / "rust-toolchain.toml"}
        if self.matched:
            self.prepare_matched()
        else:
            self.prepare_diagnostic()
        if any(plan.gpu for plan in self.plans):
            self.prepare_gpu()
        hashes = {}
        for name, path in list(self.paths.items()):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing required input: {name}")
            self.paths[name] = os.path.abspath(path)
            hashes[name] = sha256(path)
        self.summary["hashes"] = hashes
        self.summary["hardware"] = {"gpu": self.gpu_hardware, "cpu_names": _system.processor_names(),
                                    "os": _system.operating_system(), "python": platform.python_version()}
        self.summary["speed_settings"] = _system.speed_settings()
        self.summary["settings"] = self.settings_record()
        print("Engines: " + ", ".join(self.summary["settings"]["engines"]), flush=True)

    def add_api_inputs(self):
        self.paths.update(
            api_executable=executable(self.args.api_exe or
                                      REPOSITORY / f"target/release/examples/bench_query_api{EXECUTABLE_SUFFIX}"),
            api_example_source=REPOSITORY / "crates/kodamaembed/examples/bench_query_api.rs",
            centered_ptx=REPOSITORY / "crates/kodamaembed-cuda/ptx/repro.ptx")

    def prepare_matched(self):
        args = self.args
        cases = Path(args.cases).resolve()
        specification = json.loads(cases.read_text(encoding="utf-8"))
        if specification.get("schema_version") != 1 or not specification.get("cases"):
            raise ValueError("Cases requires schema_version 1 and a nonempty cases array.")
        self.engines = engines = selected_engines(args.engines, self.table, self.groups)
        families = selected_families(args.families)
        if any(self.table[name].device == "cpu" for name in engines) and args.cpu_threads is None:
            raise ValueError("CPU legs require --cpu-threads")
        self.paths["cases"] = cases
        for name in engines:
            engine = self.table[name]
            if engine.kind == "ort" and "python_executable" not in self.paths:
                self.paths.update(python_executable=interpreter(args.python, "--python", name),
                                  python_benchmark=HERE / "bench_ort_query.py")
            elif engine.kind == "torch" and "torch_executable" not in self.paths:
                self.paths.update(torch_executable=interpreter(args.torch_python, "--torch-python", name),
                                  torch_benchmark=HERE / "bench_torch_query.py")
            elif engine.kind == "rust" and "api_executable" not in self.paths:
                self.add_api_inputs()
                self.paths["api_matched_source"] = REPOSITORY / "crates/kodamaembed/examples/support/matched_query.rs"
        self.summary["cases"] = []
        names = set()
        for item in specification["cases"]:
            name = str(item.get("name", ""))
            if not CASE_NAME.fullmatch(name) or name in names:
                raise ValueError("Case names must be unique lowercase letters, digits and hyphens.")
            names.add(name)
            needed = {"checkpoint", "fixture"} | {self.table[engine].graph for engine in engines
                                                  if self.table[engine].graph}
            if any(self.table[engine].kind == "ort" for engine in engines):
                needed.add("export_receipt")
            if "trt-fp16" in engines:
                needed.add("trt_cache")
            inputs = {}
            for key in sorted(needed):
                value = item.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Case {name} is missing {key}.")
                inputs[key] = os.path.abspath(cases.parent / value)
            fixture = json.loads(Path(inputs["fixture"]).read_text(encoding="utf-8"))
            if (fixture.get("family") not in FAMILIES or fixture.get("role") not in ROLES
                    or not fixture.get("texts") or fixture.get("label") != name):
                raise ValueError(f"Invalid matched fixture for {name}.")
            lengths = fixture.get("expected_lengths")
            if args.max_tokens is not None and not lengths:
                raise ValueError(f"Case {name}'s fixture states no lengths for --max-tokens.")
            if fixture["family"] not in families or (args.max_tokens is not None and max(lengths) > args.max_tokens):
                continue
            for key in sorted(needed - {"checkpoint", "trt_cache"}):
                self.paths[f"{name}/{key}"] = inputs[key]
            for file in CHECKPOINT_FILES:
                self.paths[f"{name}/{file}"] = Path(inputs["checkpoint"]) / file
            self.summary["cases"].append({"name": name, "family": fixture["family"], "role": fixture["role"],
                                          "sequences": len(fixture["texts"]), "lengths": lengths})
            for engine in engines:
                if fixture["family"] in self.table[engine].families:
                    self.plans.append(Plan(name, engine, f"{name}/{engine}", inputs, fixture["family"],
                                           fixture["role"], self.table[engine].device == "gpu"))
        if not self.summary["cases"]:
            raise ValueError("--families and --max-tokens leave no case to run.")
        if not self.plans:
            raise ValueError("No selected engine runs these cases' models.")

    def prepare_diagnostic(self):
        args = self.args
        values = {"--python": args.python, "--cuda-onnx": args.cuda_onnx, "--trt-onnx": args.trt_onnx,
                  "--trt-cache": args.trt_cache, "--checkpoint": args.checkpoint}
        missing = [option for option, value in values.items() if value is None]
        if missing:
            raise ValueError("Diagnostic mode requires " + ", ".join(missing) + ".")
        if args.torch_python or args.engines != "gpu" or args.families or args.max_tokens:
            raise ValueError("--torch-python, --engines, --families and --max-tokens require --cases.")
        checkpoint = Path(args.checkpoint)
        self.add_api_inputs()
        self.paths.update(
            python_executable=executable(args.python), python_benchmark=HERE / "bench_ort_query.py",
            cuda_executable=executable(args.cuda_exe or
                                       REPOSITORY / f"target/release/examples/bench_cuda_query{EXECUTABLE_SUFFIX}"),
            cuda_example_source=REPOSITORY / "crates/kodamaembed-cuda/examples/bench_cuda_query.rs",
            cuda_onnx=args.cuda_onnx, trt_onnx=args.trt_onnx, checkpoint_config=checkpoint / "config.json",
            checkpoint_tokenizer=checkpoint / "tokenizer.json", checkpoint_weights=checkpoint / "model.safetensors")
        self.diagnostic = {"checkpoint": os.path.abspath(checkpoint), "trt_cache": os.path.abspath(args.trt_cache)}
        self.engines = list(DIAGNOSTIC_ENGINES)
        self.plans = [Plan("diagnostic", engine, engine, gpu=True) for engine in DIAGNOSTIC_ENGINES]

    def prepare_gpu(self):
        args = self.args
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            raise ValueError("CUDA_VISIBLE_DEVICES must be unset so CUDA and telemetry identify the same GPU.")
        smi = Smi([executable(args.nvidia_smi)], REPOSITORY)
        self.gpu_hardware = smi.hardware()
        minimum = args.min_power_w if args.min_power_w is not None else self.gpu_hardware["default_power_limit_w"]
        if minimum is None:
            raise ValueError("This GPU reports no default power limit; pass --min-power-w.")
        limits = Limits(minimum, args.max_temp, bool(args.allowed_idle_pid))
        self.gpu = GpuWatch(smi, limits, args.allowed_idle_pid, args.max_wait_minutes * 60)
        self.summary["background_process_observations"] = self.gpu.observations

    def settings_record(self):
        args = self.args
        gpu = self.gpu is not None
        return {
            "rounds": args.rounds, "tokens": None if self.matched else 16, "samples_per_run": SAMPLES,
            "calls_per_sample": CALLS, "settle_seconds": args.settle, "warm_seconds": args.warm,
            "cpus": list(self.requested.cpus) if self.requested.cpus is not None else None,
            "priority": self.requested.priority, "gpu_leg_python_threads": args.threads,
            "cpu_leg_threads": args.cpu_threads,
            "execution_speed_throttling": "disabled" if PLATFORM == "windows" else None,
            "max_gate_temperature_c": args.max_temp if gpu else None,
            "minimum_enforced_power_w": self.gpu.limits.minimum_power_w if gpu else None,
            "minimum_power_source": None if not gpu else "explicit" if args.min_power_w is not None else "gpu-default",
            "max_gate_minutes": args.max_wait_minutes, "max_run_minutes": args.max_run_minutes,
            "gate_consecutive_samples": GATE_SAMPLES, "gate_interval_ms": int(GATE_INTERVAL_SECONDS * 1000),
            "telemetry_interval_ms": MONITOR_INTERVAL_MS, "timing_margin_ms": TIMING_MARGIN_MS,
            "maximum_sample_gap_ms": MAXIMUM_GAP_MS, "order": "forward, reverse, alternating",
            "automatic_retries": 0, "allowed_idle_process_ids": list(args.allowed_idle_pid),
            "allowed_context_max_gate_utilization_percent": IDLE_CONTEXT_UTILIZATION,
            "allowed_context_utilization_gate": "Required when allowed_idle_process_ids is nonempty.",
            "scope": MATCHED_SCOPE if self.matched else "diagnostic",
            "trt_builder_level": args.trt_level,
            "engines": [name for name in self.engines if any(plan.engine == name for plan in self.plans)],
        }

    def diagnostic_command(self, plan, result):
        settle, warm = repr(self.args.settle), repr(self.args.warm)
        if plan.engine in ("ort-cuda", "trt-fp16"):
            graph = self.paths["cuda_onnx" if plan.engine == "ort-cuda" else "trt_onnx"]
            return self.paths["python_executable"], [
                self.paths["python_benchmark"], "--onnx", graph, "--provider", ENGINES[plan.engine].selector,
                "--cache", self.diagnostic["trt_cache"], "--out", result, "--settle", settle, "--warm", warm,
                "--tokens", "16", *self.settings.scheduling_arguments()]
        if plan.engine == "kodamaembed-synthetic":
            return self.paths["cuda_executable"], ["reproducible", "16", settle, warm, result]
        return self.paths["api_executable"], [self.diagnostic["checkpoint"], "e5", "cuda-centered", settle, warm,
                                              result]

    def rotate(self):
        for round_index in range(self.args.rounds):
            order = self.plans if round_index % 2 == 0 else list(reversed(self.plans))
            for plan in order:
                self.run_leg(round_index + 1, plan)

    def run_leg(self, round_number, plan):
        prefix = f"{round_number:02d}-{plan.key.replace('/', '-')}"
        leg = {"round": round_number, "case": plan.case, "key": plan.key, "engine": plan.engine,
               "device": "gpu" if plan.gpu else "cpu", "status": "gating", "valid": False, "gate": [],
               "timing": None, "telemetry": None, "exit_code": None, "process_scheduling": None,
               "load_average_before": None, "load_average_after": None,
               "stdout": f"{prefix}.stdout.log", "stderr": f"{prefix}.stderr.log"}
        if plan.gpu:
            leg.update(monitor_stdout=f"{prefix}-telemetry.stdout.log", monitor_stderr=f"{prefix}-telemetry.stderr.log")
        self.legs.append(leg)
        self.summary["status"] = "running"
        if plan.gpu:
            self.stage = f"{prefix} gate"
            print(f"Gating {prefix}", flush=True)
            for reading in self.gpu.gate(self.stage):
                leg["gate"].append(reading)
                self.save()
        json_result = self.matched or plan.engine in ("ort-cuda", "trt-fp16")
        result = self.output / (prefix + (".json" if json_result else ".csv"))
        leg["result"] = result.name
        if self.matched:
            program, arguments = matched_command(self.table[plan.engine], plan.inputs, self.paths, self.settings,
                                                 str(result), self.fixture_option)
        else:
            program, arguments = self.diagnostic_command(plan, str(result))
        self.stage = f"{prefix} benchmark"
        leg["status"] = "running"
        self.save()
        monitor = benchmark = None
        try:
            if plan.gpu:
                monitor = self.gpu.start_monitor(self.output, f"{prefix}-telemetry")
            leg["load_average_before"] = _system.load_average()
            print(f"Running {prefix}", flush=True)
            benchmark = Tree(program, arguments, directory=REPOSITORY, scheduling=self.requested,
                             log_dir=self.output, log_prefix=prefix)
            leg["process_scheduling"] = benchmark.scheduling
            deadline = time.monotonic() + self.args.max_run_minutes * 60
            while not benchmark.wait(0.25):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Benchmark timed out.")
                if monitor is not None and not monitor.running():
                    raise RuntimeError("GPU telemetry monitor stopped during the benchmark.")
            outcome = benchmark.finish()
            leg["exit_code"] = outcome.exit_code
            leg["load_average_after"] = _system.load_average()
            telemetry = self.gpu.stop_monitor(monitor) if monitor is not None else None
            if outcome.exit_code != 0:
                raise RuntimeError("Benchmark exited unsuccessfully; see retained logs.")
            self.stage = f"{prefix} validation"
            if self.matched:
                self.accept(leg, plan, result)
            else:
                leg["timing"] = read_timing(result, matched=False)
                if plan.engine == "kodamaembed-api":
                    match = FINGERPRINT.search(outcome.stdout)
                    if match is None:
                        raise ValueError("API output did not report its arithmetic fingerprint.")
                    leg["timing"]["metadata"] = {"fingerprint": match.group()}
            if telemetry is not None:
                leg["telemetry"] = check_timing(telemetry, leg["timing"]["timing_start_unix_ms"],
                                                leg["timing"]["timing_end_unix_ms"], self.gpu.limits)
                if not leg["telemetry"]["valid"]:
                    raise RuntimeError("Timing telemetry was rejected; see summary reasons.")
            leg["valid"] = True
            leg["status"] = "accepted"
            self.save()
            print(f"Accepted {prefix}: {self.headline(leg['timing'])}", flush=True)
        finally:
            for tree in (benchmark, monitor):
                if tree is not None and not tree.finished:
                    try:
                        tree.finish(stop=True)
                    except Exception as error:
                        self.summary.setdefault("cleanup_errors", []).append(f"{prefix}: {error}")

    def accept(self, leg, plan, result):
        """Sets a matched leg's timing and artifacts from its receipt; raises unless they are acceptable."""
        leg["timing"], leg["artifacts"] = read_matched_result(result, ENGINES[plan.engine])
        accept_matched(leg["timing"], plan, self.summary["hashes"], self.token_hashes, self.requested)

    @staticmethod
    def headline(timing):
        return f"median {timing['median_ms']:.6f} ms"

    def aggregates(self):
        return aggregate(self.plans, self.legs, self.args.rounds)

    def complete(self):
        self.stage = "completion"
        self.summary["aggregates"] = self.aggregates()
        self.summary["final_hashes"] = assert_input_hashes(self.paths, self.summary["hashes"])
        if self.matched:
            for leg in self.legs:
                assert_artifact_hashes(self.output, leg["artifacts"])
        self.summary["valid"] = True
        self.summary["status"] = "complete"
        self.save()
        print("Complete rotation accepted. Results are in summary.json.", flush=True)

    def fail(self, error):
        self.summary["status"] = "failed"
        if self.legs and not self.legs[-1]["valid"]:
            self.legs[-1]["status"] = "failed"
        self.summary["failure"] = {"stage": self.stage, "error_type": type(error).__qualname__,
                                   "details": "failure.log"}
        (self.output / "failure.log").write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8")
        self.save()
        print(f"Comparison stopped at {self.stage}; the partial summary and logs are in {self.output}.",
              file=sys.stderr, flush=True)


def seconds(text):
    value = float(text)
    if not math.isfinite(value) or not 0 <= value <= 3600:
        raise argparse.ArgumentTypeError("seconds must be within 0-3600")
    return value


def bounded(low, high, kind=int):
    def parse(text):
        value = kind(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be within {low}-{high}")
        return value
    return parse


def shared_options(description, engines):
    """The options a runner of any protocol takes; `engines` is its table."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", required=True, help="a new or empty directory")
    parser.add_argument("--engines", default="gpu",
                        help="comma-separated engines or groups: gpu, cpu, " + ", ".join(engines))
    parser.add_argument("--families", help="run only the cases of these comma-separated families: "
                                           + ", ".join(FAMILIES))
    parser.add_argument("--max-tokens", type=bounded(1, 2**31 - 1),
                        help="run only the cases whose longest text has at most this many tokens")
    parser.add_argument("--python", help="interpreter for the ONNX Runtime engines")
    parser.add_argument("--torch-python", help="interpreter for the torch engines")
    parser.add_argument("--api-exe", help="bench_query_api, by default from target/release/examples")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--cpus", type=parse_cpus,
                        help="logical processors for every benchmark, such as 0-15 (Linux and Windows)")
    parser.add_argument("--priority", choices=PRIORITIES, default="high" if PLATFORM == "windows" else "inherit",
                        help="high on Linux and macOS needs root or CAP_SYS_NICE; the default is high on Windows")
    parser.add_argument("--threads", type=bounded(1, 1024), default=8,
                        help="threads of the Python engines on GPU legs")
    parser.add_argument("--cpu-threads", type=bounded(1, 1024),
                        help="compute and tokenizer threads of every engine on CPU legs")
    parser.add_argument("--allowed-idle-pid", type=bounded(1, 2**31 - 1), action="append", default=[])
    parser.add_argument("--trt-level", type=bounded(0, 5), default=5)
    parser.add_argument("--rounds", type=bounded(3, 100), default=3)
    parser.add_argument("--settle", type=seconds, default=30.0)
    parser.add_argument("--max-temp", type=bounded(1, 100, float), default=86.0)
    parser.add_argument("--min-power-w", type=bounded(1, 2000, float),
                        help="minimum enforced GPU power limit; the GPU's default limit if omitted")
    parser.add_argument("--max-wait-minutes", type=bounded(1, 1440), default=15)
    parser.add_argument("--max-run-minutes", type=bounded(1, 1440), default=15)
    return parser


def arguments(argv=None):
    parser = shared_options(__doc__, ENGINES)
    parser.add_argument("--cases", help="matched cases JSON; without it the diagnostic legs run")
    parser.add_argument("--warm", type=seconds, default=1.0)
    parser.add_argument("--cuda-exe", help="bench_cuda_query for the diagnostic legs")
    parser.add_argument("--cuda-onnx")
    parser.add_argument("--trt-onnx")
    parser.add_argument("--trt-cache")
    parser.add_argument("--checkpoint")
    return parser.parse_args(argv)


def main(argv=None):
    return Comparison(arguments(argv)).run()


if __name__ == "__main__":
    sys.exit(main())

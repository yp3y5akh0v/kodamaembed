"""Checks binding a benchmark receipt's rotations to their own samples, their quality runs and
this checkout's sources; the published release must also pass validate_release()."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from _matched import ENCODERS, FAMILIES, ROLES
from _process import PRIORITIES, fulfills
from _telemetry import (GATE_INTERVAL_SECONDS, GATE_SAMPLES, IDLE_CONTEXT_UTILIZATION, MAXIMUM_GAP_MS,
                        MONITOR_INTERVAL_MS, TIMING_MARGIN_MS, Limits, gate_passes, judge_timing)
from run_quality import BENCHMARK_FILES, DATASETS, FIQA_PASSAGES, IDENTICAL, KODV_HEADER, RUNS, SCORED
from run_query_comparison import (CALLS, CHECKPOINT_FILES, CONTAINMENT, ENGINES, FINGERPRINT, GROUPS, MATCHED_SCOPE,
                                  SAMPLES)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SCOPE = MATCHED_SCOPE + "; warm sequential single-query latency"
CALLER_RETENTION = ("three returned outputs retained until the sample clock stops; "
                    "validation and release outside timing")
# A sample's wall-clock interval may differ from its three timed calls by this much per call.
INTERVAL_TOLERANCE_MS = 0.05
RUST_BATCH_TOKENS = 16384

# The runner's names for repository files; each measured hash must match this checkout.
REPOSITORY_FILES = {
    "runner": "scripts/benchmarks/run_query_comparison.py",
    "sequence_runner": "scripts/benchmarks/run_length_sequence.py",
    "sequence_helper": "scripts/benchmarks/_sequence.py",
    "process_helper": "scripts/benchmarks/_process.py",
    "telemetry_helper": "scripts/benchmarks/_telemetry.py",
    "system_helper": "scripts/benchmarks/_system.py",
    "matched_helper": "scripts/benchmarks/_matched.py",
    "python_benchmark": "scripts/benchmarks/bench_ort_query.py",
    "torch_benchmark": "scripts/benchmarks/bench_torch_query.py",
    "api_example_source": "crates/kodamaembed/examples/bench_query_api.rs",
    "api_matched_source": "crates/kodamaembed/examples/support/matched_query.rs",
    "api_sequence_source": "crates/kodamaembed/examples/support/length_sequence.rs",
    "cargo_lock": "Cargo.lock",
    "rust_toolchain": "rust-toolchain.toml",
    "centered_ptx": "crates/kodamaembed-cuda/ptx/repro.ptx",
}
# Every file whose bytes can change a measurement or its record; .md files are excluded.
SOURCE_FILE = re.compile(
    r"Cargo\.(?:toml|lock)|rust-toolchain\.toml|crates/.+|scripts/benchmarks/(?:"
    r"(?:_matched|_process|_sequence|_system|_telemetry|bench_ort_query|bench_torch_query|export_bert|"
    r"export_receipt|prepare_benchmark_fixtures|prepare_quality_subset|receipts|run_length_sequence|run_quality|"
    r"run_query_comparison|sequence_receipt)\.py|"
    r"requirements(?:-export|-torch)?\.txt|(?:cases|sequence)\.example\.json)")
SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
CASE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,79}", re.ASCII)
ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*", re.ASCII)
# A path of the measuring machine: a drive, a share, a home or any absolute directory.
PRIVATE_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\[^\\]+\\|~[\\/]|(?:^|[\s\"'=(])/[^/\s\"']+/", re.ASCII)
PRIVATE_KEYS = {"process_id", "allowed_idle_process_ids", "gpu_identity", "uuid", "device_uuid"}

# A receipt holds one source inventory and the rotations measured on it, each appended whole.
RECEIPT_FIELDS = {"schema_version", "status", "scope", "source", "rotations"}
ROTATION_FIELDS = {"provenance", "platform", "hardware", "speed_settings", "settings", "input_hashes",
                   "final_input_hashes", "cases", "legs", "background_process_observations", "aggregates", "counts",
                   "excluded_rotations", "quality", "export_policy", "limitations"}
PROVENANCE_FIELDS = {"rotation_created_utc", "rotation_summary_sha256", "quality_manifest_sha256"}
SETTINGS_FIELDS = {
    "rounds", "tokens", "samples_per_run", "calls_per_sample", "settle_seconds", "warm_seconds", "cpus", "priority",
    "gpu_leg_python_threads", "cpu_leg_threads", "execution_speed_throttling", "max_gate_temperature_c",
    "minimum_enforced_power_w", "minimum_power_source", "max_gate_minutes", "max_run_minutes",
    "gate_consecutive_samples", "gate_interval_ms", "telemetry_interval_ms", "timing_margin_ms",
    "maximum_sample_gap_ms", "order", "automatic_retries", "allowed_idle_contexts",
    "allowed_context_max_gate_utilization_percent", "allowed_context_utilization_gate", "scope",
    "trt_builder_level", "engines"}
SPEED_SETTINGS = {"linux": {"frequency_governors", "turbo", "isolated_cpus", "nohz_full_cpus"},
                  "windows": {"power_scheme"}, "macos": {"low_power_mode"}}
SCHEDULING_FIELDS = {"windows": {"platform", "priority_class", "affinity_mask", "execution_speed_control_mask",
                                 "execution_speed_state_mask"},
                     "linux": {"platform", "cpus", "nice"}, "macos": {"platform", "nice"}}
LEG_FIELDS = {"round", "case", "engine", "device", "status", "valid", "exit_code", "samples_ms",
              "sample_intervals_unix_ms", "timing_start_unix_ms", "timing_end_unix_ms", "median_ms", "min_ms", "max_ms",
              "metadata", "process_scheduling", "load_average_before", "load_average_after", "gate", "telemetry",
              "artifacts"}
READING_FIELDS = {"unix_ms", "enforced_power_w", "sm_mhz", "memory_mhz", "temperature_c", "utilization_percent",
                  "software_power_cap", "software_thermal_slowdown"}
TELEMETRY_FIELDS = {"valid", "reasons", "samples", "total_monitor_samples", "ignored_shutdown_fragments",
                    "software_power_cap_samples", "power_cap_policy"}
FRAGMENT_FIELDS = {"line_number", "raw", "fields_present", "unix_ms", "reason"}
COMMON_METADATA = {"caller_retention", "calls_per_sample", "checkpoint_sha256", "engine", "family", "fixture_sha256",
                   "hidden", "input_ids_types_sha256", "label", "lengths", "load_seconds", "max_length",
                   "output_norm_max", "output_norm_min", "real_tokens", "role", "scope", "scope_description",
                   "settle_seconds", "shape", "warm_calls", "warm_seconds"}
RUST_METADATA = {"batch_tokens", "cpu_workers", "device", "fingerprint", "mode", "threads", "token_audit"}
PYTHON_METADATA = {"harness_sha256", "outputs", "outputs_payload_sha256", "outputs_sha256", "packages",
                   "padded_array_sha256", "padded_tokens", "python_executable_sha256", "scheduling",
                   "scheduling_after_timing", "scheduling_before_timing", "threads", "tokenizer_method"}
ORT_METADATA = {"export_lineage", "onnx_sha256", "placement_audit_sha256", "pooling_precision", "provider_options",
                "providers"}
TORCH_METADATA = {"all_tokens_valid", "attention_classes", "attention_dispatch", "attention_profile", "helper_sha256",
                  "ordinary_forward_max_abs_difference", "pooling_precision", "transformer_precision"}
# What a torch leg records of its encoder's own settings.
TORCH_ENCODER_METADATA = {"bert": set(), "modernbert": {"reference_compile"}}
TORCH_DEVICE_METADATA = {"cpu": {"interop_threads", "intra_op_threads"},
                         "gpu": {"allow_fp16_reduced_precision_reduction", "allow_tf32", "capture_seconds",
                                 "cuda_arch_list", "cuda_build", "device_capability", "device_name",
                                 "flash_attention_built", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"}}
TRANSFORMER_PRECISION = {"cpu": "FP32 parameters and activations",
                         "gpu": "FP16 parameters and activations; PyTorch library accumulation rules"}
PACKAGES = {"ort": {"numpy", "onnx", "onnxruntime", "onnxruntime-gpu", "tokenizers", "tensorrt-cu12-libs"},
            "torch": {"numpy", "torch", "transformers", "tokenizers", "safetensors"}}
PROVIDERS = {"cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
             "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
             "cpu": ["CPUExecutionProvider"]}
# Provider options as ONNX Runtime reports them for the configurations the engines request.
PROVIDER_SETTINGS = {"CUDAExecutionProvider": {"device_id": "0", "use_tf32": "0"},
                     "TensorrtExecutionProvider": {"device_id": "0", "trt_fp16_enable": "1", "trt_cuda_graph_enable": "1",
                                                   "trt_engine_cache_enable": "1", "trt_timing_cache_enable": "1"}}
PYTHON_SCHEDULING = ("scheduling", "scheduling_before_timing", "scheduling_after_timing")

QUALITY_POLICY = "saved-retrieval-quality-v1"
METRIC_CONVENTION = "binary-distinct-relevance-v1"
QUALITY_LIMITS = {"mean_passage_cosine_distance": ("<=", 1e-3), "mean_query_cosine_distance": ("<=", 1e-3),
                  "max_l2": ("<=", 0.1), "max_unit_norm_error": ("<=", 1e-5), "non_finite": ("<=", 0.0),
                  "mean_top10_overlap": (">=", 0.9), "ndcg_loss": ("<=", 0.005), "recall_loss": ("<=", 0.01),
                  "hit_loss": ("<=", 0.01)}
FIDELITY_GATES = ("mean_passage_cosine_distance", "mean_query_cosine_distance", "max_l2", "max_unit_norm_error",
                  "non_finite")
RETRIEVAL_GATES = ("mean_top10_overlap", "ndcg_loss", "recall_loss", "hit_loss")
METRIC_FIELDS = {"queries", "ndcg_at_10", "recall_at_10", "hit_at_10"}
VECTOR_METRIC_FIELDS = {"rows", "min_cosine", "mean_cosine", "max_l2", "max_abs", "max_norm_error", "non_finite"}
TOTALS_FIELDS = {"queries", "identical_top10_sets", "mean_top10_overlap", "mean_passage_cosine_distance",
                 "mean_query_cosine_distance", "max_unit_norm_error", "min_cosine", "max_l2", "non_finite",
                 "reference", "candidate", "quality_gates"}
TASK_FIELDS = {"task", "identical_top10_sets", "mean_top10_overlap", "max_unit_norm_error", "queries", "passages",
               "query_vectors", "reference", "candidate", "quality_gates"}
QUALITY_FIELDS = {"validator_sha256", "started_utc", "finished_utc", "models", "datasets", "cases",
                  "individual_failed_tasks"}

RELEASE_CASES = {f"{name}-b1-l{length}": (name, length) for name, family in FAMILIES.items()
                 for length in family.lengths}
RELEASE_CPU_ENGINES = ("ort-cpu", "torch-cpu", "kodamaembed-cpu-fp32", "kodamaembed-cpu-centered")
# The release's CPU group ends at this length: a longer case takes a processor over ten seconds a call.
RELEASE_CPU_TOKENS = 2048


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), f"{label}: unexpected or missing fields")


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: expected a finite number")
    return value


def count(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: expected an integer of at least {minimum}")
    return value


def text(value, label):
    require(isinstance(value, str) and value.strip() != "", f"{label}: expected text")
    return value


def digest(value, label):
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None, f"{label}: expected a SHA-256 digest")
    return value


def file_sha256(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def privacy(value, label="receipt"):
    """No process identifier, device identifier or machine path anywhere."""
    if isinstance(value, dict):
        for key, item in value.items():
            require(key not in PRIVATE_KEYS, f"{label}: local identifier {key}")
            privacy(key, label)
            privacy(item, label)
    elif isinstance(value, list):
        for item in value:
            privacy(item, label)
    elif isinstance(value, str):
        require(PRIVATE_PATH.search(value) is None, f"{label}: machine path in {value!r}")


def source_files(root):
    """Every source file of `root` that SOURCE_FILE binds, with its digest, in path order."""
    paths = [path for path in (root / "crates").rglob("*") if path.is_file()]
    paths += [root / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
    paths += [path for path in (root / "scripts" / "benchmarks").iterdir() if path.is_file()]
    files = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if SOURCE_FILE.fullmatch(relative) and path.suffix != ".md":
            files.append({"file": relative, "sha256": file_sha256(path)})
    return sorted(files, key=lambda item: item["file"])


def inventory_digest(files):
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def engines_of(rotation, table=ENGINES):
    return [table[name] for name in rotation["settings"]["engines"]]


def expected_inputs(rotation, table=ENGINES):
    """The input names a rotation of these cases and engines hashes."""
    engines = engines_of(rotation, table)
    kinds = {engine.kind for engine in engines}
    names = {"runner", "process_helper", "telemetry_helper", "system_helper", "matched_helper", "cargo_lock",
             "rust_toolchain", "cases"}
    if "ort" in kinds:
        names |= {"python_executable", "python_benchmark"}
    if "torch" in kinds:
        names |= {"torch_executable", "torch_benchmark"}
    if "rust" in kinds:
        names |= {"api_executable", "api_example_source", "api_matched_source", "centered_ptx"}
    for case in rotation["cases"]:
        name = case["name"]
        names |= {f"{name}/fixture", *(f"{name}/{file}" for file in CHECKPOINT_FILES)}
        names |= {f"{name}/{engine.graph}" for engine in engines if engine.graph}
        if "ort" in kinds:
            names.add(f"{name}/export_receipt")
    return names


def gpu_limits(settings):
    return Limits(settings["minimum_enforced_power_w"], settings["max_gate_temperature_c"],
                  settings["allowed_idle_contexts"] > 0)


def check_inputs(rotation, expected, sources):
    """A rotation's input inventory: the `expected` names, each a digest, unchanged over the rotation,
    and every repository file among them as `sources`, this checkout's digests, hold it."""
    hashes = rotation["input_hashes"]
    require(isinstance(hashes, dict) and set(hashes) == expected, "input inventory differs")
    for name, value in hashes.items():
        digest(value, f"input {name}")
    require(rotation["final_input_hashes"] == hashes, "inputs changed during the rotation")
    for name, relative in REPOSITORY_FILES.items():
        if name in hashes:
            require(sources.get(relative) == hashes[name], f"measured {name} differs from this checkout")


def check_settings(rotation):
    settings = rotation["settings"]
    fields(settings, SETTINGS_FIELDS, "settings")
    check_shared_settings(rotation, ENGINES)
    require(settings["tokens"] is None and settings["samples_per_run"] == SAMPLES
            and settings["calls_per_sample"] == CALLS, "settings: sampling protocol differs")
    require(number(settings["warm_seconds"], "warm") >= 0, "settings: negative warm-up")
    require(settings["scope"] == MATCHED_SCOPE, "settings: scope differs")
    return settings


def check_shared_settings(rotation, table):
    """The settings a runner of any protocol records: engines of `table`, rounds, scheduling,
    gates and limits."""
    settings = rotation["settings"]
    engines = settings["engines"]
    require(isinstance(engines, list) and engines and len(set(engines)) == len(engines)
            and all(name in table for name in engines), "settings: invalid engine list")
    devices = {table[name].device for name in engines}
    count(settings["rounds"], "rounds", 3)
    require(number(settings["settle_seconds"], "settle") >= 0, "settings: negative settle")
    cpus = settings["cpus"]
    require(cpus is None or (isinstance(cpus, list) and cpus and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
                             and cpus == sorted(set(cpus))), "settings: invalid processor list")
    require(settings["priority"] in PRIORITIES, "settings: invalid priority")
    count(settings["gpu_leg_python_threads"], "Python threads", 1)
    require(settings["cpu_leg_threads"] is None if "cpu" not in devices
            else count(settings["cpu_leg_threads"], "CPU threads", 1), "settings: CPU threads differ from the legs")
    system = rotation["platform"]["system"]
    require(settings["execution_speed_throttling"] == ("disabled" if system == "windows" else None),
            "settings: execution speed throttling differs from the platform")
    if "gpu" in devices:
        number(settings["max_gate_temperature_c"], "maximum gate temperature")
        number(settings["minimum_enforced_power_w"], "minimum enforced power")
        require(settings["minimum_power_source"] in ("explicit", "gpu-default"), "settings: invalid power source")
    else:
        require(settings["max_gate_temperature_c"] is None and settings["minimum_enforced_power_w"] is None
                and settings["minimum_power_source"] is None, "settings: GPU gate without a GPU leg")
    require(count(settings["max_gate_minutes"], "gate minutes", 1) and count(settings["max_run_minutes"], "run minutes", 1),
            "settings: invalid time limits")
    require(settings["gate_consecutive_samples"] == GATE_SAMPLES
            and settings["gate_interval_ms"] == int(GATE_INTERVAL_SECONDS * 1000)
            and settings["telemetry_interval_ms"] == MONITOR_INTERVAL_MS
            and settings["timing_margin_ms"] == TIMING_MARGIN_MS
            and settings["maximum_sample_gap_ms"] == MAXIMUM_GAP_MS, "settings: gate or telemetry protocol differs")
    require(settings["order"] == "forward, reverse, alternating" and settings["automatic_retries"] == 0,
            "settings: rotation protocol differs")
    count(settings["allowed_idle_contexts"], "allowed idle contexts")
    require(settings["allowed_context_max_gate_utilization_percent"] == IDLE_CONTEXT_UTILIZATION
            and isinstance(settings["allowed_context_utilization_gate"], str), "settings: idle-context rule differs")
    require(type(settings["trt_builder_level"]) is int and 0 <= settings["trt_builder_level"] <= 5,
            "settings: invalid TensorRT builder level")


def check_machine(rotation, table=ENGINES):
    platform_record, hardware = rotation["platform"], rotation["hardware"]
    fields(platform_record, {"system", "process_containment"}, "platform")
    system = platform_record["system"]
    require(system in SCHEDULING_FIELDS and platform_record["process_containment"] == CONTAINMENT[system],
            "platform differs from the runner's containment")
    fields(hardware, {"gpu", "cpu_names", "os", "python"}, "hardware")
    require(isinstance(hardware["cpu_names"], list) and hardware["cpu_names"]
            and all(isinstance(name, str) for name in hardware["cpu_names"]), "hardware: invalid processor names")
    text(hardware["os"], "operating system")
    text(hardware["python"], "Python version")
    if any(engine.device == "gpu" for engine in engines_of(rotation, table)):
        fields(hardware["gpu"], {"name", "driver_version", "default_power_limit_w"}, "GPU")
        text(hardware["gpu"]["name"], "GPU name")
        text(hardware["gpu"]["driver_version"], "GPU driver")
        limit = hardware["gpu"]["default_power_limit_w"]
        require(limit is None or number(limit, "default power limit") > 0, "GPU: invalid default power limit")
    else:
        require(hardware["gpu"] is None, "hardware: GPU recorded without a GPU leg")
    fields(rotation["speed_settings"], SPEED_SETTINGS[system], "speed settings")


def check_cases(cases, extra=frozenset()):
    """Each case by name; `extra` names the fields a protocol adds to a case."""
    require(isinstance(cases, list) and cases, "missing cases")
    names = set()
    for case in cases:
        fields(case, {"name", "family", "role", "sequences", "lengths"} | extra, "case")
        require(isinstance(case["name"], str) and CASE_NAME.fullmatch(case["name"]) and case["name"] not in names,
                "case names must be unique lowercase letters, digits and hyphens")
        names.add(case["name"])
        require(case["family"] in FAMILIES and case["role"] in ROLES, "case: invalid family or role")
        lengths = case["lengths"]
        require(isinstance(lengths, list) and lengths and all(type(length) is int and 1 <= length for length in lengths)
                and case["sequences"] == len(lengths), "case: invalid geometry")
    return {case["name"]: case for case in cases}


def check_scheduling(state, rotation, label):
    system = rotation["platform"]["system"]
    fields(state, SCHEDULING_FIELDS[system], label)
    settings = rotation["settings"]
    require(state["platform"] == system and fulfills(state, settings["cpus"], settings["priority"]),
            f"{label} differs from the requested scheduling")


def check_samples(leg, label):
    samples, intervals = leg["samples_ms"], leg["sample_intervals_unix_ms"]
    require(isinstance(samples, list) and isinstance(intervals, list) and len(samples) == len(intervals) == SAMPLES,
            f"{label}: missing samples or intervals")
    start = number(leg["timing_start_unix_ms"], "timing start")
    end = number(leg["timing_end_unix_ms"], "timing end")
    previous = start
    for sample, interval in zip(samples, intervals):
        require(number(sample, "sample") > 0, f"{label}: nonpositive latency")
        require(isinstance(interval, list) and len(interval) == 2, f"{label}: invalid sample interval")
        left, right = (number(value, "interval") for value in interval)
        require(previous <= left < right <= end, f"{label}: overlapping or out-of-bounds interval")
        require(abs((right - left) / CALLS - sample) <= INTERVAL_TOLERANCE_MS,
                f"{label}: a sample does not fit its wall-clock interval")
        previous = right
    for name, value in (("median_ms", statistics.median(samples)), ("min_ms", min(samples)), ("max_ms", max(samples))):
        require(leg[name] == value, f"{label}: {name} differs from its samples")


def engine_metadata(engine, encoder):
    """The fields beyond the common ones that a leg of `engine` records for a checkpoint of `encoder`."""
    if engine.kind == "rust":
        return RUST_METADATA
    if engine.kind == "ort":
        return PYTHON_METADATA | ORT_METADATA | ({"trt_cache_sha256"} if engine.selector == "tensorrt" else set())
    return PYTHON_METADATA | TORCH_METADATA | TORCH_DEVICE_METADATA[engine.device] | TORCH_ENCODER_METADATA[encoder]


def check_metadata(leg, rotation, case, shared, label):
    engine, family = ENGINES[leg["engine"]], FAMILIES[case["family"]]
    metadata, settings, hashes = leg["metadata"], rotation["settings"], rotation["input_hashes"]
    fields(metadata, COMMON_METADATA | engine_metadata(engine, family.encoder), f"{label} metadata")
    require(metadata["engine"] == engine.receipt, f"{label}: engine label differs")
    require(metadata["scope"] == "end-to-end" and metadata["calls_per_sample"] == CALLS
            and metadata["caller_retention"] == CALLER_RETENTION, f"{label}: timed boundary differs")
    require(metadata["family"] == case["family"] and metadata["role"] == case["role"]
            and metadata["label"] == case["name"]
            and metadata["lengths"] == case["lengths"] and metadata["real_tokens"] == sum(case["lengths"])
            and metadata["shape"] == [case["sequences"], max(case["lengths"])], f"{label}: timed geometry differs")
    require(metadata["hidden"] == family.hidden and metadata["max_length"] == family.max_length,
            f"{label}: model geometry differs")
    require(metadata["settle_seconds"] == settings["settle_seconds"]
            and metadata["warm_seconds"] == settings["warm_seconds"], f"{label}: warm-up differs")
    require(count(metadata["warm_calls"], "warm calls", 20) and number(metadata["load_seconds"], "load") >= 0,
            f"{label}: invalid warm-up or load time")
    text(metadata["scope_description"], "scope description")
    check_lineage(metadata, hashes, case, shared, label)
    if engine.kind == "rust":
        check_rust_metadata(metadata, engine, settings, label)
    else:
        check_python_metadata(leg, rotation, case, engine, shared, label)


def check_lineage(metadata, hashes, case, shared, label):
    """A leg's fixture and checkpoint against the run's inputs, its tokens against the case's
    other legs, and its outputs' norms."""
    require(metadata["fixture_sha256"] == hashes[f"{case['name']}/fixture"], f"{label}: fixture lineage differs")
    fields(metadata["checkpoint_sha256"], CHECKPOINT_FILES, f"{label} checkpoint")
    require(all(metadata["checkpoint_sha256"][name] == hashes[f"{case['name']}/{name}"] for name in CHECKPOINT_FILES),
            f"{label}: checkpoint lineage differs")
    tokens = digest(metadata["input_ids_types_sha256"], "token digest")
    require(shared.setdefault(("tokens", case["name"]), tokens) == tokens, f"{label}: engines tokenized different inputs")
    require(0.999 <= number(metadata["output_norm_min"], "norm") <= number(metadata["output_norm_max"], "norm") <= 1.001,
            f"{label}: outputs are not normalized")


def check_rust_metadata(metadata, engine, settings, label):
    threads = settings["cpu_leg_threads"] if engine.device == "cpu" else None
    workers = {"inference": threads, "tokenizer": threads} if threads is not None else None
    require(metadata["mode"] == engine.selector, f"{label}: KodamaEmbed mode differs")
    require(metadata["threads"] == threads and metadata["cpu_workers"] == workers, f"{label}: CPU workers differ")
    require(metadata["batch_tokens"] == RUST_BATCH_TOKENS, f"{label}: batch tokens differ")
    require(isinstance(metadata["fingerprint"], str) and FINGERPRINT.fullmatch(metadata["fingerprint"]),
            f"{label}: invalid fingerprint")
    text(metadata["device"], "KodamaEmbed device")
    text(metadata["token_audit"], "token audit")


def check_python_metadata(leg, rotation, case, engine, shared, label):
    metadata = leg["metadata"]
    threads = check_python_run(leg, rotation, engine, shared, "matched_helper", label)
    fields(metadata["padded_array_sha256"], {"input_ids", "token_type_ids", "attention_mask"}, f"{label} padded arrays")
    for value in metadata["padded_array_sha256"].values():
        digest(value, "padded array")
    count(metadata["padded_tokens"], "padded tokens", 1)
    require(metadata["tokenizer_method"] in ("encode_batch_fast", "encode_batch"), f"{label}: unknown tokenizer method")
    if engine.kind == "ort":
        check_ort_metadata(leg, rotation, case, engine, label)
    else:
        check_torch_metadata(metadata, engine, case, threads, rotation["input_hashes"], label)


def check_python_run(leg, rotation, engine, shared, ort_harness, label):
    """A Python engine's threads, interpreter, source, packages, scheduling and outputs; returns
    its threads. `ort_harness` names the input that performs the ONNX Runtime legs."""
    metadata, settings, hashes = leg["metadata"], rotation["settings"], rotation["input_hashes"]
    ort = engine.kind == "ort"
    threads = settings["cpu_leg_threads"] if engine.device == "cpu" else settings["gpu_leg_python_threads"]
    require(metadata["threads"] == threads, f"{label}: threads differ")
    require(metadata["python_executable_sha256"] == hashes["python_executable" if ort else "torch_executable"],
            f"{label}: Python lineage differs")
    require(metadata["harness_sha256"] == hashes[ort_harness if ort else "torch_benchmark"],
            f"{label}: benchmark source differs")
    fields(metadata["packages"], PACKAGES[engine.kind], f"{label} packages")
    require(shared.setdefault(("packages", engine.kind), metadata["packages"]) == metadata["packages"],
            f"{label}: package versions changed")
    states = [metadata[name] for name in PYTHON_SCHEDULING]
    for name, state in zip(PYTHON_SCHEDULING, states):
        check_scheduling(state, rotation, f"{label} {name}")
    require(states[0] == states[1] == states[2], f"{label}: scheduling changed during the benchmark")
    require(leg["artifacts"].get(metadata["outputs"]) == digest(metadata["outputs_sha256"], "outputs"),
            f"{label}: output artifact differs")
    digest(metadata["outputs_payload_sha256"], "output payload")
    text(metadata["pooling_precision"], "pooling precision")
    return threads


def check_ort_metadata(leg, rotation, case, engine, label):
    options = leg["metadata"]["provider_options"]
    # The engines replay a CUDA graph and profile the fixture's padded shape.
    shape = ",".join(f"{input_name}:{case['sequences']}x{max(case['lengths'])}"
                     for input_name in ENCODERS[FAMILIES[case["family"]].encoder].inputs)
    check_ort_run(leg, rotation, case, engine, PROVIDER_SETTINGS, dict.fromkeys(("min", "opt", "max"), shape), label)
    if engine.selector == "cuda":
        require(options["CUDAExecutionProvider"].get("enable_cuda_graph") == "1", f"{label}: CUDA graph not enabled")


def check_ort_run(leg, rotation, case, engine, provider_settings, profile, label):
    """An ONNX Runtime leg's providers with the options of `provider_settings`, its TensorRT
    build with `profile`'s shapes, and its graph, export and placement lineage."""
    metadata, hashes, name = leg["metadata"], rotation["input_hashes"], case["name"]
    require(metadata["providers"] == PROVIDERS[engine.selector], f"{label}: execution providers differ")
    options = metadata["provider_options"]
    require(isinstance(options, dict) and set(options) == set(metadata["providers"])
            and all(isinstance(values, dict) and all(isinstance(value, str) for value in values.values())
                    for values in options.values()), f"{label}: invalid provider options")
    for provider, expected in provider_settings.items():
        if provider in options:
            require(all(options[provider].get(key) == value for key, value in expected.items()),
                    f"{label}: {provider} configuration differs")
    if engine.selector == "tensorrt":
        trt = options["TensorrtExecutionProvider"]
        require(trt.get("trt_builder_optimization_level") == str(rotation["settings"]["trt_builder_level"])
                and all(trt.get(f"trt_profile_{kind}_shapes") == shape for kind, shape in profile.items()),
                f"{label}: TensorRT build differs")
        require(not {"trt_engine_cache_path", "trt_timing_cache_path"} & set(trt), f"{label}: local cache path")
        cache = metadata["trt_cache_sha256"]
        require(isinstance(cache, dict) and cache and all(
            isinstance(file, str) and ARTIFACT_NAME.fullmatch(file) and digest(value, f"{label} TensorRT cache")
            for file, value in cache.items()), f"{label}: invalid TensorRT cache hashes")
    require(isinstance(metadata["onnx_sha256"], dict)
            and list(metadata["onnx_sha256"].values()) == [hashes[f"{name}/{engine.graph}"]],
            f"{label}: graph differs from its input")
    fields(metadata["export_lineage"], {"receipt", "receipt_sha256", "checkpoint_and_graph_hashes_match"},
           f"{label} export lineage")
    require(metadata["export_lineage"]["checkpoint_and_graph_hashes_match"] is True
            and metadata["export_lineage"]["receipt_sha256"] == hashes[f"{name}/export_receipt"],
            f"{label}: export lineage differs")
    placements = [value for name, value in leg["artifacts"].items() if name.endswith(".placement.json")]
    require(placements == [metadata["placement_audit_sha256"]], f"{label}: placement artifact differs")


def check_torch_encoder(metadata, case, label):
    """A torch leg's attention class and its encoder's own settings."""
    encoder = FAMILIES[case["family"]].encoder
    require(metadata["attention_classes"] == [ENCODERS[encoder].attention],
            f"{label}: attention implementation differs")
    require(all(type(metadata[name]) is bool for name in TORCH_ENCODER_METADATA[encoder]),
            f"{label}: invalid encoder setting")


def check_torch_metadata(metadata, engine, case, threads, hashes, label):
    require(metadata["helper_sha256"] == hashes["matched_helper"], f"{label}: helper source differs")
    check_torch_encoder(metadata, case, label)
    require(metadata["attention_profile"] is None, f"{label}: attention implementation differs")
    require(type(metadata["all_tokens_valid"]) is bool, f"{label}: invalid mask specialization")
    require(number(metadata["ordinary_forward_max_abs_difference"], "forward difference") >= 0,
            f"{label}: invalid forward difference")
    text(metadata["attention_dispatch"], "attention dispatch")
    require(metadata["transformer_precision"] == TRANSFORMER_PRECISION[engine.device], f"{label}: precision differs")
    if engine.device == "cpu":
        require(metadata["intra_op_threads"] == threads and metadata["interop_threads"] == 1,
                f"{label}: torch threads differ")
    else:
        require(metadata["allow_tf32"] is False and number(metadata["capture_seconds"], "capture") >= 0,
                f"{label}: CUDA graph settings differ")


def check_reading(reading, label):
    fields(reading, READING_FIELDS, label)
    for name in READING_FIELDS - {"software_power_cap", "software_thermal_slowdown"}:
        number(reading[name], f"{label} {name}")
    require(type(reading["software_power_cap"]) is bool and type(reading["software_thermal_slowdown"]) is bool,
            f"{label}: invalid throttle flag")


def check_gate(gate, limits, start, label):
    """The readings the runner waited through: the gate opens on its first run of
    GATE_SAMPLES passing readings, spaced by the gate interval, before timing."""
    require(isinstance(gate, list) and gate, f"{label}: missing starting gate")
    passing = 0
    for reading in gate:
        check_reading(reading, f"{label} gate")
        require(passing < GATE_SAMPLES, f"{label}: gate continued after it opened")
        passing = passing + 1 if gate_passes(reading, limits) else 0
    require(passing == GATE_SAMPLES, f"{label}: incomplete starting gate")
    stamps = [reading["unix_ms"] for reading in gate]
    require(all(right - left >= GATE_INTERVAL_SECONDS * 1000 for left, right in zip(stamps, stamps[1:])),
            f"{label}: gate readings closer than the gate interval")
    require(stamps[-1] < start, f"{label}: gate did not finish before timing")


def check_telemetry(telemetry, limits, start, end, label):
    """The telemetry verdict must be the one its own samples give."""
    fields(telemetry, TELEMETRY_FIELDS, f"{label} telemetry")
    require(isinstance(telemetry["samples"], list) and isinstance(telemetry["ignored_shutdown_fragments"], list),
            f"{label}: invalid telemetry")
    for reading in telemetry["samples"]:
        check_reading(reading, f"{label} telemetry")
    for fragment in telemetry["ignored_shutdown_fragments"]:
        fields(fragment, FRAGMENT_FIELDS, f"{label} telemetry fragment")
    verdict = judge_timing(telemetry["samples"], telemetry["ignored_shutdown_fragments"], start, end, limits)
    recorded = {name: value for name, value in telemetry.items() if name != "total_monitor_samples"}
    require(telemetry["valid"] is True and recorded == {name: value for name, value in verdict.items()
                                                       if name != "total_monitor_samples"},
            f"{label}: telemetry differs from its samples")
    require(count(telemetry["total_monitor_samples"], "monitor samples") >= len(telemetry["samples"]),
            f"{label}: fewer monitor samples than published")


def check_artifacts(artifacts, label):
    require(isinstance(artifacts, dict) and len(artifacts) >= 2, f"{label}: missing artifact hashes")
    for name, value in artifacts.items():
        require(isinstance(name, str) and ARTIFACT_NAME.fullmatch(name) is not None, f"{label}: artifact is not a basename")
        digest(value, f"{label} artifact")


def check_legs(rotation, cases, table=ENGINES, leg_fields=LEG_FIELDS, samples=check_samples, metadata=check_metadata):
    """Every leg of a complete alternating rotation, in order; a protocol gives its engine
    `table`, its `leg_fields` and the checks of a leg's `samples` and `metadata`."""
    settings = rotation["settings"]
    legs = rotation["legs"]
    plans = [(case["name"], engine) for case in rotation["cases"] for engine in settings["engines"]
             if case["family"] in table[engine].families]
    order = [(number, case, engine) for number in range(1, settings["rounds"] + 1)
             for case, engine in (plans if number % 2 else plans[::-1])]
    require(isinstance(legs, list) and [(leg.get("round"), leg.get("case"), leg.get("engine")) for leg in legs] == order,
            "legs differ from a complete alternating rotation")
    limits = gpu_limits(settings) if settings["minimum_enforced_power_w"] is not None else None
    shared = {}
    for leg in legs:
        label = f"round {leg['round']} {leg['case']} {leg['engine']}"
        fields(leg, leg_fields, label)
        engine = table[leg["engine"]]
        require(leg["device"] == engine.device and leg["status"] == "accepted" and leg["valid"] is True
                and leg["exit_code"] == 0, f"{label}: not an accepted leg")
        samples(leg, label)
        check_artifacts(leg["artifacts"], label)
        metadata(leg, rotation, cases[leg["case"]], shared, label)
        check_scheduling(leg["process_scheduling"], rotation, f"{label} process scheduling")
        for name in ("load_average_before", "load_average_after"):
            require(leg[name] is None or number(leg[name], name) >= 0, f"{label}: invalid {name}")
        if engine.device == "gpu":
            check_gate(leg["gate"], limits, leg["timing_start_unix_ms"], label)
            check_telemetry(leg["telemetry"], limits, leg["timing_start_unix_ms"], leg["timing_end_unix_ms"], label)
        else:
            require(leg["gate"] == [] and leg["telemetry"] is None, f"{label}: GPU evidence on a CPU leg")


def check_observations(rotation):
    observations = rotation["background_process_observations"]
    gpu_legs = sum(leg["device"] == "gpu" for leg in rotation["legs"])
    require(isinstance(observations, list) and len(observations) == 2 * gpu_legs,
            "expected two GPU process checks per GPU leg")
    allowed = rotation["settings"]["allowed_idle_contexts"]
    for observation in observations:
        fields(observation, {"checked_utc", "stage", "allowed_idle_contexts"}, "GPU process check")
        text(observation["checked_utc"], "check time")
        text(observation["stage"], "check stage")
        require(count(observation["allowed_idle_contexts"], "idle contexts") <= allowed,
                "more idle GPU contexts than allowed")


def aggregate(rotation):
    """Per case and device: every engine's run medians and the speedup of each
    KodamaEmbed mode over the other engines there, recomputed from the samples."""
    rows = []
    for case in rotation["cases"]:
        for device in ("gpu", "cpu"):
            engines = [name for name in rotation["settings"]["engines"]
                       if ENGINES[name].device == device and case["family"] in ENGINES[name].families]
            if not engines:
                continue
            stats = {}
            for engine in engines:
                runs = sorted((leg for leg in rotation["legs"] if leg["case"] == case["name"] and leg["engine"] == engine),
                              key=lambda leg: leg["round"])
                medians = [statistics.median(leg["samples_ms"]) for leg in runs]
                stats[engine] = {"run_medians_ms": medians, "median_of_medians_ms": statistics.median(medians),
                                 "minimum_run_median_ms": min(medians), "maximum_run_median_ms": max(medians)}
            competitors = [name for name in engines if ENGINES[name].kind != "rust"]
            fastest = min(competitors, key=lambda name: stats[name]["median_of_medians_ms"]) if competitors else None
            speedups = {}
            for engine in engines:
                if ENGINES[engine].kind == "rust" and fastest is not None:
                    own = stats[engine]["median_of_medians_ms"]
                    speedups[engine] = {
                        "vs_fastest_competitor": stats[fastest]["median_of_medians_ms"] / own,
                        "pairwise": {name: stats[name]["median_of_medians_ms"] / own for name in competitors}}
            rows.append({"case": case["name"], "family": case["family"], "lengths": case["lengths"], "device": device,
                         "engines": stats, "fastest_competitor": fastest,
                         "fastest_competitor_ms": stats[fastest]["median_of_medians_ms"] if fastest else None,
                         "speedups": speedups})
    return rows


def counts(rotation, rows):
    legs = rotation["legs"]
    samples = sum(len(leg["samples_ms"]) for leg in legs)
    modes = {}
    for row in rows:
        for engine, speedup in row["speedups"].items():
            entry = modes.setdefault(engine, {"cases": 0, "fastest_cases": 0, "at_least_2x_cases": 0,
                                              "pairwise_wins": 0, "pairwise_comparisons": 0})
            entry["cases"] += 1
            entry["fastest_cases"] += int(speedup["vs_fastest_competitor"] > 1)
            entry["at_least_2x_cases"] += int(speedup["vs_fastest_competitor"] >= 2)
            entry["pairwise_wins"] += sum(int(ratio > 1) for ratio in speedup["pairwise"].values())
            entry["pairwise_comparisons"] += len(speedup["pairwise"])
    return {"cases": len(rotation["cases"]), "accepted_legs": len(legs), "samples": samples,
            "timed_calls": samples * CALLS, "kodamaembed": modes}


def check_excluded(excluded, statistic="median_ms"):
    """The disclosed stopped rotations; `statistic` is what a protocol keeps of an accepted leg."""
    require(isinstance(excluded, list), "invalid excluded rotations")
    for rotation in excluded:
        fields(rotation, {"rotation_summary_sha256", "recorded_status", "recorded_valid", "included_in_ranking",
                          "accepted_legs", "stopped_at", "reason", "legs"}, "excluded rotation")
        digest(rotation["rotation_summary_sha256"], "excluded rotation summary")
        require(rotation["recorded_status"] in ("preparing", "running", "failed") and rotation["recorded_valid"] is False
                and rotation["included_in_ranking"] is False, "an excluded rotation must have stopped")
        fields(rotation["stopped_at"], {"stage", "round", "case", "engine"}, "stopping point")
        text(rotation["stopped_at"]["stage"], "stopping stage")
        text(rotation["reason"], "stopping reason")
        for leg in rotation["legs"]:
            fields(leg, {"round", "case", "engine", "status"} | ({statistic} if leg.get("status") == "accepted"
                                                                   else set()), "excluded leg")
        require(rotation["accepted_legs"] == sum(leg["status"] == "accepted" for leg in rotation["legs"]),
                "excluded accepted-leg count differs")


def gate_values(row, totals):
    """The values the validator's quality gates hold for one report row."""
    values = {f"{name}_loss": row["reference"][f"{name}_at_10"] - row["candidate"][f"{name}_at_10"]
              for name in ("ndcg", "recall", "hit")}
    values.update(max_unit_norm_error=row["max_unit_norm_error"], mean_top10_overlap=row["mean_top10_overlap"])
    if totals:
        values.update({name: row[name] for name in
                       ("mean_passage_cosine_distance", "mean_query_cosine_distance", "max_l2", "non_finite")})
    else:
        passages, queries = row["passages"], row["query_vectors"]
        values.update(mean_passage_cosine_distance=1.0 - passages["mean_cosine"],
                      mean_query_cosine_distance=1.0 - queries["mean_cosine"],
                      max_l2=max(passages["max_l2"], queries["max_l2"]),
                      non_finite=passages["non_finite"] + queries["non_finite"])
    return values


def check_gates(row, totals, label):
    """Recomputes the validator's gates from the row's metrics; returns whether it passed."""
    gates = row["quality_gates"]
    fields(gates, set(QUALITY_LIMITS) | {"policy", "evaluated", "fidelity_pass", "retrieval_pass", "pass"},
           f"{label} quality gates")
    require(gates["policy"] == QUALITY_POLICY, f"{label}: quality policy differs")
    for side in ("reference", "candidate"):
        fields(row[side], METRIC_FIELDS, f"{label} {side} metrics")
        for name, value in row[side].items():
            number(value, f"{label} {side} {name}")
    if not totals:
        for side in ("passages", "query_vectors"):
            fields(row[side], VECTOR_METRIC_FIELDS, f"{label} {side} fidelity")
            for name, value in row[side].items():
                number(value, f"{label} {side} {name}")
    values = gate_values(row, totals)
    passed = {}
    for name, (relation, threshold) in QUALITY_LIMITS.items():
        gate = gates[name]
        fields(gate, {"value", "relation", "threshold", "pass"}, f"{label} {name} gate")
        require(gate["relation"] == relation and gate["threshold"] == threshold, f"{label}: {name} threshold changed")
        require(number(gate["value"], f"{label} {name}") == values[name], f"{label}: {name} differs from its metrics")
        passed[name] = gate["value"] <= threshold if relation == "<=" else gate["value"] >= threshold
        require(gate["pass"] is passed[name], f"{label}: {name} pass flag contradicts its value")
    evaluated = row["reference"]["queries"] > 0 and row["reference"]["queries"] == row["candidate"]["queries"]
    fidelity = evaluated and all(passed[name] for name in FIDELITY_GATES)
    retrieval = evaluated and all(passed[name] for name in RETRIEVAL_GATES)
    require(gates["evaluated"] is evaluated and gates["fidelity_pass"] is fidelity
            and gates["retrieval_pass"] is retrieval and gates["pass"] is (fidelity and retrieval),
            f"{label}: quality result contradicts its gates")
    return gates["pass"]


def check_quality_case(case, quality, label):
    fields(case, {"family", "dataset", "engines", "vectors", "report"}, label)
    family, dataset = case["family"], case["dataset"]
    require(family in quality["models"] and dataset in quality["datasets"], f"{label}: unknown model or dataset")
    scored = {"reference", SCORED[family]}
    require(set(case["engines"]) in (scored, scored | set(IDENTICAL[family])), f"{label}: invalid runs")
    for name, run in case["engines"].items():
        fields(run, {"engine", "device", "fingerprint", "seconds"}, f"{label} {name}")
        require(run["engine"] == RUNS[name] and isinstance(run["fingerprint"], str)
                and FINGERPRINT.fullmatch(run["fingerprint"]) and number(run["seconds"], "seconds") >= 0,
                f"{label}: invalid {name} run")
        text(run["device"], f"{name} device")
    report = case["report"]
    fields(report, {"metric_convention", "totals", "tasks"}, f"{label} report")
    require(report["metric_convention"] == METRIC_CONVENTION, f"{label}: metric convention differs")
    fields(report["totals"], TOTALS_FIELDS, f"{label} totals")
    totals_pass = check_gates(report["totals"], True, f"{label} totals")
    tasks = report["tasks"]
    require(isinstance(tasks, list) and tasks and len({task.get("task") for task in tasks}) == len(tasks),
            f"{label}: missing or duplicate tasks")
    failures, rows = [], {}
    for task in tasks:
        fields(task, TASK_FIELDS, f"{label} task")
        name = text(task["task"], "task")
        if not check_gates(task, False, f"{label} {name}"):
            failures.append({"family": family, "dataset": dataset, "task": name, "quality_gates": task["quality_gates"]})
        rows[f"{name}.passages.kodv"] = task["passages"]["rows"]
        rows[f"{name}.queries.kodv"] = task["query_vectors"]["rows"]
    digests = {f"{name}_sha256" for name in case["engines"]}
    require(isinstance(case["vectors"], list) and sorted(vector["name"] for vector in case["vectors"]) == sorted(rows),
            f"{label}: vector files differ from the report's tasks")
    for vector in case["vectors"]:
        fields(vector, {"name", "rows", "dim", "bytes"} | digests, f"{label} vector")
        require(vector["rows"] == rows[vector["name"]] and vector["dim"] == FAMILIES[family].hidden
                and vector["bytes"] == KODV_HEADER.size + 4 * vector["rows"] * vector["dim"],
                f"{label}: vector geometry differs")
        for name in digests:
            digest(vector[name], f"{label} {name}")
    return totals_pass, failures


def check_quality(quality):
    fields(quality, QUALITY_FIELDS, "quality")
    digest(quality["validator_sha256"], "validator")
    text(quality["started_utc"], "quality start")
    text(quality["finished_utc"], "quality finish")
    require(isinstance(quality["models"], dict) and quality["models"], "quality: missing models")
    for family, files in quality["models"].items():
        require(family in FAMILIES, "quality: unknown model family")
        fields(files, CHECKPOINT_FILES, f"quality {family} model")
        for value in files.values():
            digest(value, f"quality {family} model")
    require(isinstance(quality["datasets"], dict) and quality["datasets"], "quality: missing datasets")
    for name, dataset in quality["datasets"].items():
        fields(dataset, {"benchmark", "files"}, f"quality {name} dataset")
        require(dataset["benchmark"] in ("beir", "locomo"), f"quality {name}: unknown benchmark")
        names = [item.get("name") for item in dataset["files"]]
        require(names == list(BENCHMARK_FILES[dataset["benchmark"]]) if dataset["benchmark"] in BENCHMARK_FILES
                else len(names) == 1, f"quality {name}: dataset files differ from its benchmark")
        for item in dataset["files"]:
            fields(item, {"name", "bytes", "sha256"}, f"quality {name} file")
            require(ARTIFACT_NAME.fullmatch(item["name"]) and count(item["bytes"], "bytes", 1),
                    f"quality {name}: invalid file")
            digest(item["sha256"], f"quality {name} file")
    cases, seen, failures = quality["cases"], set(), []
    require(isinstance(cases, list) and cases, "quality: missing cases")
    for case in cases:
        key = (case.get("family"), case.get("dataset"))
        require(key not in seen, "quality: duplicate case")
        seen.add(key)
        _, case_failures = check_quality_case(case, quality, f"quality {key[0]} {key[1]}")
        failures += case_failures
    require(quality["individual_failed_tasks"] == failures, "quality: individual failures differ from the reports")


def validate_rotations(receipt, scope, check, root):
    """Raises ValueError unless `receipt` holds rotations of `scope`'s protocol on this checkout's
    sources, each passing `check` and none held twice; returns what `check` gave for each."""
    fields(receipt, RECEIPT_FIELDS, "receipt")
    privacy(receipt)
    require(receipt["schema_version"] == SCHEMA_VERSION and receipt["status"] == "validated"
            and receipt["scope"] == scope, "receipt identity differs")
    source = receipt["source"]
    fields(source, {"files", "files_sha256"}, "source")
    require(source["files"] == source_files(root), "source inventory differs from this checkout")
    require(source["files_sha256"] == inventory_digest(source["files"]), "source inventory digest differs")
    sources = {item["file"]: item["sha256"] for item in source["files"]}
    rotations = receipt["rotations"]
    require(isinstance(rotations, list) and rotations, "a receipt holds at least one rotation")
    results = [check(rotation, sources) for rotation in rotations]
    summaries = [rotation["provenance"]["rotation_summary_sha256"] for rotation in rotations]
    require(len(set(summaries)) == len(summaries), "a rotation is held twice")
    return results


def check_rotation(rotation, sources):
    """One rotation of the matched protocol against its own samples and `sources`, this
    checkout's digests; returns its recomputed aggregates."""
    fields(rotation, ROTATION_FIELDS, "rotation")
    provenance = rotation["provenance"]
    fields(provenance, PROVENANCE_FIELDS, "provenance")
    digest(provenance["rotation_summary_sha256"], "rotation summary")
    text(provenance["rotation_created_utc"], "rotation time")
    require((provenance["quality_manifest_sha256"] is None) == (rotation["quality"] is None)
            and (provenance["quality_manifest_sha256"] is None
                 or digest(provenance["quality_manifest_sha256"], "quality manifest")), "quality provenance differs")
    check_machine(rotation)
    check_settings(rotation)
    cases = check_cases(rotation["cases"])
    check_inputs(rotation, expected_inputs(rotation), sources)
    check_legs(rotation, cases)
    check_observations(rotation)
    rows = aggregate(rotation)
    require(rotation["aggregates"] == rows, "aggregates differ from the samples")
    require(rotation["counts"] == counts(rotation, rows), "counts differ from the samples")
    check_excluded(rotation["excluded_rotations"])
    if rotation["quality"] is not None:
        check_quality(rotation["quality"])
    check_disclosures(rotation)
    return rows


def validate(receipt, root=ROOT):
    """Raises ValueError unless `receipt` holds complete, self-consistent rotations of this
    checkout's sources; returns each rotation's recomputed aggregates."""
    return validate_rotations(receipt, SCOPE, check_rotation, root)


def check_disclosures(rotation):
    """The export policy against the legs' artifacts, and the operator's limitations."""
    policy = rotation["export_policy"]
    fields(policy, {"method", "raw_artifact_hashes_verified", "excluded_fields"}, "export policy")
    text(policy["method"], "export method")
    require(policy["raw_artifact_hashes_verified"] == sum(len(leg["artifacts"]) for leg in rotation["legs"]),
            "export policy: raw artifact count differs")
    require(isinstance(policy["excluded_fields"], dict)
            and all(isinstance(reason, str) and reason.strip() for reason in policy["excluded_fields"].values()),
            "export policy: invalid exclusions")
    require(isinstance(rotation["limitations"], list) and rotation["limitations"]
            and all(isinstance(item, str) and item.strip() for item in rotation["limitations"]), "missing limitations")


def measured(receipt, table=ENGINES):
    """Each case's family and lengths and the engines that measured it, over every rotation; raises
    if two rotations describe a case differently or measured it on the same engine."""
    cases = {}
    for rotation in receipt["rotations"]:
        for case in rotation["cases"]:
            described = (case["family"], case["lengths"])
            known, engines = cases.setdefault(case["name"], (described, set()))
            require(known == described, f"{case['name']}: rotations describe the case differently")
            for name in rotation["settings"]["engines"]:
                if case["family"] in table[name].families:
                    require(name not in engines, f"{case['name']}: two rotations measured {name}")
                    engines.add(name)
    return cases


def eligible(group, family):
    """The engines of `group` that run `family`."""
    return {name for name in group if family in ENGINES[name].families}


def check_release_quality(rotations):
    """Every release family's quality runs on every dataset, once over all `rotations`; returns
    each family's checkpoint digests."""
    models, covered = {}, set()
    for rotation in rotations:
        quality = rotation["quality"]
        if quality is None:
            continue
        require(set(quality["datasets"]) == set(DATASETS)
                and all(quality["datasets"][name]["benchmark"] == benchmark for name, benchmark in DATASETS.items()),
                "the release quality runs read other datasets")
        for family, files in quality["models"].items():
            require(family not in models, f"quality {family}: two rotations hold its runs")
            models[family] = files
        for case in quality["cases"]:
            label = f"quality {case['family']} {case['dataset']}"
            scored, identical = SCORED[case["family"]], IDENTICAL[case["family"]]
            require(set(case["engines"]) == {"reference", scored, *identical}, f"{label}: lacks a release run")
            require(all(vector[f"{run}_sha256"] == vector[f"{scored}_sha256"]
                        for run in identical for vector in case["vectors"]),
                    f"{label}: CUDA vectors differ from the CPU's")
            require(case["report"]["totals"]["quality_gates"]["pass"] is True,
                    f"{label}: aggregate quality gate failed")
            require(case["dataset"] != "fiqa"
                    or [task["passages"]["rows"] for task in case["report"]["tasks"]] == [FIQA_PASSAGES],
                    f"{label}: not the release's subset of FiQA")
            covered.add((case["family"], case["dataset"]))
    families = {family for family, _ in RELEASE_CASES.values()}
    require(set(models) == families and covered == {(family, dataset) for family in families for dataset in DATASETS},
            "the release quality runs do not cover every model on every dataset")
    return models


def validate_release(receipt, root=ROOT):
    """validate() plus the release's scope: every release case on the full GPU group and, up to
    RELEASE_CPU_TOKENS, on the full CPU group, in three-round rotations, and every model's quality runs."""
    rows = validate(receipt, root)
    rotations = receipt["rotations"]
    require(all(rotation["settings"]["rounds"] == 3 for rotation in rotations), "a release rotation has three rounds")
    cases = measured(receipt)
    require({name: described for name, (described, _) in cases.items()}
            == {name: (family, [length]) for name, (family, length) in RELEASE_CASES.items()},
            "release cases differ")
    for name, ((family, lengths), engines) in cases.items():
        require(eligible(GROUPS["gpu"], family) <= engines, f"{name}: the release lacks a GPU engine")
        cpu = {engine for engine in engines if ENGINES[engine].device == "cpu"}
        group = eligible(RELEASE_CPU_ENGINES, family) if max(lengths) <= RELEASE_CPU_TOKENS else set()
        require(cpu == group, f"{name}: the release's CPU engines differ")
    models = check_release_quality(rotations)
    for rotation in rotations:
        for case in rotation["cases"]:
            model = models[case["family"]]
            require(all(rotation["input_hashes"][f"{case['name']}/{name}"] == model[name] for name in CHECKPOINT_FILES),
                    f"{case['name']}: quality ran on a different checkpoint")
    return rows

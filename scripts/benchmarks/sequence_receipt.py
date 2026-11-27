"""The receipt of complete changing-lengths rotations and the operator's disclosures, and the
checks binding each rotation to its own samples and this checkout's sources."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

import export_receipt
import receipts
from _matched import ENCODERS, FAMILIES
from _sequence import PASSES, RETENTION, band_medians, pass_statistics, profile_shapes
from prepare_benchmark_fixtures import SEQUENCE_FIRST, sequence_order
from receipts import digest, fields, number, require, text
from run_length_sequence import ENGINES, GROUPS, SCOPE, sequence_timing

OUTPUT = receipts.ROOT / "docs" / "benchmarks" / "changing-lengths.json"
ROTATION_FIELDS = receipts.ROTATION_FIELDS - {"quality"}
PROVENANCE_FIELDS = {"rotation_created_utc", "rotation_summary_sha256"}
SETTINGS_FIELDS = (receipts.SETTINGS_FIELDS
                   - {"tokens", "samples_per_run", "calls_per_sample", "warm_seconds"}) | {"passes"}
TIMING_FIELDS = ("passes_ms", "pass_intervals_unix_ms", "timing_start_unix_ms", "timing_end_unix_ms", "first_pass",
                 "later_passes")
LEG_FIELDS = (receipts.LEG_FIELDS - {"samples_ms", "sample_intervals_unix_ms", "median_ms", "min_ms", "max_ms"}
              ) | set(TIMING_FIELDS)
COMMON_METADATA = (receipts.COMMON_METADATA - {"calls_per_sample", "real_tokens", "shape", "warm_seconds"}
                   ) | {"calls_per_pass", "passes"}
PYTHON_METADATA = receipts.PYTHON_METADATA - {"padded_array_sha256", "padded_tokens"}
TORCH_METADATA = receipts.TORCH_METADATA - {"attention_profile", "ordinary_forward_max_abs_difference"}
# An eager forward captures no CUDA graph.
TORCH_DEVICE_METADATA = {"cpu": receipts.TORCH_DEVICE_METADATA["cpu"],
                         "gpu": receipts.TORCH_DEVICE_METADATA["gpu"] - {"capture_seconds"}}
# Provider options as ONNX Runtime reports them for a session that takes any length: no graph replay.
PROVIDER_SETTINGS = {
    "CUDAExecutionProvider": {**receipts.PROVIDER_SETTINGS["CUDAExecutionProvider"], "enable_cuda_graph": "0"},
    "TensorrtExecutionProvider": {**receipts.PROVIDER_SETTINGS["TensorrtExecutionProvider"],
                                  "trt_cuda_graph_enable": "0"}}
STATISTICS = ("mean_ms", "median_ms", "p99_ms", "max_ms")
EXCLUDED_FIELDS = {name: reason for name, reason in export_receipt.EXCLUDED_FIELDS.items()
                   if not name.startswith("quality runs")}
RELEASE_CASE = "e5-sequence"


def later_mean(timing):
    return {"later_pass_mean_ms": timing["later_passes"]["mean_ms"]}


def check_settings(rotation):
    settings = rotation["settings"]
    fields(settings, SETTINGS_FIELDS, "settings")
    receipts.check_shared_settings(rotation, ENGINES)
    require(settings["passes"] == PASSES and settings["scope"] == SCOPE, "settings: sequence protocol differs")


def check_cases(rotation):
    cases = receipts.check_cases(rotation["cases"], {"order"})
    for case in cases.values():
        order = case["order"]
        require(isinstance(order, list) and all(type(index) is int for index in order)
                and sorted(order) == list(range(case["sequences"])),
                f"{case['name']}: the order must name every text once")
    return cases


def expected_inputs(rotation):
    """The input names a changing-lengths rotation of these cases and engines hashes."""
    names = receipts.expected_inputs(rotation, ENGINES) | {"sequence_runner", "sequence_helper"}
    if any(engine.kind == "rust" for engine in receipts.engines_of(rotation, ENGINES)):
        names.add("api_sequence_source")
    return names


def check_passes(leg, label):
    """A leg's passes against their wall-clock intervals, and its statistics against its samples."""
    passes = leg["passes_ms"]
    require(isinstance(passes, list) and passes and isinstance(passes[0], list), f"{label}: missing passes")
    timing = sequence_timing(leg, len(passes[0]))
    for samples, (left, right) in zip(passes, leg["pass_intervals_unix_ms"]):
        require(abs(right - left - sum(samples)) <= receipts.INTERVAL_TOLERANCE_MS * len(samples),
                f"{label}: a pass does not fit its wall-clock interval")
    for name in ("first_pass", "later_passes"):
        require(leg[name] == timing[name], f"{label}: {name} differs from its samples")


def check_metadata(leg, rotation, case, shared, label):
    engine, family = ENGINES[leg["engine"]], FAMILIES[case["family"]]
    metadata, settings = leg["metadata"], rotation["settings"]
    expected = COMMON_METADATA | (receipts.RUST_METADATA if engine.kind == "rust" else PYTHON_METADATA | (
        receipts.ORT_METADATA if engine.kind == "ort" else
        TORCH_METADATA | TORCH_DEVICE_METADATA[engine.device] | receipts.TORCH_ENCODER_METADATA[family.encoder]))
    if engine.selector == "tensorrt":
        expected = expected | {"trt_cache_sha256"}
    fields(metadata, expected, f"{label} metadata")
    require(metadata["engine"] == engine.receipt, f"{label}: engine label differs")
    require(metadata["scope"] == "end-to-end" and metadata["passes"] == PASSES and metadata["warm_calls"] == 0
            and metadata["calls_per_pass"] == case["sequences"] == len(leg["passes_ms"][0])
            and metadata["caller_retention"] == RETENTION, f"{label}: timed boundary differs")
    require(metadata["family"] == case["family"] and metadata["role"] == case["role"]
            and metadata["label"] == case["name"] and metadata["lengths"] == case["lengths"],
            f"{label}: timed geometry differs")
    require(metadata["hidden"] == family.hidden and metadata["max_length"] == family.max_length,
            f"{label}: model geometry differs")
    require(metadata["settle_seconds"] == settings["settle_seconds"] and number(metadata["load_seconds"], "load") >= 0,
            f"{label}: settle or load time differs")
    text(metadata["scope_description"], "scope description")
    receipts.check_lineage(metadata, rotation["input_hashes"], case, shared, label)
    if engine.kind == "rust":
        receipts.check_rust_metadata(metadata, engine, settings, label)
        if engine.selector.endswith("-centered"):
            vectors = centered_vectors(leg, label)
            # Centered INT8 gives the same bits on every device, so its legs save one file.
            require(shared.setdefault(("centered", case["name"]), vectors) == vectors,
                    f"{label}: centered vectors differ between legs")
    else:
        check_python_metadata(leg, rotation, case, engine, shared, label)


def centered_vectors(leg, label):
    """The digest of the one vector file a centered leg saves."""
    vectors = [value for name, value in leg["artifacts"].items() if name.endswith(".vectors.f32")]
    require(len(vectors) == 1, f"{label}: a centered leg saves one vector file")
    return vectors[0]


def check_python_metadata(leg, rotation, case, engine, shared, label):
    metadata = leg["metadata"]
    threads = receipts.check_python_run(leg, rotation, engine, shared, "sequence_helper", label)
    require(metadata["tokenizer_method"] == "encode", f"{label}: unknown tokenizer method")
    if engine.kind == "ort":
        # One engine over every length: the shortest, the middle and the longest as its profile.
        profile = profile_shapes(case["lengths"], ENCODERS[FAMILIES[case["family"]].encoder].inputs)
        receipts.check_ort_run(leg, rotation, case, engine, PROVIDER_SETTINGS, profile, label)
    else:
        check_torch_metadata(metadata, engine, case, threads, rotation["input_hashes"], label)


def check_torch_metadata(metadata, engine, case, threads, hashes, label):
    require(metadata["helper_sha256"] == hashes["sequence_helper"], f"{label}: helper source differs")
    receipts.check_torch_encoder(metadata, case, label)
    require(metadata["all_tokens_valid"] is True, f"{label}: a text of the sequence was padded")
    text(metadata["attention_dispatch"], "attention dispatch")
    require(metadata["transformer_precision"] == receipts.TRANSFORMER_PRECISION[engine.device],
            f"{label}: precision differs")
    if engine.device == "cpu":
        require(metadata["intra_op_threads"] == threads and metadata["interop_threads"] == 1,
                f"{label}: torch threads differ")
    else:
        require(metadata["allow_tf32"] is False, f"{label}: TF32 was allowed")


def speedup(own, others):
    """Each other engine's time over `own`, and the fastest one's."""
    fastest = min(others, key=others.get)
    return {"fastest_competitor": fastest, "vs_fastest_competitor": others[fastest] / own,
            "pairwise": {name: value / own for name, value in others.items()}}


def aggregate(rotation):
    """Per case and device, from the samples: each engine's statistics and band medians as the median
    over its rounds, and the other engines' later-pass mean and band medians over each KodamaEmbed mode's."""
    rows = []
    for case in rotation["cases"]:
        call_lengths = [case["lengths"][index] for index in case["order"]]
        for device in ("gpu", "cpu"):
            engines = [name for name in rotation["settings"]["engines"]
                       if ENGINES[name].device == device and case["family"] in ENGINES[name].families]
            if not engines:
                continue
            stats = {}
            for engine in engines:
                runs = sorted((leg for leg in rotation["legs"]
                               if leg["case"] == case["name"] and leg["engine"] == engine), key=lambda leg: leg["round"])
                rounds = [pass_statistics(leg["passes_ms"]) for leg in runs]
                longest = FAMILIES[case["family"]].max_length
                bands = [band_medians(leg["passes_ms"], call_lengths, longest) for leg in runs]
                stats[engine] = {
                    **{group: {name: statistics.median(item[group][name] for item in rounds) for name in STATISTICS}
                       for group in ("first_pass", "later_passes")},
                    "later_pass_means_ms": [item["later_passes"]["mean_ms"] for item in rounds],
                    "bands": [{"tokens": band[0]["tokens"],
                               "median_ms": statistics.median(item["median_ms"] for item in band)}
                              for band in zip(*bands)]}
            competitors = [name for name in engines if ENGINES[name].kind != "rust"]
            speedups = {}
            for engine in engines:
                if ENGINES[engine].kind == "rust" and competitors:
                    own = stats[engine]
                    speedups[engine] = {
                        "later_pass_mean": speedup(own["later_passes"]["mean_ms"], {
                            name: stats[name]["later_passes"]["mean_ms"] for name in competitors}),
                        "bands": [{"tokens": band["tokens"], **speedup(band["median_ms"], {
                            name: stats[name]["bands"][index]["median_ms"] for name in competitors})}
                                  for index, band in enumerate(own["bands"])]}
            rows.append({"case": case["name"], "family": case["family"], "device": device, "engines": stats,
                         "speedups": speedups})
    return rows


def counts(rotation):
    legs = rotation["legs"]
    return {"cases": len(rotation["cases"]), "accepted_legs": len(legs),
            "timed_calls": sum(len(samples) for leg in legs for samples in leg["passes_ms"])}


def check_rotation(rotation, sources):
    """One changing-lengths rotation against its own samples and `sources`, this checkout's
    digests; returns its recomputed aggregates."""
    fields(rotation, ROTATION_FIELDS, "rotation")
    provenance = rotation["provenance"]
    fields(provenance, PROVENANCE_FIELDS, "provenance")
    digest(provenance["rotation_summary_sha256"], "rotation summary")
    text(provenance["rotation_created_utc"], "rotation time")
    receipts.check_machine(rotation, ENGINES)
    check_settings(rotation)
    cases = check_cases(rotation)
    receipts.check_inputs(rotation, expected_inputs(rotation), sources)
    receipts.check_legs(rotation, cases, ENGINES, LEG_FIELDS, check_passes, check_metadata)
    receipts.check_observations(rotation)
    rows = aggregate(rotation)
    require(rotation["aggregates"] == rows, "aggregates differ from the samples")
    require(rotation["counts"] == counts(rotation), "counts differ from the samples")
    receipts.check_excluded(rotation["excluded_rotations"], "later_pass_mean_ms")
    receipts.check_disclosures(rotation)
    return rows


def validate(receipt, root=receipts.ROOT):
    """Raises ValueError unless `receipt` holds complete, self-consistent changing-lengths rotations
    of this checkout's sources; returns each rotation's recomputed aggregates."""
    return receipts.validate_rotations(receipt, SCOPE, check_rotation, root)


def validate_release(receipt, root=receipts.ROOT):
    """validate() plus the release's scope: e5-large over every length in the generator's order, on the
    full GPU group and on the centered CPU engine, whose vectors the centered CUDA legs must equal."""
    rows = validate(receipt, root)
    rotations = receipt["rotations"]
    lengths = list(range(SEQUENCE_FIRST, FAMILIES["e5"].max_length + 1))
    release = (RELEASE_CASE, "e5", "query", lengths, sequence_order(len(lengths)))
    require(all((case["name"], case["family"], case["role"], case["lengths"], case["order"]) == release
                for rotation in rotations for case in rotation["cases"]), "the release sequence differs")
    require(all(rotation["settings"]["rounds"] == 3 for rotation in rotations), "a release rotation has three rounds")
    _, engines = receipts.measured(receipt, ENGINES)[RELEASE_CASE]
    require({*GROUPS["gpu"], "kodamaembed-cpu-centered"} <= engines, "the release lacks an engine")
    centered = {centered_vectors(leg, leg["engine"]) for rotation in rotations for leg in rotation["legs"]
                if ENGINES[leg["engine"]].selector.endswith("-centered")}
    require(len(centered) == 1, "centered vectors differ between rotations")
    return rows


def build_rotation(directory, disclosures):
    """The record of the complete rotation in `directory`, unvalidated; export_receipt.publish()
    validates the receipt that holds it."""
    directory, disclosures = Path(directory), Path(disclosures)
    summary, settings, limits, notes = export_receipt.read_rotation(directory, disclosures)
    calls = {case["name"]: case["sequences"] for case in summary["cases"]}
    legs = [export_receipt.published_leg(directory, leg, summary, limits, ENGINES,
                                         lambda data, leg=leg: sequence_timing(data, calls[leg["case"]]), TIMING_FIELDS)
            for leg in summary["legs"]]
    rotation = {
        "provenance": {"rotation_created_utc": summary["created_utc"],
                       "rotation_summary_sha256": receipts.file_sha256(directory / "summary.json")},
        "platform": summary["platform"], "hardware": summary["hardware"], "speed_settings": summary["speed_settings"],
        "settings": settings, "input_hashes": summary["hashes"], "final_input_hashes": summary["final_hashes"],
        "cases": summary["cases"], "legs": legs,
        "background_process_observations": export_receipt.published_observations(
            summary["background_process_observations"]),
        "excluded_rotations": [export_receipt.stopped_rotation(disclosures.parent / item["directory"], item["reason"],
                                                               later_mean) for item in notes["stopped_rotations"]],
        "export_policy": {"method": export_receipt.EXPORT_METHOD,
                          "raw_artifact_hashes_verified": sum(len(leg["artifacts"]) for leg in legs),
                          "excluded_fields": EXCLUDED_FIELDS},
        "limitations": notes["limitations"],
    }
    rotation["aggregates"] = aggregate(rotation)
    rotation["counts"] = counts(rotation)
    return rotation


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", required=True, help="the complete rotation's output directory")
    parser.add_argument("--disclosures", required=True, help="JSON as export_receipt.py reads it")
    export_receipt.publishing_options(parser, OUTPUT, "accept a receipt that does not yet hold the whole release: "
                                                      "its sequence on every engine")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    rotation = build_rotation(args.rotation, args.disclosures)
    export_receipt.publish(rotation, SCOPE, validate if args.partial else validate_release, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

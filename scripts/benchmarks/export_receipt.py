"""A benchmark receipt: one complete rotation, its quality runs and the operator's disclosures, alone or
appended to the rotations a receipt holds; artifacts rehashed, telemetry judged again, local identifiers removed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import receipts
from _telemetry import check_timing
from receipts import fields, require
from run_query_comparison import ARTIFACT_NAME, ENGINES, json_timing, open_artifact, read_matched_result

OUTPUT = receipts.ROOT / "docs" / "benchmarks" / "release.json"
SUMMARY_FIELDS = {"schema_version", "created_utc", "status", "valid", "settings", "platform", "hardware",
                  "speed_settings", "hashes", "background_process_observations", "legs", "aggregates", "failure",
                  "cases", "final_hashes"}
SOURCE_LEG_FIELDS = {"round", "case", "key", "engine", "device", "status", "valid", "gate", "timing", "telemetry",
                     "exit_code", "process_scheduling", "load_average_before", "load_average_after", "stdout",
                     "stderr", "result", "artifacts"}
MONITOR_LOG_FIELDS = {"monitor_stdout", "monitor_stderr"}
TIMING_FIELDS = ("samples_ms", "sample_intervals_unix_ms", "timing_start_unix_ms", "timing_end_unix_ms",
                 "median_ms", "min_ms", "max_ms")
TRT_CACHE_PATHS = ("trt_engine_cache_path", "trt_timing_cache_path")
QUALITY_MANIFEST_FIELDS = {"schema_version", "status", "started_utc", "finished_utc", "cuda", "validator_sha256",
                           "models", "datasets", "cases", "failure"}
QUALITY_CASE_FIELDS = {"family", "dataset", "engines", "vectors", "report", "report_sha256"}
EXPORT_METHOD = ("Explicit field lists; an unknown source field stops the export. Samples, intervals, numeric "
                 "metadata and hashes are copied without rounding.")
EXCLUDED_FIELDS = {
    "summary.aggregates": "Recomputed from the retained samples",
    "summary.background_process_observations.processes":
        "Local process names and identifiers; each check's count of allowed idle GPU contexts is retained",
    "settings.allowed_idle_process_ids": "Local process identifiers; their count is retained",
    "leg.key": "Derived from case and engine",
    "leg.stdout, stderr, monitor_stdout, monitor_stderr, result":
        "Local file names; artifact hashes and structured telemetry are retained",
    "metadata.python_executable": "A local path, replaced by the interpreter's recorded SHA-256",
    "metadata.scheduling*.process_id": "A local process identifier; the scheduling readback is retained",
    "metadata.provider_options.TensorrtExecutionProvider.trt_*_cache_path":
        "Local cache directories; the cache files' hashes are retained",
    "quality runs: model, data and output paths and device identifiers":
        "Local; checkpoint, dataset and vector hashes and engine fingerprints are retained",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def published_metadata(metadata, engine, hashes, system):
    """An engine receipt's metadata without local paths or identifiers."""
    result = copy.deepcopy(metadata)
    if engine.kind in ("ort", "torch"):
        require("python_executable" in result, "source metadata lacks its interpreter")
        del result["python_executable"]
        result["python_executable_sha256"] = hashes["python_executable" if engine.kind == "ort" else "torch_executable"]
        for name in receipts.PYTHON_SCHEDULING:
            fields(result[name], receipts.SCHEDULING_FIELDS[system] | {"process_id"}, f"source {name}")
            del result[name]["process_id"]
    if engine.selector == "tensorrt":
        options = result["provider_options"]["TensorrtExecutionProvider"]
        for key in TRT_CACHE_PATHS:
            options.pop(key, None)
    return result


def published_leg(directory, source, summary, limits, table=ENGINES, read=json_timing, timing_fields=TIMING_FIELDS):
    """One accepted leg, verified against the receipt, artifacts and monitor log its rotation retained
    in `directory`; a protocol gives its engine `table`, its timing's `read` and its `timing_fields`."""
    engine = table[source["engine"]]
    fields(source, SOURCE_LEG_FIELDS | (MONITOR_LOG_FIELDS if engine.device == "gpu" else set()), "source leg")
    require(source["key"] == f"{source['case']}/{source['engine']}" and source["status"] == "accepted"
            and source["valid"] is True and source["result"] in source["artifacts"], "source leg is not accepted")
    require(isinstance(source["result"], str) and ARTIFACT_NAME.fullmatch(source["result"]),
            "the result must be a local filename")
    timing, artifacts = read_matched_result(directory / source["result"], engine, read)
    require(artifacts == source["artifacts"], "the summary's artifacts differ from the leg's receipt")
    for name in source["artifacts"]:
        if name.endswith(".placement.json"):
            with open_artifact(directory / name) as audit:
                placement = audit.read()
            require(hashlib.sha256(placement).hexdigest() == artifacts[name], "the placement audit changed")
            require(json.loads(placement)["unexpected"] == [], "a graph node left its requested execution provider")
    require(timing == source["timing"], "the summary's timing differs from the leg's receipt")
    telemetry = None
    if engine.device == "gpu":
        log = (directory / source["monitor_stdout"]).read_bytes().decode("utf-8", "replace")
        telemetry = check_timing(log, timing["timing_start_unix_ms"], timing["timing_end_unix_ms"], limits)
        require(telemetry == source["telemetry"], "the summary's telemetry differs from the monitor log")
    return {"round": source["round"], "case": source["case"], "engine": source["engine"], "device": engine.device,
            "status": "accepted", "valid": True, "exit_code": source["exit_code"],
            **{name: timing[name] for name in timing_fields},
            "metadata": published_metadata(timing["metadata"], engine, summary["hashes"],
                                           summary["platform"]["system"]),
            "process_scheduling": source["process_scheduling"],
            "load_average_before": source["load_average_before"], "load_average_after": source["load_average_after"],
            "gate": source["gate"], "telemetry": telemetry, "artifacts": source["artifacts"]}


def published_settings(settings):
    result = {name: value for name, value in settings.items() if name != "allowed_idle_process_ids"}
    result["allowed_idle_contexts"] = len(settings["allowed_idle_process_ids"])
    return result


def published_observations(observations):
    """Each GPU process check as the count of allowed idle contexts it found."""
    result = []
    for observation in observations:
        fields(observation, {"checked_utc", "stage", "processes"}, "source GPU process check")
        for process in observation["processes"]:
            fields(process, {"process_id", "name", "explicitly_allowed"}, "source GPU process")
            require(process["explicitly_allowed"] is True, "an unallowed GPU process ran beside a benchmark")
        result.append({"checked_utc": observation["checked_utc"], "stage": observation["stage"],
                       "allowed_idle_contexts": len(observation["processes"])})
    return result


def matched_statistic(timing):
    return {"median_ms": timing["median_ms"]}


def stopped_rotation(directory, reason, statistic=matched_statistic):
    """A rotation that stopped before completing, disclosed with its accepted legs and each
    one's `statistic`."""
    path = Path(directory) / "summary.json"
    summary = load(path)
    require(summary.get("status") in ("preparing", "running", "failed") and summary.get("valid") is False,
            f"{path.parent.name} did not stop before completing")
    legs = []
    for leg in summary["legs"]:
        item = {name: leg[name] for name in ("round", "case", "engine", "status")}
        if leg["status"] == "accepted":
            item.update(statistic(leg["timing"]))
        legs.append(item)
    last = summary["legs"][-1] if summary["legs"] else {}
    failure = summary["failure"]
    return {"rotation_summary_sha256": receipts.file_sha256(path), "recorded_status": summary["status"],
            "recorded_valid": False, "included_in_ranking": False,
            "accepted_legs": sum(leg["status"] == "accepted" for leg in legs),
            "stopped_at": {"stage": failure["stage"].rsplit(" ", 1)[-1] if failure else "interrupted",
                           "round": last.get("round"), "case": last.get("case"), "engine": last.get("engine")},
            "reason": receipts.text(reason, "stopping reason"), "legs": legs}


def published_quality(directory):
    """The quality runs with every report and vector file rehashed; returns them
    with the manifest's digest."""
    manifest_path = Path(directory) / "quality.json"
    manifest = load(manifest_path)
    fields(manifest, QUALITY_MANIFEST_FIELDS, "quality manifest")
    require(manifest["schema_version"] == 1 and manifest["status"] == "complete" and manifest["failure"] is None,
            "the quality runs did not complete")
    cases, failures = [], []
    for case in manifest["cases"]:
        fields(case, QUALITY_CASE_FIELDS, "quality case")
        report_path = manifest_path.parent / case["report"]
        require(".." not in Path(case["report"]).parts and receipts.file_sha256(report_path) == case["report_sha256"],
                "a quality report changed after its run")
        for vector in case["vectors"]:
            for run in case["engines"]:
                require(receipts.file_sha256(report_path.parent / run / vector["name"]) == vector[f"{run}_sha256"],
                        "a saved vector file changed after its run")
        report = load(report_path)
        failures += [{"family": case["family"], "dataset": case["dataset"], "task": task["task"],
                      "quality_gates": task["quality_gates"]}
                     for task in report["tasks"] if task["quality_gates"]["pass"] is not True]
        cases.append({"family": case["family"], "dataset": case["dataset"], "engines": case["engines"],
                      "vectors": case["vectors"], "report": report})
    quality = {name: manifest[name] for name in ("validator_sha256", "started_utc", "finished_utc", "models",
                                                 "datasets")}
    quality.update(cases=cases, individual_failed_tasks=failures)
    return quality, receipts.file_sha256(manifest_path)


def read_rotation(directory, disclosures):
    """A complete rotation's summary, its published settings and gate limits, and the
    operator's disclosures."""
    summary = load(Path(directory) / "summary.json")
    fields(summary, SUMMARY_FIELDS, "rotation summary")
    require(summary["schema_version"] == 1 and summary["status"] == "complete" and summary["valid"] is True
            and summary["failure"] is None, "the rotation did not complete")
    settings = published_settings(summary["settings"])
    limits = receipts.gpu_limits(settings) if settings["minimum_enforced_power_w"] is not None else None
    notes = load(disclosures)
    fields(notes, {"limitations", "stopped_rotations"}, "disclosures")
    for item in notes["stopped_rotations"]:
        fields(item, {"directory", "reason"}, "stopped rotation disclosure")
    return summary, settings, limits, notes


def build_rotation(directory, disclosures, quality=None):
    """The record of the complete rotation in `directory`, unvalidated; publish() validates the
    receipt that holds it."""
    directory, disclosures = Path(directory), Path(disclosures)
    summary_path = directory / "summary.json"
    summary, settings, limits, notes = read_rotation(directory, disclosures)
    quality_record, manifest_sha256 = published_quality(quality) if quality is not None else (None, None)
    legs = [published_leg(directory, leg, summary, limits) for leg in summary["legs"]]
    rotation = {
        "provenance": {"rotation_created_utc": summary["created_utc"],
                       "rotation_summary_sha256": receipts.file_sha256(summary_path),
                       "quality_manifest_sha256": manifest_sha256},
        "platform": summary["platform"], "hardware": summary["hardware"], "speed_settings": summary["speed_settings"],
        "settings": settings, "input_hashes": summary["hashes"], "final_input_hashes": summary["final_hashes"],
        "cases": summary["cases"], "legs": legs,
        "background_process_observations": published_observations(summary["background_process_observations"]),
        "excluded_rotations": [stopped_rotation(disclosures.parent / item["directory"], item["reason"])
                               for item in notes["stopped_rotations"]],
        "quality": quality_record,
        "export_policy": {"method": EXPORT_METHOD,
                          "raw_artifact_hashes_verified": sum(len(leg["artifacts"]) for leg in legs),
                          "excluded_fields": EXCLUDED_FIELDS},
        "limitations": notes["limitations"],
    }
    rotation["aggregates"] = receipts.aggregate(rotation)
    rotation["counts"] = receipts.counts(rotation, rotation["aggregates"])
    return rotation


def receipt_of(rotations, scope, root=receipts.ROOT):
    """A receipt holding `rotations` beside the source inventory of the checkout at `root`."""
    files = receipts.source_files(root)
    return {"schema_version": receipts.SCHEMA_VERSION, "status": "validated", "scope": scope,
            "source": {"files": files, "files_sha256": receipts.inventory_digest(files)}, "rotations": rotations}


def publish(rotation, scope, validate, args):
    """Writes a receipt of `rotation` to a new args.out, or under --append adds it to the rotations
    args.out holds, once `validate` accepts the whole; --check writes nothing and requires args.out to hold it."""
    out = args.out
    if args.append or args.check:
        receipt = load(out)
        fields(receipt, receipts.RECEIPT_FIELDS, "receipt")
        if args.check:
            require(rotation in receipt["rotations"], f"{out.name} does not hold this rotation as a fresh export gives it")
        else:
            receipt["rotations"].append(rotation)
    else:
        require(not out.exists(), f"{out.name} exists; --append adds this rotation to it")
        receipt = receipt_of([rotation], scope)
    validate(receipt)
    payload = (json.dumps(receipt, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    if not args.check:
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out.parent) as directory:
            temporary = Path(directory) / out.name
            temporary.write_bytes(payload)
            temporary.replace(out)
    print(json.dumps({"receipt": out.name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                      "rotations": len(receipt["rotations"]), "counts": rotation["counts"]}, indent=2))


def publishing_options(parser, default, partial):
    """The options of every protocol's export: where the receipt is and how the rotation joins it."""
    parser.add_argument("--out", type=Path, default=default)
    parser.add_argument("--partial", action="store_true", help=partial)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--append", action="store_true",
                      help="add the rotation to those --out holds, so a later measurement joins its receipt")
    mode.add_argument("--check", action="store_true",
                      help="write nothing; require --out to hold the rotation as a fresh export gives it")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", required=True, help="the complete rotation's output directory")
    parser.add_argument("--disclosures", required=True,
                        help='JSON, directories relative to it: {"limitations": ["..."], '
                             '"stopped_rotations": [{"directory": "rotation-01", "reason": "..."}]}')
    parser.add_argument("--quality", help="run_quality.py's output directory for the rotation's models")
    publishing_options(parser, OUTPUT, "accept a receipt that does not yet hold the whole release: every case, "
                                       "engine and quality run")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    rotation = build_rotation(args.rotation, args.disclosures, args.quality)
    publish(rotation, receipts.SCOPE, receipts.validate if args.partial else receipts.validate_release, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

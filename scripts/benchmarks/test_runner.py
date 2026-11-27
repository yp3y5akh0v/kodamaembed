"""The comparison runner's rules without a benchmark: leg acceptance, engine selection, runs
that stop in preparation and each leg's command."""

import contextlib
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import bench_ort_query
import bench_torch_query
import run_query_comparison as runner
from _process import PLATFORM, Scheduling

CASE = "minilm-b1-l16"
RETENTION = ("three returned outputs retained until the sample clock stops; "
             "validation and release outside timing")

# What a Python engine reports here when asked for processors 0-15 at high priority.
REQUESTED = Scheduling(None if PLATFORM == "macos" else tuple(range(16)), "high")
if PLATFORM == "windows":
    STATE = {"platform": "windows", "process_id": 4242, "priority_class": 128, "affinity_mask": 65535,
             "execution_speed_control_mask": 1, "execution_speed_state_mask": 0}
    PRIORITY, PROCESSORS = ("priority_class", 32), ("affinity_mask", 1)
elif PLATFORM == "linux":
    STATE = {"platform": "linux", "process_id": 4242, "cpus": list(range(16)), "nice": -10}
    PRIORITY, PROCESSORS = ("nice", 0), ("cpus", [0])
else:
    STATE = {"platform": "macos", "process_id": 4242, "nice": -10}
    PRIORITY, PROCESSORS = ("nice", 0), None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EvidenceTests(unittest.TestCase):
    """The checks one leg's receipt and artifacts must pass."""

    def setUp(self):
        self.directory = None
        self.reset()

    def tearDown(self):
        self.directory.cleanup()

    def reset(self):
        """A fresh accepted receipt with its artifacts, for the next case."""
        if self.directory is not None:
            self.directory.cleanup()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.result = self.root / "leg.json"
        self.vectors = self.root / "leg.outputs.npy"
        self.vectors.write_bytes(bytes([1, 2, 3, 4]))
        self.placement = self.root / "leg.placement.json"
        self.placement.write_text('{"unexpected":[]}')
        self.input = self.root / "input.bin"
        self.input.write_text("initial")
        self.receipt = {
            "schema_version": 1, "samples_ms": [1.0] * 51,
            "timing_start_unix_ms": 1000.0, "timing_end_unix_ms": 1600.0,
            "sample_intervals_unix_ms": [[1001.0 + index * 10, 1005.0 + index * 10] for index in range(51)],
            "metadata": {
                "engine": "cuda", "scope": "end-to-end", "calls_per_sample": 3, "family": "minilm", "role": "query",
                "fixture_sha256": "a" * 64, "scheduling": dict(STATE), "scheduling_before_timing": dict(STATE),
                "scheduling_after_timing": dict(STATE), "caller_retention": RETENTION,
                "checkpoint_sha256": {"config.json": "b" * 64, "tokenizer.json": "c" * 64, "model.safetensors": "d" * 64},
                "input_ids_types_sha256": "e" * 64, "onnx_sha256": {"model.onnx": "f" * 64},
                "export_lineage": {"receipt_sha256": "1" * 64, "checkpoint_and_graph_hashes_match": True},
                "outputs": "leg.outputs.npy", "outputs_sha256": digest(self.vectors),
                "placement_audit_sha256": digest(self.placement),
            },
        }
        self.write()

    def write(self):
        self.result.write_text(json.dumps(self.receipt))

    def timing(self, matched=True):
        return runner.read_timing(self.result, matched)

    def accept(self, engine="ort-cuda"):
        graphs = {"cuda_onnx": "model.onnx", "trt_onnx": "model.onnx", "cpu_onnx": "model.onnx"}
        plan = runner.Plan(CASE, engine, f"{CASE}/{engine}", graphs, "minilm", "query")
        hashes = {f"{CASE}/fixture": "a" * 64, f"{CASE}/config.json": "b" * 64, f"{CASE}/tokenizer.json": "c" * 64,
                  f"{CASE}/model.safetensors": "d" * 64, f"{CASE}/export_receipt": "1" * 64,
                  **{f"{CASE}/{graph}": "f" * 64 for graph in graphs}}
        timing, artifacts = runner.read_matched_result(self.result, runner.ENGINES[engine])
        runner.accept_matched(timing, plan, hashes, {}, REQUESTED)
        return artifacts

    def assert_rejected(self, action):
        with self.assertRaises((ValueError, OSError)):
            action()

    def test_matched_timing_keeps_its_intervals(self):
        timing = self.timing()
        self.assertEqual((len(timing["sample_intervals_unix_ms"]), timing["median_ms"]), (51, 1.0))

    def test_diagnostic_timing_needs_no_intervals(self):
        del self.receipt["sample_intervals_unix_ms"]
        self.write()
        self.timing(matched=False)
        log = self.root / "leg.csv"
        with log.open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["sample", "latency_ms", "start_unix_ms", "end_unix_ms"])
            writer.writerows([index, 1.0, 1001 + index * 10, 1005 + index * 10] for index in range(51))
        self.assertEqual(len(runner.read_timing(log, False)["samples_ms"]), 51)

    def test_matched_timing_rejects_every_malformed_interval(self):
        intervals = "sample_intervals_unix_ms"
        for name, change in {
            "missing": lambda receipt: receipt.pop(intervals),
            "wrong count": lambda receipt: receipt.update({intervals: [[1001, 1005]]}),
            "out of bounds": lambda receipt: receipt[intervals][50].__setitem__(1, 1601),
            "overlap": lambda receipt: receipt[intervals][1].__setitem__(0, 1002),
            "malformed pair": lambda receipt: receipt[intervals].__setitem__(0, [1001, 1002, 1003]),
            "numeric string": lambda receipt: receipt[intervals][0].__setitem__(0, "1001"),
            "boolean": lambda receipt: receipt[intervals][0].__setitem__(0, True),
            "not finite": lambda receipt: receipt[intervals][0].__setitem__(0, float("nan")),
            "nonpositive sample": lambda receipt: receipt["samples_ms"].__setitem__(0, 0.0),
            "missing sample": lambda receipt: receipt["samples_ms"].pop(),
            "missing end timestamp": lambda receipt: receipt.pop("timing_end_unix_ms"),
        }.items():
            with self.subTest(name=name):
                receipt = copy.deepcopy(self.receipt)
                change(receipt)
                self.result.write_text(json.dumps(receipt))
                self.assert_rejected(self.timing)

    def test_input_snapshots_reject_changes_and_substituted_keys(self):
        paths = {"input": self.input, "input_alias": self.input}
        snapshot = {"input": digest(self.input), "input_alias": digest(self.input)}
        self.assertEqual(runner.assert_input_hashes(paths, snapshot), snapshot)
        self.assert_rejected(lambda: runner.assert_input_hashes({"input": self.input}, {}))
        self.assert_rejected(lambda: runner.assert_input_hashes({"input": self.input}, {"other": snapshot["input"]}))
        self.input.write_text("changed")
        self.assert_rejected(lambda: runner.assert_input_hashes(paths, snapshot))
        self.input.unlink()
        self.assert_rejected(lambda: runner.assert_input_hashes(paths, snapshot))

    def test_each_engine_kind_is_accepted_with_its_artifacts(self):
        self.assertEqual(set(self.accept()), {"leg.outputs.npy", "leg.placement.json", "leg.json"})
        for engine, label in (("trt-fp16", "tensorrt"), ("ort-cpu", "cpu")):
            self.receipt["metadata"]["engine"] = label
            self.write()
            self.accept(engine)
        for engine in ("torch-sdpa-graph", "torch-cpu"):
            self.receipt["metadata"]["engine"] = engine
            self.write()
            self.assertEqual(set(self.accept(engine)), {"leg.outputs.npy", "leg.json"})
        rust = self.root / "leg.vectors.f32"
        rust.write_bytes(bytes([1, 2, 3, 4]))
        self.receipt["metadata"].update(engine="kodamaembed", mode="cuda-centered")
        self.receipt["vectors"] = {"path": rust.name, "sha256": digest(rust)}
        self.write()
        self.assertEqual(set(self.accept("kodamaembed-cuda-centered")), {"leg.vectors.f32", "leg.json"})
        self.assert_rejected(lambda: self.accept("kodamaembed-cpu-fp32"))

    def test_artifact_problems_are_rejected(self):
        cases = {
            "changed vector": lambda metadata: self.vectors.write_text("changed"),
            "missing vector": lambda metadata: self.vectors.unlink(),
            "changed placement": lambda metadata: self.placement.write_text("changed"),
            "missing placement": lambda metadata: self.placement.unlink(),
            "empty digest": lambda metadata: metadata.update(outputs_sha256=""),
            "object digest": lambda metadata: metadata.update(outputs_sha256={"sha256": metadata["outputs_sha256"]}),
            "digest array": lambda metadata: metadata.update(outputs_sha256=[metadata["outputs_sha256"]]),
            "uppercase digest": lambda metadata: metadata.update(outputs_sha256=metadata["outputs_sha256"].upper()),
            "sibling artifact": lambda metadata: (self.root.joinpath("other.outputs.npy").write_bytes(bytes([1, 2, 3, 4])),
                                                  metadata.update(outputs="other.outputs.npy")),
            "parent path": lambda metadata: metadata.update(outputs="../leg.outputs.npy"),
            "absolute path": lambda metadata: metadata.update(outputs=str(self.vectors)),
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                self.reset()
                change(self.receipt["metadata"])
                self.write()
                self.assert_rejected(self.accept)

    def test_links_are_not_artifacts(self):
        target = self.root / "elsewhere.npy"
        target.write_bytes(self.vectors.read_bytes())
        self.vectors.unlink()
        try:
            os.symlink(target, self.vectors)
        except OSError as error:
            if getattr(error, "winerror", None) != 1314:
                raise
            # NTFS junctions exercise the reparse-point check without the symlink privilege.
            import _winapi
            directory = self.root / "elsewhere"
            directory.mkdir()
            _winapi.CreateJunction(str(directory), str(self.vectors))
        with self.assertRaisesRegex(ValueError, "link or reparse point"):
            self.accept()

    def test_artifact_snapshots_reject_paths_and_stream_names(self):
        for name in ("../leg.outputs.npy", "leg.outputs.npy:stream"):
            with self.subTest(name=name):
                self.assert_rejected(lambda: runner.assert_artifact_hashes(self.root, {name: digest(self.vectors)}))

    def test_artifacts_changed_after_acceptance_fail_completion(self):
        for target in ("vectors", "placement", "result"):
            with self.subTest(target=target):
                self.reset()
                accepted = self.accept()
                runner.assert_artifact_hashes(self.root, accepted)
                getattr(self, target).write_text("later")
                self.assert_rejected(lambda: runner.assert_artifact_hashes(self.root, accepted))

    def test_graph_and_export_lineage_must_match_the_snapshot(self):
        self.accept()
        for name, change in {
            "graph": lambda metadata: metadata["onnx_sha256"].update({"model.onnx": "3" * 64}),
            "export receipt": lambda metadata: metadata["export_lineage"].update(receipt_sha256="4" * 64),
            "both": lambda metadata: (metadata["onnx_sha256"].update({"model.onnx": "3" * 64}),
                                      metadata["export_lineage"].update(receipt_sha256="4" * 64)),
        }.items():
            with self.subTest(name=name):
                self.reset()
                change(self.receipt["metadata"])
                self.write()
                self.assert_rejected(self.accept)

    def test_every_invalid_scheduling_readback_is_rejected(self):
        mutations = {
            "absent": lambda metadata: metadata.update(scheduling=None),
            "platform": lambda metadata: metadata["scheduling"].update(platform="other"),
            "zero process": lambda metadata: metadata["scheduling"].update(process_id=0),
            "boolean process": lambda metadata: metadata["scheduling"].update(process_id=True),
            "string process": lambda metadata: metadata["scheduling"].update(process_id="4242"),
            "priority": lambda metadata: metadata["scheduling"].update(dict([PRIORITY])),
            "missing before": lambda metadata: metadata.update(scheduling_before_timing=None),
            "missing after": lambda metadata: metadata.update(scheduling_after_timing=None),
            "changed before": lambda metadata: metadata["scheduling_before_timing"].update(dict([PRIORITY])),
            "changed process": lambda metadata: metadata["scheduling_after_timing"].update(process_id=4243),
        }
        if PROCESSORS is not None:
            mutations["processors"] = lambda metadata: metadata["scheduling"].update(dict([PROCESSORS]))
            mutations["changed after"] = lambda metadata: metadata["scheduling_after_timing"].update(dict([PROCESSORS]))
        if PLATFORM == "windows":
            mutations["automatic power"] = lambda metadata: metadata["scheduling"].update(execution_speed_control_mask=0)
            mutations["throttled"] = lambda metadata: metadata["scheduling"].update(execution_speed_state_mask=1)
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                self.reset()
                mutate(self.receipt["metadata"])
                self.write()
                self.assert_rejected(self.accept)


INPUTS = {"fixture": "fixture.json", "checkpoint": "model", "cuda_onnx": "model-gpu-fp16-pooled.onnx",
          "trt_onnx": "model-pooled.onnx", "cpu_onnx": "model-pooled.onnx", "export_receipt": "export.json",
          "trt_cache": "cache"}
PATHS = {"python_executable": "python", "python_benchmark": "bench_ort_query.py", "torch_executable": "torch-python",
         "torch_benchmark": "bench_torch_query.py", "api_executable": "bench_query_api"}
SETTINGS = runner.LegSettings(settle=30.0, warm=1.0, gpu_threads=8, cpu_threads=4, trt_level=5, scheduling=REQUESTED)
PROGRAMS = {"ort": ("python", "bench_ort_query.py", bench_ort_query.argument_parser),
            "torch": ("torch-python", "bench_torch_query.py", bench_torch_query.argument_parser)}


class CommandTests(unittest.TestCase):
    """Each matched leg's command as the engine it starts reads it."""

    def command(self, engine):
        return runner.matched_command(engine, INPUTS, PATHS, SETTINGS, "leg.json")

    def test_each_python_engine_reads_its_command_as_the_runner_means_it(self):
        for engine in runner.ENGINES.values():
            if engine.kind == "rust":
                continue
            with self.subTest(engine=engine.name):
                program, (script, *options) = self.command(engine)
                interpreter, benchmark, parser = PROGRAMS[engine.kind]
                self.assertEqual((program, script), (interpreter, benchmark))
                parsed = parser().parse_args(options)
                self.assertEqual(parsed.provider if engine.kind == "ort" else parsed.device, engine.selector)
                self.assertEqual((parsed.fixture, parsed.model_dir, parsed.out, parsed.settle, parsed.warm),
                                 (Path("fixture.json"), Path("model"), Path("leg.json"), 30.0, 1.0))
                self.assertEqual(parsed.threads, 4 if engine.device == "cpu" else 8)
                self.assertEqual((parsed.cpus, parsed.priority), (REQUESTED.cpus, "high"))
                if engine.kind == "ort":
                    self.assertEqual((parsed.onnx, parsed.export_receipt),
                                     (Path(INPUTS[engine.graph]), Path("export.json")))
                    if engine.selector == "tensorrt":
                        self.assertEqual((parsed.cache, parsed.level), (Path("cache"), 5))
                    else:
                        self.assertIsNone(parsed.cache)

    def test_each_kodamaembed_command_selects_its_mode_and_cpu_threads(self):
        for engine in runner.ENGINES.values():
            if engine.kind != "rust":
                continue
            with self.subTest(engine=engine.name):
                program, arguments = self.command(engine)
                self.assertEqual(program, "bench_query_api")
                # The example takes matched options only after a leading --fixture.
                self.assertEqual(arguments[0], "--fixture")
                options = dict(zip(arguments[::2], arguments[1::2]))
                self.assertEqual(options["--mode"], engine.selector)
                self.assertEqual(options.get("--threads"), "4" if engine.device == "cpu" else None)


class SelectionTests(unittest.TestCase):
    def test_groups_and_names_expand_in_order_once_each(self):
        self.assertEqual(runner.selected_engines("gpu"), list(runner.GROUPS["gpu"]))
        self.assertEqual(runner.selected_engines("torch-cpu,cpu,torch-cpu")[:3],
                         ["torch-cpu", "ort-cpu", "kodamaembed-cpu-fp32"])
        with self.assertRaisesRegex(ValueError, "unknown engine: ort-gpu"):
            runner.selected_engines("ort-gpu")


class ExampleTests(unittest.TestCase):
    def test_the_example_cases_are_every_family_s_cases_with_their_fixtures(self):
        example = Path(runner.__file__).with_name("cases.example.json")
        cases = json.loads(example.read_text(encoding="utf-8"))["cases"]
        families = {f"{name}-b1-l{length}": name for name, family in runner.FAMILIES.items()
                    for length in family.lengths}
        self.assertEqual(sorted(case["name"] for case in cases), sorted(families))
        for case in cases:
            self.assertEqual(case["fixture"], f"fixtures/{families[case['name']]}/{case['name']}.json")


class PreparationTests(unittest.TestCase):
    """Runs that stop before any benchmark starts, with the reason on record."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        fixture = {"label": CASE, "family": "minilm", "role": "query", "texts": ["one query"]}
        (self.root / "fixture.json").write_text(json.dumps(fixture))
        (self.root / "cases.json").write_text(json.dumps({"schema_version": 1, "cases": [{
            "name": CASE, "checkpoint": "model", "fixture": "fixture.json", "cpu_onnx": "model.onnx",
            "export_receipt": "export.json"}]}))
        self.output = self.root / "rotation"

    def tearDown(self):
        self.directory.cleanup()

    def stopped(self, *options):
        """The failure log of a run that must stop during preparation."""
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(["--output", str(self.output), "--cases", str(self.root / "cases.json"), *options])
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual((code, summary["status"], summary["valid"], summary["legs"]), (1, "failed", False, []))
        self.assertEqual(summary["failure"]["stage"], "preparation")
        return (self.output / "failure.log").read_text()

    def test_quantized_engines_plan_e5_cases_only_in_the_selected_order(self):
        e5 = "e5-b1-l16"
        fixture = {"label": e5, "family": "e5", "role": "query", "texts": ["one query"]}
        (self.root / "e5.json").write_text(json.dumps(fixture))
        (self.root / "both.json").write_text(json.dumps({"schema_version": 1, "cases": [
            {"name": CASE, "checkpoint": "model", "fixture": "fixture.json"},
            {"name": e5, "checkpoint": "model", "fixture": "e5.json"}]}))
        api = ("--cpu-threads", "1", "--api-exe", sys.executable)
        comparison = runner.Comparison(runner.arguments([
            "--output", str(self.output), "--cases", str(self.root / "both.json"),
            "--engines", "kodamaembed-cpu-centered,kodamaembed-cpu-fp32", *api]))
        comparison.prepare_matched()
        plans = [(plan.case, plan.engine) for plan in comparison.plans]
        self.assertEqual(plans, [(CASE, "kodamaembed-cpu-fp32"), (e5, "kodamaembed-cpu-centered"),
                                 (e5, "kodamaembed-cpu-fp32")])
        # The receipt check rebuilds every case's legs from the recorded engine order.
        families = {CASE: "minilm", e5: "e5"}
        engines = comparison.settings_record()["engines"]
        self.assertEqual([(case, engine) for case, family in families.items() for engine in engines
                          if family in runner.ENGINES[engine].families], plans)
        log = self.stopped("--engines", "kodamaembed-cpu-centered", *api)
        self.assertIn("No selected engine runs these cases' models.", log)

    def test_a_rotation_takes_the_named_families_and_no_case_longer_than_asked(self):
        names = []
        for family, lengths in (("minilm", (16, 64)), ("modernbert-embed", (16, 64, 4096))):
            for length in lengths:
                names.append(f"{family}-b1-l{length}")
                fixture = {"label": names[-1], "family": family, "role": "query", "texts": ["one query"],
                           "expected_lengths": [length]}
                (self.root / f"{names[-1]}.json").write_text(json.dumps(fixture))
        (self.root / "all.json").write_text(json.dumps({"schema_version": 1, "cases": [
            {"name": name, "checkpoint": "model", "fixture": f"{name}.json"} for name in names]}))

        def planned(*options):
            comparison = runner.Comparison(runner.arguments([
                "--output", str(self.output), "--cases", str(self.root / "all.json"),
                "--engines", "kodamaembed-cpu-fp32", "--cpu-threads", "1", "--api-exe", sys.executable, *options]))
            comparison.prepare_matched()
            self.assertEqual([case["name"] for case in comparison.summary["cases"]],
                             [plan.case for plan in comparison.plans])
            return [plan.case for plan in comparison.plans]

        self.assertEqual(planned(), names)
        self.assertEqual(planned("--families", "modernbert-embed"), names[2:])
        self.assertEqual(planned("--families", "modernbert-embed,minilm", "--max-tokens", "64"), names[:4])
        self.assertEqual(planned("--max-tokens", "16"), [names[0], names[2]])
        with self.assertRaisesRegex(ValueError, "unknown family: bert"):
            planned("--families", "bert")
        with self.assertRaisesRegex(ValueError, "leave no case"):
            planned("--families", "e5")
        with self.assertRaisesRegex(ValueError, "leave no case"):
            planned("--max-tokens", "8")

    def test_a_token_limit_needs_each_fixture_to_state_its_lengths(self):
        log = self.stopped("--engines", "kodamaembed-cpu-fp32", "--cpu-threads", "1", "--api-exe", sys.executable,
                           "--max-tokens", "64")
        self.assertIn(f"Case {CASE}'s fixture states no lengths", log)

    def test_cpu_legs_need_a_thread_count(self):
        self.assertIn("CPU legs require --cpu-threads", self.stopped("--engines", "ort-cpu", "--python", sys.executable))

    def test_each_engine_needs_its_interpreter(self):
        self.assertIn("torch-cpu requires --torch-python", self.stopped("--engines", "torch-cpu", "--cpu-threads", "1"))

    def test_every_input_must_exist(self):
        log = self.stopped("--engines", "ort-cpu", "--cpu-threads", "1", "--python", sys.executable)
        self.assertIn(f"Missing required input: {CASE}/cpu_onnx", log)

    def test_the_output_must_be_new_or_empty(self):
        self.output.mkdir()
        (self.output / "earlier.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            runner.main(["--output", str(self.output), "--cases", str(self.root / "cases.json")])


if __name__ == "__main__":
    unittest.main(verbosity=2)

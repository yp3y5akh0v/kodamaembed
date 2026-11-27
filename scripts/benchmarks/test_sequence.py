"""The changing-lengths benchmark: its statistics and acceptance rules, then a whole rotation with its
receipt through the real ONNX Runtime and torch engines on the CPU, on a one-layer BERT of MiniLM's width."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import _sequence
import export_receipt
import receipts
import run_length_sequence as runner
import sequence_receipt
from _matched import ENCODERS
from run_query_comparison import LegSettings, Plan, matched_command
from test_rotation import CPUS, HERE, PRIORITY, WORDS, write_checkpoint
from test_runner import INPUTS, PATHS, PROGRAMS, REQUESTED

CASE = "minilm-sequence"
CALLS = 4
SETTINGS = LegSettings(settle=30.0, warm=None, gpu_threads=8, cpu_threads=4, trt_level=5, scheduling=REQUESTED)


def leg_receipt(metadata=None):
    """A leg receipt of three passes of CALLS one-millisecond calls."""
    intervals = [[1000.0 + 10 * index, 1000.0 + 10 * index + CALLS] for index in range(_sequence.PASSES)]
    return {"schema_version": 1, "status": "complete", "passes_ms": [[1.0] * CALLS for _ in intervals],
            "pass_intervals_unix_ms": intervals, "timing_start_unix_ms": 1000.0, "timing_end_unix_ms": 1030.0,
            "metadata": metadata}


class StatisticsTests(unittest.TestCase):
    def test_the_percentile_is_the_nearest_rank(self):
        self.assertEqual(_sequence.percentile(list(range(1, 101)), 0.99), 99)
        # Of 497 calls the fifth largest: a session's first calls lie above it.
        self.assertEqual(_sequence.percentile(list(range(1, 498)), 0.99), 493)
        self.assertEqual(_sequence.percentile([5.0], 0.99), 5.0)

    def test_the_first_pass_stands_apart_from_the_later_ones(self):
        statistics = _sequence.pass_statistics([[1.0, 3.0], [2.0, 2.0], [4.0, 8.0]])
        self.assertEqual(statistics["first_pass"], {"mean_ms": 2.0, "median_ms": 2.0, "p99_ms": 3.0, "max_ms": 3.0})
        self.assertEqual(statistics["later_passes"], {"mean_ms": 4.0, "median_ms": 3.0, "p99_ms": 8.0, "max_ms": 8.0})

    def test_bands_reach_a_quarter_a_half_and_the_whole_of_the_longest_input(self):
        self.assertEqual(_sequence.bands(range(16, 513), 512), [[16, 128], [129, 256], [257, 512]])
        passes = [[9.0] * 4, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0]]
        self.assertEqual(_sequence.band_medians(passes, [16, 128, 129, 512], 512), [
            {"tokens": [16, 128], "median_ms": 1.5}, {"tokens": [129, 256], "median_ms": 3.0},
            {"tokens": [257, 512], "median_ms": 5.0}])
        with self.assertRaisesRegex(ValueError, "no text of 129 to 256 tokens"):
            _sequence.band_medians(passes, [16, 128, 300, 512], 512)

    def test_a_tensorrt_profile_spans_the_shortest_the_middle_and_the_longest_length(self):
        shapes = _sequence.profile_shapes(list(range(16, 513)), ENCODERS["bert"].inputs)
        self.assertEqual(shapes, {kind: f"input_ids:1x{tokens},token_type_ids:1x{tokens},attention_mask:1x{tokens}"
                                  for kind, tokens in (("min", 16), ("opt", 264), ("max", 512))})
        shapes = _sequence.profile_shapes([16, 8192], ENCODERS["modernbert"].inputs)
        self.assertEqual(shapes["max"], "input_ids:1x8192,attention_mask:1x8192")


class TimingTests(unittest.TestCase):
    """What a leg's passes must hold before the runner reads its statistics."""

    def test_complete_passes_give_their_statistics(self):
        timing = runner.sequence_timing(leg_receipt(), CALLS)
        self.assertEqual(timing["first_pass"], {"mean_ms": 1.0, "median_ms": 1.0, "p99_ms": 1.0, "max_ms": 1.0})
        self.assertEqual(timing["later_passes"], timing["first_pass"])
        self.assertEqual((len(timing["passes_ms"]), len(timing["pass_intervals_unix_ms"])), (3, 3))

    def test_every_malformed_pass_is_rejected(self):
        for name, change in {
            "missing pass": lambda receipt: receipt["passes_ms"].pop(),
            "missing call": lambda receipt: receipt["passes_ms"][1].pop(),
            "nonpositive sample": lambda receipt: receipt["passes_ms"][0].__setitem__(0, 0.0),
            "boolean sample": lambda receipt: receipt["passes_ms"][0].__setitem__(0, True),
            "not finite": lambda receipt: receipt["passes_ms"][2].__setitem__(1, float("inf")),
            "missing interval": lambda receipt: receipt["pass_intervals_unix_ms"].pop(),
            "overlap": lambda receipt: receipt["pass_intervals_unix_ms"][1].__setitem__(0, 1003.0),
            "out of bounds": lambda receipt: receipt["pass_intervals_unix_ms"][2].__setitem__(1, 1031.0),
            "malformed pair": lambda receipt: receipt["pass_intervals_unix_ms"].__setitem__(0, [1000.0]),
            "numeric string": lambda receipt: receipt["pass_intervals_unix_ms"][0].__setitem__(0, "1000"),
            "missing end timestamp": lambda receipt: receipt.pop("timing_end_unix_ms"),
        }.items():
            with self.subTest(name=name):
                receipt = leg_receipt()
                change(receipt)
                with self.assertRaises(ValueError):
                    runner.sequence_timing(receipt, CALLS)
        with self.assertRaises(ValueError):
            runner.sequence_timing(leg_receipt(), CALLS + 1)
        with self.assertRaises(ValueError):
            runner.sequence_timing([], CALLS)


class CommandTests(unittest.TestCase):
    """Each leg's command as the engine it starts reads it."""

    def command(self, engine):
        return matched_command(engine, INPUTS, PATHS, SETTINGS, "leg.json", runner.Sequence.fixture_option)

    def test_each_python_engine_reads_a_sequence_and_no_warm_up(self):
        for engine in runner.ENGINES.values():
            if engine.kind == "rust":
                continue
            with self.subTest(engine=engine.name):
                program, (script, *options) = self.command(engine)
                interpreter, benchmark, parser = PROGRAMS[engine.kind]
                self.assertEqual((program, script), (interpreter, benchmark))
                self.assertNotIn("--warm", options)
                parsed = parser().parse_args(options)
                self.assertEqual((parsed.sequence, parsed.fixture, parsed.model_dir, parsed.out, parsed.settle),
                                 (Path("fixture.json"), None, Path("model"), Path("leg.json"), 30.0))
                self.assertEqual(parsed.provider if engine.kind == "ort" else parsed.device, engine.selector)
                if engine.selector == "tensorrt":
                    self.assertEqual((parsed.cache, parsed.level), (Path("cache"), 5))

    def test_each_kodamaembed_command_opens_with_the_sequence(self):
        for engine in runner.ENGINES.values():
            if engine.kind != "rust":
                continue
            with self.subTest(engine=engine.name):
                program, arguments = self.command(engine)
                # The example takes sequence options only after a leading --sequence.
                self.assertEqual((program, arguments[0]), ("bench_query_api", "--sequence"))
                options = dict(zip(arguments[::2], arguments[1::2]))
                self.assertEqual((options["--mode"], options.get("--warm")), (engine.selector, None))


class AcceptanceTests(unittest.TestCase):
    """The runner's checks of one KodamaEmbed leg's receipt and saved vectors."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.sequence = runner.Sequence(runner.arguments(["--output", str(self.root / "rotation"), "--cases", "unused"]))
        self.sequence.calls[CASE] = CALLS
        self.sequence.summary["hashes"] = {f"{CASE}/fixture": "a" * 64, f"{CASE}/config.json": "b" * 64,
                                           f"{CASE}/tokenizer.json": "c" * 64, f"{CASE}/model.safetensors": "d" * 64}

    def tearDown(self):
        self.directory.cleanup()

    def accept(self, engine, vectors=b"\x01\x02\x03\x04", **changes):
        """Accepts a leg of `engine` whose saved vectors are `vectors`, with `changes` to its metadata."""
        mode = runner.ENGINES[engine].selector
        (self.root / f"{engine}.vectors.f32").write_bytes(vectors)
        receipt = leg_receipt({
            "engine": "kodamaembed", "mode": mode, "scope": "end-to-end", "family": "e5", "role": "query",
            "fixture_sha256": "a" * 64, "passes": 3, "calls_per_pass": CALLS, "warm_calls": 0,
            "caller_retention": _sequence.RETENTION, "input_ids_types_sha256": "e" * 64,
            "checkpoint_sha256": {"config.json": "b" * 64, "tokenizer.json": "c" * 64, "model.safetensors": "d" * 64},
            **changes})
        receipt["vectors"] = {"path": f"{engine}.vectors.f32", "sha256": hashlib.sha256(vectors).hexdigest()}
        result = self.root / f"{engine}.json"
        result.write_text(json.dumps(receipt), encoding="utf-8")
        leg = {}
        self.sequence.accept(leg, Plan(CASE, engine, f"{CASE}/{engine}", {}, "e5", "query", True), result)
        return leg

    def test_a_leg_is_accepted_with_its_passes_and_artifacts(self):
        leg = self.accept("kodamaembed-cuda-centered")
        self.assertEqual(set(leg["artifacts"]), {"kodamaembed-cuda-centered.vectors.f32", "kodamaembed-cuda-centered.json"})
        self.assertEqual(leg["timing"]["later_passes"]["mean_ms"], 1.0)

    def test_centered_legs_must_save_the_same_vectors(self):
        self.accept("kodamaembed-cuda-centered")
        self.accept("kodamaembed-cpu-centered")
        with self.assertRaisesRegex(ValueError, "Centered vectors differ"):
            self.accept("kodamaembed-cpu-centered", vectors=b"\x01\x02\x03\x05")
        # The Float modes round differently and are held to no such rule.
        self.accept("kodamaembed-cuda-fp16", vectors=b"\x09\x09\x09\x09")

    def test_another_timed_boundary_is_rejected(self):
        for name, change in {"warmed session": {"warm_calls": 1}, "fewer passes": {"passes": 2},
                             "other call count": {"calls_per_pass": CALLS + 1},
                             "released outputs": {"caller_retention": "outputs released a call"},
                             "other mode": {"mode": "cuda-fp16"}, "other fixture": {"fixture_sha256": "f" * 64}}.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.accept("kodamaembed-cuda-centered", **change)


class RotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        checkpoint = cls.root / "model"
        checkpoint.mkdir()
        write_checkpoint(checkpoint)
        cls.run_script("export_bert.py", "--model-dir", checkpoint, "--out-dir", cls.root / "onnx", "--pooled")
        # Each word is one token, so a prefix of the document exists at every length.
        document = {"_id": "document", "text": " ".join(WORDS[index % len(WORDS)] for index in range(300))}
        (cls.root / "corpus.jsonl").write_text(json.dumps(document) + "\n", encoding="utf-8")
        cls.run_script("prepare_benchmark_fixtures.py", "--corpus", cls.root / "corpus.jsonl", "--model-dir", checkpoint,
                       "--family", "minilm", "--out-dir", cls.root / "fixtures", "--sequence")
        fixture = json.loads((cls.root / "fixtures" / f"{CASE}.json").read_text(encoding="utf-8"))
        cls.lengths = fixture["expected_lengths"]
        cls.complete = cls.rotate("complete", fixture)
        mismatched = copy.deepcopy(fixture)
        mismatched["expected_ids"][0][1] += 1
        cls.stopped = cls.rotate("stopped", mismatched)
        fixed = json.loads((cls.root / "fixtures" / "minilm-b1-l16.json").read_text(encoding="utf-8"))
        cls.unordered = cls.rotate("unordered", {**fixed, "label": CASE})
        cls.disclosures = cls.root / "disclosures.json"
        cls.disclosures.write_text(json.dumps({
            "limitations": ["A one-layer BERT with random weights measures the harness, not a model."],
            "stopped_rotations": [{"directory": "stopped/rotation",
                                   "reason": "The fixture's token IDs differ from the tokenizer's."}]}),
            encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    @staticmethod
    def run_script(name, *arguments):
        completed = subprocess.run([sys.executable, str(HERE / name), *map(str, arguments)], capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{name} failed:\n{completed.stderr}")

    @classmethod
    def rotate(cls, name, fixture):
        """Runs ort-cpu and torch-cpu over `fixture`; returns the exit code, output directory and summary."""
        work = cls.root / name
        work.mkdir()
        (work / "fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
        cases = work / "cases.json"
        cases.write_text(json.dumps({"schema_version": 1, "cases": [{
            "name": CASE, "checkpoint": str(cls.root / "model"), "fixture": "fixture.json",
            "cpu_onnx": str(cls.root / "onnx" / "model-pooled.onnx"),
            "export_receipt": str(cls.root / "onnx" / "export.json")}]}), encoding="utf-8")
        output = work / "rotation"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(["--output", str(output), "--cases", str(cases), "--engines", "ort-cpu,torch-cpu",
                                "--python", sys.executable, "--torch-python", sys.executable, "--cpu-threads", "1",
                                "--settle", "0", "--priority", PRIORITY, *CPUS])
        return code, output, json.loads((output / "summary.json").read_text(encoding="utf-8"))

    def receipt(self):
        """A valid receipt of the complete rotation alone."""
        rotation = sequence_receipt.build_rotation(self.complete[1], self.disclosures)
        receipt = export_receipt.receipt_of([rotation], runner.SCOPE)
        sequence_receipt.validate(receipt)
        return receipt

    def test_the_fixture_holds_every_length_in_the_published_order(self):
        _, _, summary = self.complete
        [case] = summary["cases"]
        self.assertEqual(self.lengths, list(range(16, 257)))
        self.assertEqual(case["order"], [(83 * index + 27) % 241 for index in range(241)])
        self.assertTrue(all(left != right for left, right in zip(case["order"], case["order"][1:])))

    def test_both_engines_complete_a_rotation_with_the_same_vectors(self):
        code, output, summary = self.complete
        self.assertEqual(code, 0, (output / "failure.log").read_text() if code else "")
        self.assertEqual((summary["status"], summary["valid"]), ("complete", True))
        self.assertEqual(summary["final_hashes"], summary["hashes"])
        orders = [[leg["engine"] for leg in summary["legs"] if leg["round"] == number] for number in (1, 2, 3)]
        self.assertEqual(orders, [["ort-cpu", "torch-cpu"], ["torch-cpu", "ort-cpu"], ["ort-cpu", "torch-cpu"]])
        for leg in summary["legs"]:
            self.assertEqual((leg["status"], leg["exit_code"], leg["gate"], leg["telemetry"]), ("accepted", 0, [], None))
            self.assertEqual([len(samples) for samples in leg["timing"]["passes_ms"]], [len(self.lengths)] * 3)
            self.assertEqual(leg["timing"]["metadata"]["warm_calls"], 0)
        self.assertEqual([row["valid_runs"] for row in summary["aggregates"].values()], [3, 3])
        ort, torch = (np.load(output / f"01-{CASE}-{engine}.outputs.npy") for engine in ("ort-cpu", "torch-cpu"))
        self.assertEqual(ort.shape, (len(self.lengths), 384))
        np.testing.assert_allclose(ort, torch, rtol=0, atol=1e-5)

    def test_a_torch_leg_names_its_encoder_in_what_it_timed(self):
        _, _, summary = self.complete
        [scope] = {leg["timing"]["metadata"]["scope_description"] for leg in summary["legs"]
                   if leg["engine"] == "torch-cpu"}
        self.assertEqual(scope, "fresh text tokenization and prefix, eager BERT SDPA forward/FP32 "
                                "pooling/normalization, pooled FP32 vector, one text a call")

    def test_a_benchmark_that_fails_stops_the_rotation_with_its_evidence(self):
        code, output, summary = self.stopped
        self.assertEqual((code, summary["status"], summary["valid"]), (1, "failed", False))
        self.assertEqual(summary["failure"]["stage"], f"01-{CASE}-ort-cpu benchmark")
        [leg] = summary["legs"]
        self.assertEqual((leg["engine"], leg["status"], leg["exit_code"]), ("ort-cpu", "failed", 1))
        self.assertIn("fixture IDs or types differ", (output / leg["stderr"]).read_text(encoding="utf-8"))

    def test_a_fixture_without_an_order_stops_before_any_benchmark(self):
        code, output, summary = self.unordered
        self.assertEqual((code, summary["failure"]["stage"], summary["legs"]), (1, "preparation", []))
        self.assertIn("needs a sequence fixture", (output / "failure.log").read_text(encoding="utf-8"))

    def test_the_rotation_exports_a_receipt_without_local_identifiers(self):
        receipt = self.receipt()
        [rotation] = receipt["rotations"]
        for leg in rotation["legs"]:
            self.assertNotIn("python_executable", leg["metadata"])
            for name in receipts.PYTHON_SCHEDULING:
                self.assertNotIn("process_id", leg["metadata"][name])
        self.assertNotIn(json.dumps(str(self.root))[1:-1], json.dumps(receipt))
        self.assertEqual(rotation["counts"], {"cases": 1, "accepted_legs": 6, "timed_calls": 6 * 3 * len(self.lengths)})
        self.assertEqual(rotation["export_policy"]["raw_artifact_hashes_verified"],
                         sum(len(leg["artifacts"]) for leg in self.complete[2]["legs"]))
        [stopped] = rotation["excluded_rotations"]
        self.assertEqual((stopped["accepted_legs"], stopped["stopped_at"]["stage"], stopped["stopped_at"]["engine"]),
                         (0, "benchmark", "ort-cpu"))
        [row] = rotation["aggregates"]
        self.assertEqual((row["device"], row["speedups"]), ("cpu", {}))
        for engine in ("ort-cpu", "torch-cpu"):
            statistics = row["engines"][engine]
            self.assertEqual([band["tokens"] for band in statistics["bands"]], [[16, 64], [65, 128], [129, 256]])
            self.assertEqual(len(statistics["later_pass_means_ms"]), 3)
            self.assertLessEqual(statistics["later_passes"]["median_ms"], statistics["later_passes"]["p99_ms"])
        with self.assertRaisesRegex(ValueError, "the release sequence differs"):
            sequence_receipt.validate_release(receipt)

    def test_the_command_line_export_writes_appends_and_checks_a_receipt(self):
        out = self.root / "published" / "benchmarks" / "receipt.json"
        arguments = ["--rotation", str(self.complete[1]), "--disclosures", str(self.disclosures), "--out", str(out),
                     "--partial"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sequence_receipt.main(arguments), 0)
            self.assertEqual(sequence_receipt.main([*arguments, "--check"]), 0)
            with self.assertRaisesRegex(ValueError, "--append adds this rotation"):
                sequence_receipt.main(arguments)
            with self.assertRaisesRegex(ValueError, "held twice"):
                sequence_receipt.main([*arguments, "--append"])
            written = out.read_bytes()
            out.write_bytes(written.replace(b'"warm_calls": 0', b'"warm_calls": 1', 1))
            with self.assertRaisesRegex(ValueError, "does not hold this rotation"):
                sequence_receipt.main([*arguments, "--check"])
            self.assertNotEqual(out.read_bytes(), written)

    def test_every_receipt_corruption_is_rejected(self):
        base = self.receipt()

        def only(receipt):
            return receipt["rotations"][0]

        def source_drift(receipt):
            receipt["source"]["files"][0]["sha256"] = "0" * 64
            receipt["source"]["files_sha256"] = receipts.inventory_digest(receipt["source"]["files"])

        def repository_drift(receipt):
            for name in ("input_hashes", "final_input_hashes"):
                only(receipt)[name]["sequence_runner"] = "0" * 64

        def first(receipt):
            return only(receipt)["legs"][0]

        def slower_call(receipt):
            first(receipt)["passes_ms"][1][0] += 1.0

        def order(receipt):
            return only(receipt)["cases"][0]["order"]

        mutations = {
            "no rotation": lambda receipt: receipt["rotations"].clear(),
            "a rotation held twice": lambda receipt: receipt["rotations"].append(copy.deepcopy(only(receipt))),
            "missing leg": lambda receipt: only(receipt)["legs"].pop(),
            "reordered legs": lambda receipt: only(receipt)["legs"].reverse(),
            "unknown leg field": lambda receipt: first(receipt).update(note="x"),
            "unknown metadata field": lambda receipt: first(receipt)["metadata"].update(note="x"),
            "unknown settings field": lambda receipt: only(receipt)["settings"].update(note="x"),
            "warm-up setting": lambda receipt: only(receipt)["settings"].update(warm_seconds=1.0),
            "fewer passes": lambda receipt: only(receipt)["settings"].update(passes=2),
            "process identifier": lambda receipt: first(receipt)["metadata"]["scheduling"].update(process_id=1),
            "machine path": lambda receipt: only(receipt)["limitations"].append("measured in /srv/benchmarks/rotation"),
            "changed input": lambda receipt: only(receipt)["final_input_hashes"].update(runner="0" * 64),
            "repository drift": repository_drift,
            "source drift": source_drift,
            "repeated text": lambda receipt: order(receipt).__setitem__(0, order(receipt)[1]),
            "another order": lambda receipt: order(receipt).reverse(),
            "edited sample": slower_call,
            "wrong statistic": lambda receipt: first(receipt)["later_passes"].update(mean_ms=0.001),
            "nonfinite sample": lambda receipt: first(receipt)["passes_ms"][0].__setitem__(0, float("nan")),
            "missing pass": lambda receipt: first(receipt)["passes_ms"].pop(),
            "missing call": lambda receipt: first(receipt)["passes_ms"][2].pop(),
            "pass outside its interval": lambda receipt: first(receipt)["pass_intervals_unix_ms"][0].__setitem__(
                1, first(receipt)["pass_intervals_unix_ms"][0][0] + 1e-6),
            "warmed session": lambda receipt: first(receipt)["metadata"].update(warm_calls=20),
            "wrong engine label": lambda receipt: first(receipt)["metadata"].update(engine="cuda"),
            "foreign interpreter": lambda receipt: first(receipt)["metadata"].update(python_executable_sha256="0" * 64),
            "foreign harness": lambda receipt: first(receipt)["metadata"].update(harness_sha256="0" * 64),
            "changed output artifact": lambda receipt: first(receipt)["artifacts"].update(
                {first(receipt)["metadata"]["outputs"]: "0" * 64}),
            "telemetry on a CPU leg": lambda receipt: first(receipt).update(telemetry={}),
            "wrong aggregate": lambda receipt: only(receipt)["aggregates"][0]["engines"]["ort-cpu"][
                "later_passes"].update(mean_ms=1.0),
            "wrong band": lambda receipt: only(receipt)["aggregates"][0]["engines"]["ort-cpu"]["bands"][0].update(
                tokens=[1, 64]),
            "wrong count": lambda receipt: only(receipt)["counts"].update(timed_calls=1),
            "stopped rotation marked valid": lambda receipt: only(receipt)["excluded_rotations"][0].update(
                recorded_valid=True),
            "raw artifact count": lambda receipt: only(receipt)["export_policy"].update(raw_artifact_hashes_verified=0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt = copy.deepcopy(base)
                mutate(receipt)
                with self.assertRaises(ValueError):
                    sequence_receipt.validate(receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)

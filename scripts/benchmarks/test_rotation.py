"""Whole rotations through the real ONNX Runtime and torch engines on the CPU, on a one-layer BERT of MiniLM's
width and a two-layer ModernBERT; KodamaEmbed's legs need its release build and the real checkpoints."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import export_receipt
import receipts
import run_query_comparison as runner
from _process import PLATFORM

HERE = Path(__file__).resolve().parent
TEXT = "Where did the hikers go with their kids last summer?"
WORDS = ("where", "did", "the", "hikers", "go", "with", "their", "kids", "last", "summer", "?")
CPUS = [] if PLATFORM == "macos" else ["--cpus", str(min(os.sched_getaffinity(0)) if PLATFORM == "linux" else 0)]
# High priority needs no privilege only on Windows.
PRIORITY = "high" if PLATFORM == "windows" else "inherit"


def write_tokenizer(directory):
    """A WordPiece tokenizer covering TEXT; returns it with its vocabulary's size."""
    from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
    from tokenizers.models import WordPiece

    vocabulary = {token: index for index, token in enumerate(("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS))}
    tokenizer = Tokenizer(WordPiece(vocabulary, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])])
    tokenizer.save(str(directory / "tokenizer.json"))
    return tokenizer, len(vocabulary)


def write_checkpoint(directory):
    """A one-layer BERT of MiniLM's width with random weights and its tokenizer, which is returned."""
    import torch
    from transformers import BertConfig, BertModel

    tokenizer, vocabulary = write_tokenizer(directory)
    torch.manual_seed(0)
    config = BertConfig(vocab_size=vocabulary, hidden_size=384, num_hidden_layers=1, num_attention_heads=12,
                        intermediate_size=1536)
    BertModel(config, add_pooling_layer=False).save_pretrained(directory, safe_serialization=True)
    return tokenizer


def write_modernbert(directory):
    """A ModernBERT of modernbert-embed-base's width with random weights and its tokenizer, which is
    returned: one layer sees the whole text, the other a window of four tokens."""
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    tokenizer, vocabulary = write_tokenizer(directory)
    torch.manual_seed(0)
    config = ModernBertConfig(vocab_size=vocabulary, hidden_size=768, num_hidden_layers=2, num_attention_heads=12,
                              intermediate_size=1152, global_attn_every_n_layers=2, local_attention=4,
                              pad_token_id=0, bos_token_id=2, cls_token_id=2, eos_token_id=3, sep_token_id=3)
    ModernBertModel(config).save_pretrained(directory, safe_serialization=True)
    return tokenizer


def exported(root, write):
    """Writes a checkpoint to `root`/model and its graphs to `root`/onnx; returns `write`'s tokenizer."""
    checkpoint = root / "model"
    checkpoint.mkdir()
    tokenizer = write(checkpoint)
    export = subprocess.run([sys.executable, str(HERE / "export_bert.py"), "--model-dir", str(checkpoint),
                             "--out-dir", str(root / "onnx"), "--pooled"], capture_output=True, text=True)
    if export.returncode != 0:
        raise RuntimeError(f"export_bert.py failed:\n{export.stderr}")
    return tokenizer


def query_fixture(tokenizer, family, prefix=""):
    """TEXT as one query of `family`, with the tokens `tokenizer` gives it behind `prefix`."""
    encoding = tokenizer.encode(prefix + TEXT)
    return {"label": f"{family}-b1-l{len(encoding.ids)}", "family": family, "role": "query", "texts": [TEXT],
            "expected_lengths": [len(encoding.ids)], "expected_ids": [encoding.ids],
            "expected_types": [encoding.type_ids]}


def rotate(root, name, fixture, engines="ort-cpu,torch-cpu"):
    """Runs `engines` over `fixture` on `root`'s checkpoint and graphs; returns the exit code, output
    directory and summary."""
    work = root / name
    work.mkdir()
    (work / "fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
    cases = work / "cases.json"
    cases.write_text(json.dumps({"schema_version": 1, "cases": [{
        "name": fixture["label"], "checkpoint": str(root / "model"), "fixture": "fixture.json",
        "cpu_onnx": str(root / "onnx" / "model-pooled.onnx"),
        "export_receipt": str(root / "onnx" / "export.json")}]}), encoding="utf-8")
    output = work / "rotation"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        code = runner.main(["--output", str(output), "--cases", str(cases), "--engines", engines,
                            "--python", sys.executable, "--torch-python", sys.executable, "--cpu-threads", "1",
                            "--settle", "0", "--warm", "0", "--priority", PRIORITY, *CPUS])
    return code, output, json.loads((output / "summary.json").read_text(encoding="utf-8"))


def torch_scope(summary):
    """What a rotation's torch legs say they timed."""
    [scope] = {leg["timing"]["metadata"]["scope_description"] for leg in summary["legs"] if leg["engine"] == "torch-cpu"}
    return scope


def fp16_conversion(root):
    """How `root`'s export receipt says its FP16 graph was converted."""
    return json.loads((root / "onnx" / "export.json").read_text(encoding="utf-8"))["fp16"]


class RotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        fixture = query_fixture(exported(cls.root, write_checkpoint), "minilm")
        cls.case = fixture["label"]
        cls.complete = rotate(cls.root, "complete", fixture)
        cls.later = rotate(cls.root, "later", fixture, "torch-cpu")
        mismatched = copy.deepcopy(fixture)
        mismatched["expected_ids"][0][1] += 1
        cls.stopped = rotate(cls.root, "stopped", mismatched)
        cls.disclosures = cls.root / "disclosures.json"
        cls.disclosures.write_text(json.dumps({
            "limitations": ["A one-layer BERT with random weights measures the harness, not a model."],
            "stopped_rotations": [{"directory": "stopped/rotation",
                                   "reason": "The fixture's token IDs differ from the tokenizer's."}]}),
            encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def receipt(self):
        """A valid receipt of the complete rotation alone."""
        rotation = export_receipt.build_rotation(self.complete[1], self.disclosures)
        receipt = export_receipt.receipt_of([rotation], receipts.SCOPE)
        receipts.validate(receipt)
        return receipt

    def test_both_engines_complete_a_rotation_with_the_same_vectors(self):
        code, output, summary = self.complete
        self.assertEqual(code, 0, (output / "failure.log").read_text() if code else "")
        self.assertEqual((summary["status"], summary["valid"]), ("complete", True))
        self.assertEqual(summary["final_hashes"], summary["hashes"])
        orders = [[leg["engine"] for leg in summary["legs"] if leg["round"] == number] for number in (1, 2, 3)]
        self.assertEqual(orders, [["ort-cpu", "torch-cpu"], ["torch-cpu", "ort-cpu"], ["ort-cpu", "torch-cpu"]])
        for leg in summary["legs"]:
            self.assertEqual((leg["status"], leg["exit_code"], leg["gate"], leg["telemetry"]), ("accepted", 0, [], None))
            self.assertIsNotNone(leg["process_scheduling"])
        self.assertEqual([row["valid_runs"] for row in summary["aggregates"].values()], [3, 3])
        ort, torch = (np.load(output / f"01-{self.case}-{engine}.outputs.npy") for engine in ("ort-cpu", "torch-cpu"))
        np.testing.assert_allclose(ort, torch, rtol=0, atol=1e-5)

    def test_a_torch_leg_names_its_encoder_in_what_it_timed(self):
        self.assertEqual(torch_scope(self.complete[2]),
                         "fresh text tokenization, prefix, padding, BERT SDPA forward/FP32 pooling/normalization, "
                         "pooled FP32 vector")

    def test_a_graph_without_rotary_positions_converts_to_fp16_whole(self):
        self.assertEqual(fp16_conversion(self.root), {"model_type": "bert", "opt_level": 0, "keep_io_types": True})

    def test_a_benchmark_that_fails_stops_the_rotation_with_its_evidence(self):
        code, output, summary = self.stopped
        self.assertEqual((code, summary["status"], summary["valid"]), (1, "failed", False))
        self.assertEqual(summary["failure"]["stage"], f"01-{self.case}-ort-cpu benchmark")
        [leg] = summary["legs"]
        self.assertEqual((leg["engine"], leg["status"], leg["exit_code"]), ("ort-cpu", "failed", 1))
        self.assertIn("fixture expected_ids differ", (output / leg["stderr"]).read_text(encoding="utf-8"))

    def test_the_rotation_exports_a_receipt_without_local_identifiers(self):
        receipt = self.receipt()
        [rotation] = receipt["rotations"]
        for leg in rotation["legs"]:
            self.assertNotIn("python_executable", leg["metadata"])
            for name in receipts.PYTHON_SCHEDULING:
                self.assertNotIn("process_id", leg["metadata"][name])
        self.assertNotIn(json.dumps(str(self.root))[1:-1], json.dumps(receipt))
        self.assertEqual(rotation["counts"]["accepted_legs"], 6)
        self.assertEqual(rotation["export_policy"]["raw_artifact_hashes_verified"],
                         sum(len(leg["artifacts"]) for leg in self.complete[2]["legs"]))
        [stopped] = rotation["excluded_rotations"]
        self.assertEqual((stopped["accepted_legs"], stopped["stopped_at"]["stage"], stopped["stopped_at"]["engine"]),
                         (0, "benchmark", "ort-cpu"))
        [row] = rotation["aggregates"]
        self.assertEqual((row["device"], row["speedups"]), ("cpu", {}))
        with self.assertRaisesRegex(ValueError, "release cases differ"):
            receipts.validate_release(receipt)

    def test_a_later_rotation_joins_the_receipt_of_an_earlier_one(self):
        earlier = export_receipt.build_rotation(self.complete[1], self.disclosures)
        later = export_receipt.build_rotation(self.later[1], self.disclosures)
        receipt = export_receipt.receipt_of([earlier, later], receipts.SCOPE)
        self.assertEqual([len(rows) for rows in receipts.validate(receipt)], [1, 1])
        self.assertEqual([rotation["settings"]["engines"] for rotation in receipt["rotations"]],
                         [["ort-cpu", "torch-cpu"], ["torch-cpu"]])
        with self.assertRaisesRegex(ValueError, "two rotations measured torch-cpu"):
            receipts.measured(receipt)
        receipt["rotations"][0]["settings"]["engines"].remove("torch-cpu")
        _, engines = receipts.measured(receipt)[self.case]
        self.assertEqual(engines, {"ort-cpu", "torch-cpu"})
        receipt["rotations"][1]["cases"][0]["lengths"] = [1]
        with self.assertRaisesRegex(ValueError, "describe the case differently"):
            receipts.measured(receipt)

    def test_the_command_line_export_writes_appends_and_checks_a_receipt(self):
        out = self.root / "published" / "benchmarks" / "receipt.json"
        complete = ["--disclosures", str(self.disclosures), "--out", str(out), "--partial",
                    "--rotation", str(self.complete[1])]
        later = [*complete[:-1], str(self.later[1])]
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                export_receipt.main([*later, "--append"])
            self.assertEqual(export_receipt.main(complete), 0)
            with self.assertRaisesRegex(ValueError, "--append adds this rotation"):
                export_receipt.main(later)
            with self.assertRaisesRegex(ValueError, "held twice"):
                export_receipt.main([*complete, "--append"])
            with self.assertRaisesRegex(ValueError, "does not hold this rotation"):
                export_receipt.main([*later, "--check"])
            self.assertEqual(export_receipt.main([*later, "--append"]), 0)
            for arguments in (complete, later):
                self.assertEqual(export_receipt.main([*arguments, "--check"]), 0)
            written = out.read_bytes()
            self.assertEqual(len(json.loads(written)["rotations"]), 2)
            out.write_bytes(written.replace(b'"warm_calls": 20', b'"warm_calls": 21', 1))
            with self.assertRaisesRegex(ValueError, "does not hold this rotation"):
                export_receipt.main([*complete, "--check"])
            self.assertNotEqual(out.read_bytes(), written)

    def test_every_receipt_corruption_is_rejected(self):
        base = self.receipt()

        def only(receipt):
            return receipt["rotations"][0]

        def rehash_source(receipt):
            receipt["source"]["files"].sort(key=lambda item: item["file"])
            receipt["source"]["files_sha256"] = receipts.inventory_digest(receipt["source"]["files"])

        def source_drift(receipt):
            receipt["source"]["files"][0]["sha256"] = "0" * 64
            rehash_source(receipt)

        def source_outside_inventory(receipt):
            receipt["source"]["files"].append({"file": "README.md",
                                               "sha256": receipts.file_sha256(receipts.ROOT / "README.md")})
            rehash_source(receipt)

        def repository_drift(receipt):
            for name in ("input_hashes", "final_input_hashes"):
                only(receipt)[name]["runner"] = "0" * 64

        def first(receipt):
            return only(receipt)["legs"][0]

        mutations = {
            "no rotation": lambda receipt: receipt["rotations"].clear(),
            "a rotation held twice": lambda receipt: receipt["rotations"].append(copy.deepcopy(only(receipt))),
            "a rotation's field beside the rotations": lambda receipt: receipt.update(legs=only(receipt)["legs"]),
            "another protocol's receipt": lambda receipt: receipt.update(scope="another protocol"),
            "missing leg": lambda receipt: only(receipt)["legs"].pop(),
            "duplicate leg": lambda receipt: only(receipt)["legs"].__setitem__(1, copy.deepcopy(first(receipt))),
            "reordered legs": lambda receipt: only(receipt)["legs"].reverse(),
            "unknown leg field": lambda receipt: first(receipt).update(note="x"),
            "unknown metadata field": lambda receipt: first(receipt)["metadata"].update(note="x"),
            "unknown settings field": lambda receipt: only(receipt)["settings"].update(note="x"),
            "process identifier": lambda receipt: first(receipt)["metadata"]["scheduling"].update(process_id=1),
            "machine path": lambda receipt: only(receipt)["limitations"].append("measured in /srv/benchmarks/rotation"),
            "empty limitation": lambda receipt: only(receipt)["limitations"].append(" "),
            "changed input": lambda receipt: only(receipt)["final_input_hashes"].update(runner="0" * 64),
            "repository drift": repository_drift,
            "source drift": source_drift,
            "source outside inventory": source_outside_inventory,
            "wrong median": lambda receipt: first(receipt).update(median_ms=first(receipt)["median_ms"] + 1),
            "nonfinite sample": lambda receipt: first(receipt)["samples_ms"].__setitem__(0, float("nan")),
            "missing interval": lambda receipt: first(receipt)["sample_intervals_unix_ms"].pop(),
            "sample outside its interval": lambda receipt: first(receipt)["sample_intervals_unix_ms"][0].__setitem__(
                1, first(receipt)["sample_intervals_unix_ms"][0][0] + 1e-6),
            "wrong engine label": lambda receipt: first(receipt)["metadata"].update(engine="cuda"),
            "foreign interpreter": lambda receipt: first(receipt)["metadata"].update(python_executable_sha256="0" * 64),
            "changed output artifact": lambda receipt: first(receipt)["artifacts"].update(
                {first(receipt)["metadata"]["outputs"]: "0" * 64}),
            "foreign scheduling platform": lambda receipt: first(receipt)["metadata"]["scheduling"].update(
                platform="other"),
            "telemetry on a CPU leg": lambda receipt: first(receipt).update(telemetry={}),
            "gate on a CPU leg": lambda receipt: first(receipt).update(gate=[{}]),
            "wrong aggregate": lambda receipt: only(receipt)["aggregates"][0]["engines"]["ort-cpu"].update(
                median_of_medians_ms=1.0),
            "wrong count": lambda receipt: only(receipt)["counts"].update(samples=1),
            "stopped rotation marked valid": lambda receipt: only(receipt)["excluded_rotations"][0].update(
                recorded_valid=True),
            "raw artifact count": lambda receipt: only(receipt)["export_policy"].update(raw_artifact_hashes_verified=0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt = copy.deepcopy(base)
                mutate(receipt)
                with self.assertRaises(ValueError):
                    receipts.validate(receipt)


class ModernBertRotationTests(unittest.TestCase):
    """A rotation over the other encoder type, whose graph takes no segment IDs."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        fixture = query_fixture(exported(cls.root, write_modernbert), "modernbert-embed", "search_query: ")
        cls.case = fixture["label"]
        cls.complete = rotate(cls.root, "complete", fixture)
        cls.disclosures = cls.root / "disclosures.json"
        cls.disclosures.write_text(json.dumps({
            "limitations": ["A two-layer ModernBERT with random weights measures the harness, not a model."],
            "stopped_rotations": []}), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_both_engines_complete_a_rotation_with_the_same_vectors(self):
        code, output, summary = self.complete
        self.assertEqual(code, 0, (output / "failure.log").read_text() if code else "")
        self.assertEqual([leg["status"] for leg in summary["legs"]], ["accepted"] * 6)
        ort, torch = (np.load(output / f"01-{self.case}-{engine}.outputs.npy") for engine in ("ort-cpu", "torch-cpu"))
        self.assertEqual(ort.shape, (1, 768))
        np.testing.assert_allclose(ort, torch, rtol=0, atol=1e-5)

    def test_a_torch_leg_names_its_encoder_in_what_it_timed(self):
        self.assertEqual(torch_scope(self.complete[2]),
                         "fresh text tokenization, prefix, padding, ModernBERT SDPA forward/FP32 "
                         "pooling/normalization, pooled FP32 vector")

    def test_no_rotary_position_or_angle_of_the_fp16_graph_passes_through_fp16(self):
        import onnx

        graph = onnx.shape_inference.infer_shapes(onnx.load(str(self.root / "onnx" / "model-gpu-fp16.onnx"))).graph
        kinds = {value.name: value.type.tensor_type.elem_type for value in graph.value_info}
        kinds.update({tensor.name: tensor.data_type for tensor in graph.initializer})
        # The casts that hand a finished cosine or sine to the FP16 layers are the rotary values' only way out.
        rotary = [node for node in graph.node
                  if "rotary_emb" in node.name and not node.name.endswith("_cast_to_fp16_node")]
        angles = [node for node in rotary if node.op_type in ("Cos", "Sin")]
        self.assertTrue(angles)
        self.assertEqual({kinds[node.output[0]] for node in angles}, {onnx.TensorProto.FLOAT})
        for node in rotary:
            self.assertNotIn(onnx.TensorProto.FLOAT16, [kinds.get(name) for name in (*node.input, *node.output)],
                             node.name)
        self.assertEqual(fp16_conversion(self.root)["fp32_rotary_nodes"], len(rotary))
        self.assertEqual({tensor.data_type for tensor in graph.initializer if list(tensor.dims) == [768, 768]},
                         {onnx.TensorProto.FLOAT16})

    def test_the_receipt_holds_each_torch_leg_to_its_encoder(self):
        rotation = export_receipt.build_rotation(self.complete[1], self.disclosures)
        base = export_receipt.receipt_of([rotation], receipts.SCOPE)
        receipts.validate(base)

        def torch_leg(receipt):
            return next(leg for leg in receipt["rotations"][0]["legs"] if leg["engine"] == "torch-cpu")["metadata"]

        self.assertEqual((torch_leg(base)["attention_classes"], torch_leg(base)["reference_compile"]),
                         (["ModernBertAttention"], False))
        mutations = {
            "another encoder's attention": lambda metadata: metadata.update(attention_classes=["BertSdpaSelfAttention"]),
            "no encoder setting": lambda metadata: metadata.pop("reference_compile"),
            "a setting that is no choice": lambda metadata: metadata.update(reference_compile="auto"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt = copy.deepcopy(base)
                mutate(torch_leg(receipt))
                with self.assertRaises(ValueError):
                    receipts.validate(receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)

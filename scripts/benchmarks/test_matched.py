"""CPU checks for fixture identity, pooling and matched receipt rejection."""

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import _matched

from _process import current_scheduling
from _matched import (
    audit_profile, prepare, prepared_inputs, read_fixture, save_preflight_output,
    sha256, validate_expected_tokens, validate_receipt, verify_export_receipt,
)
from prepare_benchmark_fixtures import first_prefix, running_texts


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        ids = [101] + list(range(10, 10 + len(text.split()))) + [102]
        return SimpleNamespace(ids=ids, type_ids=[0] * len(ids))

    def encode_batch(self, texts, add_special_tokens=True):
        return [self.encode(text, add_special_tokens) for text in texts]


def pooled(root):
    """Writes a graph that returns its two-token, two-wide input to `root` and pools it; returns both paths."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["hidden"], ["last_hidden_state"])], "pool-test",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 2, 2]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 2])],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, 2, 2])])
    source, target = root / "source.onnx", root / "pooled.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), source)
    prepare(SimpleNamespace(onnx=source, out=target))
    return source, target


class MatchedTests(unittest.TestCase):
    def test_invalid_scheduling_is_rejected_before_native_calls(self):
        for cpus, priority in (((), "inherit"), ((2, 1), "inherit"), ((-1,), "inherit"), (None, "realtime")):
            with self.subTest(cpus=cpus, priority=priority), mock.patch.object(_matched, "configure_current") as native:
                with self.assertRaises(ValueError):
                    _matched.configure_scheduling(cpus, priority)
                native.assert_not_called()

    def test_scheduling_arguments_parse_processor_lists(self):
        parser = argparse.ArgumentParser()
        _matched.add_scheduling_arguments(parser)
        args = parser.parse_args(["--cpus", "0-3,8", "--priority", "high"])
        self.assertEqual((args.cpus, args.priority), ((0, 1, 2, 3, 8), "high"))
        self.assertEqual(parser.parse_args([]).priority, "inherit")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--cpus", "3-1"])

    def test_scheduling_snapshots_reject_any_drift(self):
        expected = current_scheduling()
        with mock.patch.object(_matched, "current_scheduling", return_value=dict(expected)):
            self.assertEqual(_matched.scheduling_snapshot(expected), expected)
        for key in expected:
            with self.subTest(key=key), mock.patch.object(
                    _matched, "current_scheduling", return_value={**expected, key: "changed"}):
                with self.assertRaises(RuntimeError):
                    _matched.scheduling_snapshot(expected)

    def test_padding_and_expected_tokens_detect_mismatch(self):
        fixture = {"family": "minilm", "role": "query", "texts": ["one two", "three"]}
        feed, lengths, digest = prepared_inputs(FakeTokenizer(), fixture)
        self.assertEqual(lengths, [4, 3])
        self.assertEqual(feed["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 1, 0]])
        self.assertEqual(len(digest), 64)
        fixture.update(expected_lengths=lengths, expected_ids=[[101, 10, 11, 102], [101, 10, 102]],
                       expected_types=[[0] * 4, [0] * 3])
        validate_expected_tokens(fixture, feed, lengths)
        fixture["expected_ids"][0][1] = 999
        with self.assertRaisesRegex(ValueError, "expected_ids"):
            validate_expected_tokens(fixture, feed, lengths)

    def test_each_family_puts_its_prefix_before_a_text_by_role(self):
        class Recorder(FakeTokenizer):
            def encode_batch(self, texts, add_special_tokens=True):
                self.texts = texts
                return super().encode_batch(texts, add_special_tokens)

        prefixed = {("minilm", "query"): "one", ("minilm", "passage"): "one",
                    ("e5", "query"): "query: one", ("e5", "passage"): "passage: one",
                    ("modernbert-embed", "query"): "search_query: one",
                    ("modernbert-embed", "passage"): "search_document: one"}
        for (family, role), text in prefixed.items():
            with self.subTest(family=family, role=role):
                tokenizer = Recorder()
                prepared_inputs(tokenizer, {"family": family, "role": role, "texts": ["one"]})
                self.assertEqual(tokenizer.texts, [text])

    def test_a_fixture_names_a_benchmarked_family_and_a_role(self):
        fixture = {"family": "modernbert-embed", "role": "query", "label": "sample", "texts": ["one"],
                   "expected_lengths": [1], "expected_ids": [[5]], "expected_types": [[0]]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for family in _matched.FAMILIES:
                path.write_text(json.dumps({**fixture, "family": family}))
                self.assertEqual(read_fixture(path)["family"], family)
            for change in ({"family": "bert"}, {"role": "document"}):
                path.write_text(json.dumps({**fixture, **change}))
                with self.assertRaisesRegex(ValueError, "family"):
                    read_fixture(path)

    def test_fixture_rejects_boolean_ids_and_missing_rows(self):
        fixture = {"family": "minilm", "role": "query", "label": "sample", "texts": ["one"],
                   "expected_lengths": [1], "expected_ids": [[True]], "expected_types": [[0]]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(fixture))
            with self.assertRaisesRegex(ValueError, "32-bit"):
                read_fixture(path)
            fixture["expected_ids"] = []
            path.write_text(json.dumps(fixture))
            with self.assertRaisesRegex(ValueError, "one expected token row"):
                read_fixture(path)

    def test_receipt_rejects_engine_scope_and_nonfinite_samples(self):
        receipt = {"samples_ms": [1.] * 51, "timing_start_unix_ms": 10., "timing_end_unix_ms": 20.,
                   "metadata": {"engine": "cuda", "fixture_sha256": "abc", "family": "minilm", "role": "query",
                                "scope": "end-to-end", "calls_per_sample": 3}}
        validate_receipt(receipt, "cuda", "abc", "minilm", "query")
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_receipt(receipt, "tensorrt", "abc", "minilm", "query")
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_receipt(receipt, "cuda", "other-fixture", "minilm", "query")
        with self.assertRaisesRegex(ValueError, "family or role"):
            validate_receipt(receipt, "cuda", "abc", "e5", "query")
        with self.assertRaisesRegex(ValueError, "family or role"):
            validate_receipt(receipt, "cuda", "abc", "minilm", "passage")
        other = copy.deepcopy(receipt)
        other["metadata"]["scope"] = "device-tokens"
        with self.assertRaisesRegex(ValueError, "boundary"):
            validate_receipt(other, "cuda", "abc", "minilm", "query")
        other = copy.deepcopy(receipt)
        other["samples_ms"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite positive"):
            validate_receipt(other, "cuda", "abc", "minilm", "query")

    def test_output_save_rejects_unnormalized_nan_and_wrong_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, values in enumerate(([[.5, 0.]], [[float("nan"), 0.]], [[1., 0., 0.]])):
                path = root / f"bad-{index}.npy"
                with self.assertRaises(RuntimeError):
                    save_preflight_output(path, np.array(values, np.float32), 1, 2)
                self.assertFalse(path.exists())
            saved = save_preflight_output(root / "good.npy", np.array([[1., 0.]], np.float32), 1, 2)
            self.assertEqual(saved["output_norm_min"], 1.)

    def test_export_receipt_must_bind_both_checkpoint_and_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (root / name).write_bytes(name.encode())
            inputs = {name: sha256(root / name) for name in ("config.json", "tokenizer.json", "model.safetensors")}
            path = root / "export.json"
            path.write_text(json.dumps({"input_files_sha256": inputs, "files_sha256": {"model.onnx": "graphhash"}}))
            self.assertTrue(verify_export_receipt(path, root, {"model.onnx": "graphhash"})["checkpoint_and_graph_hashes_match"])
            with self.assertRaisesRegex(ValueError, "graph artifact"):
                verify_export_receipt(path, root, {"model.onnx": "different"})
            (root / "model.safetensors").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                verify_export_receipt(path, root, {"model.onnx": "graphhash"})

    def test_cpu_profile_node_is_rejected_by_placement_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps([{"cat": "Node", "name": "projection", "args": {
                "provider": "CPUExecutionProvider", "op_name": "MatMul"}}]))
            report = audit_profile(path, {"CUDAExecutionProvider"})
            self.assertEqual(len(report["unexpected"]), 1)

    def test_prefix_selection_takes_the_first_document_with_an_exact_prefix(self):
        documents = [{"_id": "b", "text": "one"}, {"_id": "a", "text": "four five six"},
                     {"_id": "c", "text": "seven eight"}]
        document, text, encoded = first_prefix(documents, FakeTokenizer(), "", 4)
        self.assertEqual((document, text, len(encoded.ids)), ("a", "four five", 4))
        with self.assertRaisesRegex(ValueError, "no word prefix"):
            first_prefix(documents, FakeTokenizer(), "", 9)

    def test_a_running_text_continues_a_document_with_those_after_it(self):
        documents = [{"_id": "b", "text": "one"}, {"_id": "a", "text": "four five six"},
                     {"_id": "c", "text": "seven eight"}]
        self.assertEqual([item["text"] for item in running_texts(documents, 4)],
                         ["one four five six", "four five six seven", "seven eight"])
        # No single document holds five words; the first one's running text does.
        document, text, encoded = first_prefix(running_texts(documents, 7), FakeTokenizer(), "", 7)
        self.assertEqual((document, text, len(encoded.ids)), ("b", "one four five six seven", 7))
        with self.assertRaisesRegex(ValueError, "no word prefix"):
            first_prefix(running_texts(documents, 9), FakeTokenizer(), "", 9)

    def test_pooled_export_masks_padding_and_returns_normalized_rows(self):
        from onnx.reference import ReferenceEvaluator

        with tempfile.TemporaryDirectory() as directory:
            _, target = pooled(Path(directory))
            output = ReferenceEvaluator(str(target)).run(None, {
                "hidden": np.array([[[3., 4.], [999., 999.]]], np.float32),
                "attention_mask": np.array([[1, 0]], np.int64)})[0]
            np.testing.assert_allclose(output, [[.6, .8]], rtol=1e-6)
            self.assertEqual(output.dtype, np.float32)

    def test_a_pooled_graph_s_receipt_does_not_change_with_the_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = pooled(Path(directory))
            receipt = json.loads(target.with_suffix(".onnx.json").read_text(encoding="utf-8"))
            self.assertEqual(set(receipt), {"source_hashes", "pooled_sha256", "hidden", "opset", "pooling",
                                            "runtime_precision"})
            self.assertEqual((receipt["source_hashes"], receipt["pooled_sha256"], receipt["hidden"], receipt["opset"]),
                             ({"source.onnx": sha256(source)}, sha256(target), 2, 18))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""The quality subset's selection through its program, on a small dataset in BEIR's layout, and
which models and runs the quality comparison takes."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import prepare_quality_subset
import run_quality

SCRIPT = Path(prepare_quality_subset.__file__)
CORPUS = [{"_id": f"p{number}", "text": f"passage {number}"} for number in range(1, 5)]
QUERIES = [{"_id": f"q{number}", "text": f"query {number}"} for number in range(1, 4)]
JUDGMENTS = "query-id\tcorpus-id\tscore\nq1\tp1\t1\nq2\tp3\t1\nq2\tp1\t1\nq3\tp4\t1\n"


def lines(records):
    return "".join(json.dumps(record) + "\n" for record in records).encode("utf-8")


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.source = Path(self.directory.name) / "source"
        self.source.mkdir()
        (self.source / "corpus.jsonl").write_bytes(lines(CORPUS))
        (self.source / "queries.jsonl").write_bytes(lines(QUERIES))
        (self.source / "test.tsv").write_bytes(JUDGMENTS.encode("utf-8"))
        self.target = Path(self.directory.name) / "subset"

    def tearDown(self):
        self.directory.cleanup()

    def run_script(self, passages):
        return subprocess.run([sys.executable, str(SCRIPT), "--source", str(self.source), "--out", str(self.target),
                               "--passages", str(passages)], capture_output=True, text=True)

    def test_the_first_passages_keep_the_judgments_that_reach_them_and_their_queries(self):
        completed = self.run_script(2)
        self.assertEqual((completed.returncode, completed.stdout.strip()), (0, "2 passages, 2 queries, 2 judgments"))
        self.assertEqual((self.target / "corpus.jsonl").read_bytes(), lines(CORPUS[:2]))
        self.assertEqual((self.target / "queries.jsonl").read_bytes(), lines(QUERIES[:2]))
        self.assertEqual((self.target / "test.tsv").read_bytes(),
                         b"query-id\tcorpus-id\tscore\nq1\tp1\t1\nq2\tp1\t1\n")

    def test_the_whole_corpus_is_its_own_subset(self):
        self.assertEqual(self.run_script(4).returncode, 0)
        for name in ("corpus.jsonl", "queries.jsonl", "test.tsv"):
            self.assertEqual((self.target / name).read_bytes(), (self.source / name).read_bytes())

    def test_a_short_corpus_and_an_existing_target_are_refused(self):
        with self.assertRaisesRegex(ValueError, "only 4 passages"):
            prepare_quality_subset.subset(self.source, self.target, 5)
        self.assertFalse(self.target.exists())
        self.target.mkdir()
        completed = self.run_script(2)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--out must be a new directory", completed.stderr)
        self.assertEqual(list(self.target.iterdir()), [])


class ModelTests(unittest.TestCase):
    def test_the_runs_take_the_families_whose_checkpoints_are_given(self):
        with tempfile.TemporaryDirectory() as directory:
            data = ["--output", str(Path(directory) / "quality"), "--validator", sys.executable, "--scifact", "s",
                    "--nfcorpus", "n", "--fiqa", "f", "--locomo", "l"]

            def families(*checkpoints):
                return list(run_quality.Quality(run_quality.arguments([*data, *checkpoints])).models)

            self.assertEqual(families("--modernbert-embed", "nomic"), ["modernbert-embed"])
            self.assertEqual(families("--e5", "large", "--minilm", "small"), ["minilm", "e5"])
            with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                run_quality.arguments(data)
            self.assertIn("at least one checkpoint", errors.getvalue())

    def test_only_a_quantized_family_is_scored_in_centered_int8_with_identical_cuda_vectors(self):
        self.assertEqual(run_quality.SCORED, {"minilm": "cuda_float", "e5": "cpu_centered",
                                              "modernbert-embed": "cuda_float"})
        self.assertEqual(run_quality.IDENTICAL, {"minilm": (), "e5": ("cuda_centered",), "modernbert-embed": ()})


if __name__ == "__main__":
    unittest.main(verbosity=2)

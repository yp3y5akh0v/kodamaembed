"""The release's quality runs: each model's scored run against FP32 on the CPU under the
saved-retrieval gates, every input, run and vector file recorded in quality.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import traceback

from _matched import FAMILIES
from _process import Tree
from run_query_comparison import CHECKPOINT_FILES, EXECUTABLE_SUFFIX, REPOSITORY, executable, now_utc, sha256

# Each dataset and the validator benchmark that reads it. fiqa is the subset of BEIR FiQA that
# prepare_quality_subset.py writes, the one dataset no design or calibration step used.
DATASETS = {"scifact": "beir", "nfcorpus": "beir", "fiqa": "beir", "locomo": "locomo"}
# The release's subset: FiQA's first passages, which bound the FP32 reference run on a processor.
FIQA_PASSAGES = 10_000
# The validator reads these files of a BEIR dataset, all from one directory.
BENCHMARK_FILES = {"beir": ("corpus.jsonl", "queries.jsonl", "test.tsv")}
# Each run and the validator engine that performs it.
RUNS = {"reference": "kodamaembed-cpu", "cpu_centered": "kodamaembed-cpu-centered",
        "cuda_float": "kodamaembed", "cuda_centered": "kodamaembed-centered"}
CUDA_RUNS = ("cuda_float", "cuda_centered")
# Each family's run scored against the reference, and the runs whose vectors must equal its
# bytes: a quantized family is scored in centered INT8 on the CPU, which CUDA must reproduce.
SCORED = {name: "cpu_centered" if family.quantized else "cuda_float" for name, family in FAMILIES.items()}
IDENTICAL = {name: ("cuda_centered",) if family.quantized else () for name, family in FAMILIES.items()}
# KODV vector files open with magic, version, rows and dimension, little-endian.
KODV_HEADER = struct.Struct("<4sIQQ")


def vector_shape(path):
    """The rows and dimension of a saved KODV version 1 file."""
    with open(path, "rb") as source:
        magic, version, rows, dim = KODV_HEADER.unpack(source.read(KODV_HEADER.size))
    if magic != b"KODV" or version != 1:
        raise ValueError(f"not a KODV version 1 vector file: {path.name}")
    return rows, dim


def dataset_files(dataset, path):
    benchmark = DATASETS[dataset]
    return [path / name for name in BENCHMARK_FILES[benchmark]] if benchmark in BENCHMARK_FILES else [path]


def checkpoints(args):
    """Each family whose checkpoint directory `args` names, with that directory."""
    given = {name: getattr(args, name.replace("-", "_")) for name in FAMILIES}
    return {name: Path(directory).absolute() for name, directory in given.items() if directory is not None}


class Quality:
    def __init__(self, args):
        self.output = Path(args.output).absolute()
        if self.output.exists() and (not self.output.is_dir() or any(self.output.iterdir())):
            raise ValueError("Output must be a new or empty directory.")
        self.output.mkdir(parents=True, exist_ok=True)
        self.cuda = args.cuda
        self.max_run_seconds = args.max_run_minutes * 60
        self.validator = executable(args.validator or
                                    REPOSITORY / f"target/release/kodamaembed-validate{EXECUTABLE_SUFFIX}")
        self.models = checkpoints(args)
        self.datasets = {"scifact": Path(args.scifact).absolute(), "nfcorpus": Path(args.nfcorpus).absolute(),
                         "fiqa": Path(args.fiqa).absolute(), "locomo": Path(args.locomo).absolute()}
        self.stage = "preparation"
        self.manifest = {"schema_version": 1, "status": "running", "started_utc": now_utc(), "finished_utc": None,
                         "cuda": self.cuda, "validator_sha256": None, "models": {}, "datasets": {}, "cases": [],
                         "failure": None}

    def save(self):
        text = json.dumps(self.manifest, indent=2, allow_nan=False)
        (self.output / "quality.json").write_text(text + "\n", encoding="utf-8")

    def run(self):
        self.save()
        try:
            inputs = self.hash_inputs()
            self.manifest.update(inputs)
            for family in self.models:
                if SCORED[family] in CUDA_RUNS and not self.cuda:
                    print(f"Quality {family}: scored on CUDA, so it needs --cuda", flush=True)
                    continue
                for dataset in DATASETS:
                    self.manifest["cases"].append(self.run_case(family, dataset))
                    self.save()
            self.stage = "completion"
            if self.hash_inputs() != inputs:
                raise RuntimeError("An input changed during the quality runs.")
            self.manifest["status"] = "complete"
        except Exception as error:
            self.manifest["status"] = "failed"
            self.manifest["failure"] = {"stage": self.stage, "error_type": type(error).__qualname__,
                                        "details": "failure.log"}
            (self.output / "failure.log").write_text("".join(traceback.format_exception(error)), encoding="utf-8")
            print(f"Quality runs stopped at {self.stage}; see {self.output}.", file=sys.stderr, flush=True)
        self.manifest["finished_utc"] = now_utc()
        self.save()
        return 0 if self.manifest["status"] == "complete" else 1

    def hash_inputs(self):
        """The validator's, every model file's and every dataset file's digest; a missing input stops the runs."""
        return {
            "validator_sha256": sha256(self.validator),
            "models": {family: {name: sha256(directory / name) for name in CHECKPOINT_FILES}
                       for family, directory in self.models.items()},
            "datasets": {dataset: {"benchmark": DATASETS[dataset],
                                   "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                                             for path in dataset_files(dataset, location)]}
                         for dataset, location in self.datasets.items()},
        }

    def run_case(self, family, dataset):
        self.stage = f"{family} {dataset}"
        print(f"Quality {self.stage}", flush=True)
        directory = self.output / family / dataset
        directory.mkdir(parents=True)
        benchmark, data = DATASETS[dataset], self.datasets[dataset]
        engines = {}
        for run in ("reference", SCORED[family], *IDENTICAL[family]):
            if run in CUDA_RUNS and not self.cuda:
                continue
            engine = RUNS[run]
            self.run_validator(run, ["retrieval", "--engine", engine, "--benchmark", benchmark, "--data", data,
                                     "--model", self.models[family], "--family", family,
                                     "--out", directory / run], directory)
            scores = json.loads((directory / run / "scores.json").read_text(encoding="utf-8"))
            engines[run] = {"engine": engine, "device": scores["engine_report"]["device"],
                            "fingerprint": scores["engine_report"]["fingerprint"], "seconds": scores["seconds"]}
        report = directory / "report.json"
        self.run_validator("compare", ["compare-retrieval", "--benchmark", benchmark, "--data", data,
                                       "--reference", directory / "reference",
                                       "--candidate", directory / SCORED[family],
                                       "--out", report, "--require-quality"], directory)
        return {"family": family, "dataset": dataset, "engines": engines, "vectors": self.vectors(directory, engines),
                "report": report.relative_to(self.output).as_posix(), "report_sha256": sha256(report)}

    def run_validator(self, name, arguments, directory):
        tree = Tree(self.validator, arguments, directory=REPOSITORY, log_dir=directory, log_prefix=name)
        try:
            if not tree.wait(self.max_run_seconds):
                raise RuntimeError(f"The {name} run exceeded its time limit.")
            outcome = tree.finish()
        finally:
            if not tree.finished:
                tree.finish(stop=True)
        if outcome.exit_code != 0:
            raise RuntimeError(f"The {name} run exited with {outcome.exit_code}; see its retained logs.")

    @staticmethod
    def vectors(directory, engines):
        """Each saved vector file with its geometry and the digest every run gave it."""
        names = sorted(path.name for path in (directory / "reference").glob("*.kodv"))
        for run in engines:
            if sorted(path.name for path in (directory / run).glob("*.kodv")) != names:
                raise RuntimeError(f"The {run} run saved different vector files.")
        result = []
        for name in names:
            rows, dim = vector_shape(directory / "reference" / name)
            entry = {"name": name, "rows": rows, "dim": dim, "bytes": (directory / "reference" / name).stat().st_size}
            entry.update({f"{run}_sha256": sha256(directory / run / name) for run in engines})
            result.append(entry)
        return result


def minutes(text):
    value = int(text)
    if not 1 <= value <= 1440:
        raise argparse.ArgumentTypeError("minutes must be within 1-1440")
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="a new or empty directory")
    parser.add_argument("--validator", help="kodamaembed-validate, by default from target/release")
    for name, family in FAMILIES.items():
        parser.add_argument(f"--{name}", help=f"the {family.model} checkpoint directory, which adds that model's runs")
    parser.add_argument("--scifact", required=True, help="directory with corpus.jsonl, queries.jsonl and test.tsv")
    parser.add_argument("--nfcorpus", required=True, help="directory with corpus.jsonl, queries.jsonl and test.tsv")
    parser.add_argument("--fiqa", required=True,
                        help=f"directory with FiQA's first {FIQA_PASSAGES:,} passages, as prepare_quality_subset.py "
                             "writes them")
    parser.add_argument("--locomo", required=True, help="locomo10.json")
    parser.add_argument("--cuda", action="store_true",
                        help="also run on CUDA: float precision, which scores a family without quantized "
                             "precision, and centered INT8, whose vectors must equal the CPU's byte for byte")
    parser.add_argument("--max-run-minutes", type=minutes, default=240)
    args = parser.parse_args(argv)
    if not checkpoints(args):
        parser.error("the runs need at least one checkpoint: " + ", ".join(f"--{name}" for name in FAMILIES))
    return args


def main(argv=None):
    return Quality(arguments(argv)).run()


if __name__ == "__main__":
    sys.exit(main())

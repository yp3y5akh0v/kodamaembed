"""Latency when the input length changes every call: every engine takes one sequence of texts three
times on a fresh session, one owned and checked process at a time, in alternating rounds."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import run_query_comparison as rotation
from _sequence import PASSES, RETENTION, pass_statistics
from run_query_comparison import (HERE, REPOSITORY, Comparison, Engine, accept_lineage, finite_number,
                                  read_matched_result, shared_options)

SCOPE = "fresh text to normalized FP32 host vectors, one text a call, the length changing every call"
# A CUDA graph replays one shape, so torch takes a sequence eagerly, under its own name.
EAGER = Engine("torch-sdpa", "gpu", "torch-sdpa", "torch", "cuda")
ENGINES = {engine.name: engine for engine in (
    EAGER if engine.name == "torch-sdpa-graph" else engine for engine in rotation.ENGINES.values())}
GROUPS = {group: tuple(EAGER.name if name == "torch-sdpa-graph" else name for name in names)
          for group, names in rotation.GROUPS.items()}


def sequence_timing(data, calls):
    """One leg's passes with their intervals and statistics; raises unless the receipt holds
    PASSES passes of `calls` positive samples inside ordered intervals."""
    if not isinstance(data, dict):
        raise ValueError("Timing receipt must be a JSON object.")
    passes, intervals = data.get("passes_ms"), data.get("pass_intervals_unix_ms")
    start, end = data.get("timing_start_unix_ms"), data.get("timing_end_unix_ms")
    if (not isinstance(passes, list) or len(passes) != PASSES or not isinstance(intervals, list)
            or len(intervals) != PASSES or not finite_number(start) or not finite_number(end)):
        raise ValueError(f"Sequence timing requires {PASSES} passes, their intervals and finite timestamps.")
    previous = start
    for samples, interval in zip(passes, intervals):
        if (not isinstance(samples, list) or len(samples) != calls
                or not all(finite_number(sample) and sample > 0 for sample in samples)):
            raise ValueError("A pass must hold one finite, positive sample a call.")
        if (not isinstance(interval, list) or len(interval) != 2
                or not all(finite_number(value) for value in interval)
                or interval[0] < previous or interval[1] <= interval[0] or interval[1] > end):
            raise ValueError("Pass intervals overlap or exceed the timing bounds.")
        previous = interval[1]
    return {"passes_ms": passes, "pass_intervals_unix_ms": intervals, "timing_start_unix_ms": start,
            "timing_end_unix_ms": end, **pass_statistics(passes), "metadata": data.get("metadata")}


class Sequence(Comparison):
    table, groups, fixture_option = ENGINES, GROUPS, "--sequence"

    def __init__(self, args):
        super().__init__(args)
        self.calls = {}
        self.centered = {}

    def prepare_matched(self):
        super().prepare_matched()
        self.paths.update(sequence_runner=Path(__file__).resolve(), sequence_helper=HERE / "_sequence.py")
        if any(ENGINES[plan.engine].kind == "rust" for plan in self.plans):
            self.paths["api_sequence_source"] = REPOSITORY / "crates/kodamaembed/examples/support/length_sequence.rs"
        for case in self.summary["cases"]:
            fixture = json.loads(Path(self.paths[f"{case['name']}/fixture"]).read_text(encoding="utf-8"))
            order, lengths = fixture.get("order"), case["lengths"]
            if (not isinstance(lengths, list) or len(lengths) != case["sequences"] or not isinstance(order, list)
                    or not all(type(index) is int for index in order)
                    or sorted(order) != list(range(len(lengths)))):
                raise ValueError(f"Case {case['name']} needs a sequence fixture, whose order names every text once.")
            case["order"] = order
            self.calls[case["name"]] = len(order)

    def settings_record(self):
        record = super().settings_record()
        for name in ("tokens", "samples_per_run", "calls_per_sample", "warm_seconds"):
            del record[name]
        return {**record, "passes": PASSES, "scope": SCOPE}

    def accept(self, leg, plan, result):
        """Sets a leg's passes and artifacts from its receipt; raises unless they are acceptable
        and every centered leg of the case saved the same vectors."""
        engine, calls = ENGINES[plan.engine], self.calls[plan.case]
        leg["timing"], leg["artifacts"] = read_matched_result(result, engine, lambda data: sequence_timing(data, calls))
        metadata = leg["timing"]["metadata"]
        if (not isinstance(metadata, dict) or metadata.get("passes") != PASSES
                or metadata.get("calls_per_pass") != calls or metadata.get("warm_calls") != 0
                or metadata.get("caller_retention") != RETENTION):
            raise ValueError("Sequence receipt timed boundary differs.")
        accept_lineage(metadata, engine, plan, self.summary["hashes"], self.token_hashes, self.requested)
        if engine.selector.endswith("-centered"):
            vectors = leg["artifacts"][f"{Path(result).stem}.vectors.f32"]
            if self.centered.setdefault(plan.case, vectors) != vectors:
                raise ValueError("Centered vectors differ between legs.")

    @staticmethod
    def headline(timing):
        return f"later passes' mean {timing['later_passes']['mean_ms']:.6f} ms"

    def aggregates(self):
        result = {}
        for plan in self.plans:
            means = [leg["timing"]["later_passes"]["mean_ms"] for leg in self.legs
                     if leg["key"] == plan.key and leg["valid"]]
            if len(means) != self.args.rounds:
                raise ValueError("The rotation is incomplete.")
            result[plan.key] = {"valid_runs": len(means), "later_pass_means_ms": means,
                                "median_of_means_ms": statistics.median(means)}
        return result


def arguments(argv=None):
    parser = shared_options(__doc__, ENGINES)
    parser.add_argument("--cases", required=True,
                        help="cases JSON as run_query_comparison.py reads it, each fixture a sequence written by "
                             "prepare_benchmark_fixtures.py --sequence")
    # No warm-up: the first pass measures what a fresh session costs.
    parser.set_defaults(warm=None)
    return parser.parse_args(argv)


def main(argv=None):
    return Sequence(arguments(argv)).run()


if __name__ == "__main__":
    sys.exit(main())

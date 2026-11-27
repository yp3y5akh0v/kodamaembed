"""The published receipts against this checkout, without models or a GPU, and every
corruption of them rejected."""

import copy
import json
import unittest

import receipts
import sequence_receipt

RECEIPT = receipts.ROOT / "docs" / "benchmarks" / "release.json"


def first(receipt):
    """The rotation a receipt opens with."""
    return receipt["rotations"][0]


def gpu_legs(receipt):
    return [leg for rotation in receipt["rotations"] for leg in rotation["legs"] if leg["device"] == "gpu"]


def gpu_leg(receipt):
    return gpu_legs(receipt)[0]


def widest_telemetry(receipt):
    """The GPU leg whose telemetry spans the longest time; only its first and last
    readings lie further apart than the largest gap the protocol allows."""
    def span(leg):
        samples = leg["telemetry"]["samples"]
        return samples[-1]["unix_ms"] - samples[0]["unix_ms"]
    return max(gpu_legs(receipt), key=span)["telemetry"]


def held_twice(receipt):
    receipt["rotations"].append(copy.deepcopy(first(receipt)))


def never_ran(receipt, engine):
    """Every rotation as its export would be had it not run `engine`."""
    for rotation in receipt["rotations"]:
        if engine not in rotation["settings"]["engines"]:
            continue
        rotation["settings"]["engines"].remove(engine)
        rotation["legs"] = [leg for leg in rotation["legs"] if leg["engine"] != engine]
        rotation["background_process_observations"] = [
            check for check in rotation["background_process_observations"]
            if not check["stage"].endswith(f"-{engine} gate")]
        inputs = receipts.expected_inputs(rotation)
        for name in ("input_hashes", "final_input_hashes"):
            rotation[name] = {key: value for key, value in rotation[name].items() if key in inputs}
        rotation["export_policy"]["raw_artifact_hashes_verified"] = sum(
            len(leg["artifacts"]) for leg in rotation["legs"])
        rotation["aggregates"] = receipts.aggregate(rotation)
        rotation["counts"] = receipts.counts(rotation, rotation["aggregates"])


def without_cpu_rotations(receipt):
    """Drops the rotations that ran on the processor alone."""
    receipt["rotations"] = [rotation for rotation in receipt["rotations"]
                            if any(leg["device"] == "gpu" for leg in rotation["legs"])]


def quality_held_twice(receipt):
    """A rotation without quality runs takes another's, with their manifest's digest."""
    holder = next(rotation for rotation in receipt["rotations"] if rotation["quality"] is not None)
    other = next(rotation for rotation in receipt["rotations"] if rotation["quality"] is None)
    other["quality"] = holder["quality"]
    other["provenance"]["quality_manifest_sha256"] = holder["provenance"]["quality_manifest_sha256"]


class PublishedReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))

    def test_the_published_receipt_is_a_valid_release(self):
        receipts.validate_release(self.receipt)

    def test_a_gate_that_waited_through_failing_readings_is_accepted(self):
        receipt = copy.deepcopy(self.receipt)
        gate = gpu_leg(receipt)["gate"]
        gate.insert(0, dict(gate[0], unix_ms=gate[0]["unix_ms"] - first(receipt)["settings"]["gate_interval_ms"],
                            temperature_c=99.0, software_thermal_slowdown=True))
        receipts.validate_release(receipt)

    def test_every_release_corruption_is_rejected(self):
        def telemetry_sample(receipt, **change):
            gpu_leg(receipt)["telemetry"]["samples"][1].update(change)

        def first_quality_case(receipt, family=None):
            return next(case for rotation in receipt["rotations"] if rotation["quality"] is not None
                        for case in rotation["quality"]["cases"] if family in (None, case["family"]))

        def interval(receipt):
            return first(receipt)["settings"]["gate_interval_ms"]

        mutations = {
            "no rotation": lambda receipt: receipt["rotations"].clear(),
            "a rotation held twice": held_twice,
            "a rotation's field beside the rotations": lambda receipt: receipt.update(legs=first(receipt)["legs"]),
            "missing GPU engine": lambda receipt: first(receipt)["settings"]["engines"].remove("trt-fp16"),
            "GPU engine no rotation ran": lambda receipt: never_ran(receipt, "torch-sdpa-graph"),
            "CPU engine no rotation ran": lambda receipt: never_ran(receipt, "torch-cpu"),
            "missing CPU rotation": without_cpu_rotations,
            "missing release case": lambda receipt: first(receipt)["cases"].pop(),
            "thermal slowdown during timing": lambda receipt: telemetry_sample(receipt, software_thermal_slowdown=True),
            "low enforced power during timing": lambda receipt: telemetry_sample(receipt, enforced_power_w=1.0),
            "telemetry gap": lambda receipt: widest_telemetry(receipt)["samples"].__delitem__(slice(1, -1)),
            "gate after it opened": lambda receipt: gpu_leg(receipt)["gate"].append(dict(
                gpu_leg(receipt)["gate"][-1], unix_ms=gpu_leg(receipt)["gate"][-1]["unix_ms"] + interval(receipt))),
            "crowded gate": lambda receipt: gpu_leg(receipt)["gate"][1].update(
                unix_ms=gpu_leg(receipt)["gate"][0]["unix_ms"] + 1),
            "gate during timing": lambda receipt: gpu_leg(receipt)["gate"][-1].update(
                unix_ms=gpu_leg(receipt)["timing_start_unix_ms"]),
            "missing quality runs": lambda receipt: first(receipt).update(quality=None),
            "missing quality case": lambda receipt: first(receipt)["quality"]["cases"].pop(),
            "quality runs in two rotations": quality_held_twice,
            "CUDA vectors differ": lambda receipt: first_quality_case(receipt, "e5")["vectors"][0].update(
                cuda_centered_sha256="0" * 64),
            "quality on another checkpoint": lambda receipt: first(receipt)["quality"]["models"]["minilm"].update(
                {"model.safetensors": "0" * 64}),
            "changed quality threshold": lambda receipt: first_quality_case(receipt)["report"]["totals"][
                "quality_gates"]["ndcg_loss"].update(threshold=1.0),
            "edited quality metric": lambda receipt: first_quality_case(receipt)["report"]["totals"][
                "candidate"].update(ndcg_at_10=0.0),
            "invented task failure": lambda receipt: first(receipt)["quality"]["individual_failed_tasks"].append(
                {"family": "minilm", "dataset": "scifact", "task": "beir", "quality_gates": {}}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt = copy.deepcopy(self.receipt)
                mutate(receipt)
                with self.assertRaises(ValueError):
                    receipts.validate_release(receipt)


class PublishedSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(sequence_receipt.OUTPUT.read_text(encoding="utf-8"))

    def test_the_published_receipt_is_a_valid_release(self):
        sequence_receipt.validate_release(self.receipt)

    def test_every_release_corruption_is_rejected(self):
        def leg_of(receipt, engine):
            return next(leg for leg in first(receipt)["legs"] if leg["engine"] == engine)

        def options(receipt, engine, provider):
            return leg_of(receipt, engine)["metadata"]["provider_options"][provider]

        def other_vectors(receipt):
            artifacts = leg_of(receipt, "kodamaembed-cuda-centered")["artifacts"]
            artifacts[next(name for name in artifacts if name.endswith(".vectors.f32"))] = "0" * 64

        mutations = {
            "no rotation": lambda receipt: receipt["rotations"].clear(),
            "a rotation held twice": held_twice,
            "missing GPU engine": lambda receipt: first(receipt)["settings"]["engines"].remove("trt-fp16"),
            "missing CPU engine": lambda receipt: first(receipt)["settings"]["engines"].remove(
                "kodamaembed-cpu-centered"),
            "another order": lambda receipt: first(receipt)["cases"][0]["order"].reverse(),
            "CUDA vectors differ": other_vectors,
            "graph replay": lambda receipt: options(receipt, "ort-cuda", "CUDAExecutionProvider").update(
                enable_cuda_graph="1"),
            "another TensorRT profile": lambda receipt: options(receipt, "trt-fp16", "TensorrtExecutionProvider").update(
                trt_profile_opt_shapes=options(receipt, "trt-fp16", "TensorrtExecutionProvider")["trt_profile_max_shapes"]),
            "warmed session": lambda receipt: leg_of(receipt, "torch-sdpa")["metadata"].update(warm_calls=20),
            "thermal slowdown during timing": lambda receipt: gpu_leg(receipt)["telemetry"]["samples"][1].update(
                software_thermal_slowdown=True),
            "telemetry gap": lambda receipt: widest_telemetry(receipt)["samples"].__delitem__(slice(1, -1)),
            "edited sample": lambda receipt: gpu_leg(receipt)["passes_ms"][1].__setitem__(
                0, gpu_leg(receipt)["passes_ms"][1][0] + 1.0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt = copy.deepcopy(self.receipt)
                mutate(receipt)
                with self.assertRaises(ValueError):
                    sequence_receipt.validate_release(receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)

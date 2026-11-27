"""Exercise the ORT benchmark's retained-output checks without a GPU."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import _matched


def retained_output_validation():
    path = Path(_matched.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bench = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "bench")
    start = next(index for index, node in enumerate(bench.body)
                 if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(51)")
    end = next(index for index in range(start + 1, len(bench.body))
               if isinstance(bench.body[index], ast.With)
               and ast.unparse(bench.body[index].items[0].context_expr) == "output_path.open('xb')")
    # The benchmark's own statements remain the test subject when its checks change.
    statements = copy.deepcopy(bench.body[start:end])
    return compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec")


class RetainedOutputTests(unittest.TestCase):
    @staticmethod
    def validate(outputs):
        values = iter(outputs)
        namespace = vars(_matched).copy()
        namespace.update(
            np=np, time=SimpleNamespace(perf_counter=lambda: 1.0, time_ns=lambda: 1_000_000),
            run=lambda: next(values), samples=[], intervals=[], shape=(1, 2), hidden=2,
            scheduling=None, scheduling_snapshot=lambda expected: expected,
        )
        exec(retained_output_validation(), namespace)

    def test_all_normalized_retained_outputs_pass(self):
        self.validate([np.array([[1.0, 0.0]], dtype=np.float32) for _ in range(51 * 3)])

    def test_zero_middle_output_is_rejected_even_when_final_output_is_valid(self):
        outputs = [np.array([[1.0, 0.0]], dtype=np.float32) for _ in range(51 * 3)]
        outputs[1] = np.zeros((1, 2), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "L2 normalized"):
            self.validate(outputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)

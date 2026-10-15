"""Small numerical regressions without downloading data or running TFCE.

Run: .venv/bin/python -m unittest discover -s tests
"""
import ast
import copy
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import ttest_1samp
from nltools.algorithms import fdr
from dartbrains_tools.simulations import simulate_max_t

SOURCE = Path(__file__).resolve().parents[1] / "content/Thresholding_Group_Analyses.py"


def cell_containing(text, namespace):
    source = SOURCE.read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and text in ast.get_source_segment(source, node):
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
            return namespace["_"](**{arg.arg: namespace[arg.arg] for arg in node.args.args})
    raise AssertionError(f"Missing cell: {text}")


class ThresholdingNumerics(unittest.TestCase):
    def test_maximum_reuse_matches_voxelwise_thresholding(self):
        simulate = simulate_max_t
        maxima = simulate(n_simulations=12, n_voxels=25, n_subjects=8, seed=42)
        rng = np.random.default_rng(42)
        maps = [ttest_1samp(rng.standard_normal((8, 25)), 0, axis=0).statistic for _ in range(12)]
        # Preserve the original inclusive threshold across a range of cutoffs.
        for cutoff in [0, 2, 4, 100]:
            expected = np.mean([np.any(np.abs(values) >= cutoff) for values in maps])
            self.assertEqual(np.mean(maxima >= cutoff), expected)
        np.testing.assert_array_equal(maxima, simulate(12, 25, 8, 42))

    def test_bh_boundary_ties_and_zero_are_kept(self):
        class Map:
            rendered = []

            def __init__(self, values):
                self.data = np.array(values, dtype=float)

            def copy(self):
                return copy.deepcopy(self)

            def iplot(self):
                self.rendered.append(self.data.copy())

        class Markdown:
            @staticmethod
            def md(text):
                return text

        for name in ["con1_stats", "con1_v_con2_stats"]:
            for p, expected in [([.001, .02, .02, .9], [1, 2, 3, 0]), ([0, .9], [1, 0]), ([.5, .9], None)]:
                Map.rendered.clear()
                stats = {"p": Map(p), "t": Map(np.arange(1, len(p) + 1))}
                original = stats["t"].data.copy()
                cell_containing(f"fdr({name}['p'].data", {name: stats, "fdr": fdr, "mo": Markdown})
                if expected is None:
                    self.assertEqual(Map.rendered, [])
                else:
                    np.testing.assert_array_equal(Map.rendered[-1], expected)
                np.testing.assert_array_equal(stats["t"].data, original)


if __name__ == "__main__":
    unittest.main()

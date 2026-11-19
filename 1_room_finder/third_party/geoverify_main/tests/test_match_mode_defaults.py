"""同一處理階段的匹配模式預設須與基準產出相同（direct）。"""
import ast
import inspect
import unittest

import roma_geo
from roma_geo import MATCH_MODE_DEFAULT
from roma_match.matcher import RoMaMatcher
from roma_match.parallel_matcher import ParallelScorer
from pose_regions import estimate_relative_pose


def _cli_default(fn, flag="--mode"):
    src = inspect.getsource(fn)
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            for kw in node.keywords:
                if kw.arg == "default":
                    return ast.literal_eval(kw.value)
    return None


class TestMatchModeDefaults(unittest.TestCase):
    def test_constant_is_direct(self):
        self.assertEqual(MATCH_MODE_DEFAULT, "direct")

    def test_verify_pair_default(self):
        self.assertEqual(inspect.signature(roma_geo.verify_pair).parameters["mode"].default,
                         MATCH_MODE_DEFAULT)

    def test_cli_roma_geo(self):
        from roma_geo.__main__ import main
        self.assertEqual(_cli_default(main), MATCH_MODE_DEFAULT)

    def test_cli_estimate_relative_pose(self):
        self.assertEqual(_cli_default(estimate_relative_pose), MATCH_MODE_DEFAULT)

    def test_pipeline_match_pairs_default(self):
        from roma_geo.pipeline import match_pairs
        self.assertEqual(inspect.signature(match_pairs).parameters["mode"].default,
                         MATCH_MODE_DEFAULT)

    def test_batch_process_default(self):
        self.assertEqual(inspect.signature(RoMaMatcher.batch_process).parameters["mode"].default,
                         MATCH_MODE_DEFAULT)

    def test_parallel_scorer_default(self):
        self.assertEqual(inspect.signature(ParallelScorer.__init__).parameters["mode"].default,
                         MATCH_MODE_DEFAULT)


if __name__ == "__main__":
    unittest.main(verbosity=2)

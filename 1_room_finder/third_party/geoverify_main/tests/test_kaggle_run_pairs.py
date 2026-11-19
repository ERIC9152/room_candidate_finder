"""kaggle_run 完整鏈須先從連通矩陣挑對，不可退回舊 _summary 快照。"""
import importlib.util
import os
import unittest

from _support import ROOT  # noqa: F401


def _load_kaggle_run():
    path = os.path.join(ROOT, "environments", "kaggle_run.py")
    spec = importlib.util.spec_from_file_location("kaggle_run_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestKaggleFullChainPairSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_kaggle_run()

    def test_select_pairs_comes_before_batch_match(self):
        steps = self.mod.full_chain_steps(
            py="python", workers="auto", batch_limit=0,
            time_budget_h=8.0, regions_dir="/tmp/regions")
        titles = [t for t, _ in steps]
        self.assertIn("5 從連通矩陣挑配對", titles)
        self.assertIn("6 批次匹配（填滿快取）", titles)
        self.assertLess(titles.index("5 從連通矩陣挑配對"),
                        titles.index("6 批次匹配（填滿快取）"))

    def test_select_pairs_command_forces_matrix_source(self):
        steps = dict(self.mod.full_chain_steps(
            py="python", workers="1", batch_limit=0,
            time_budget_h=0, regions_dir="/tmp/regions"))
        cmd = steps["5 從連通矩陣挑配對"]
        self.assertIn("environments/select_pairs.py", cmd)
        self.assertIn("--force", cmd)

    def test_select_pairs_failure_aborts_the_chain(self):
        self.assertIn("5 從連通矩陣挑配對", self.mod.BLOCKING_STEPS)
        self.assertIn("6 批次匹配（填滿快取）", self.mod.BLOCKING_STEPS)


if __name__ == "__main__":
    unittest.main(verbosity=2)

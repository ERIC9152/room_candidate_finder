"""匯入本套件不得蓋住使用端自己的 config / pipeline，也不得改動搜尋路徑。"""
import json
import os
import sys
import unittest

from _support import SRC, temp_dir  # noqa: F401

import roma_geo


class TestImportDoesNotStealNames(unittest.TestCase):
    def test_user_config_and_pipeline_survive_with_roma_geo(self):
        with temp_dir() as d:
            open(os.path.join(d, "config.py"), "w", encoding="utf-8").write("marker = 'user-config'\n")
            open(os.path.join(d, "pipeline.py"), "w", encoding="utf-8").write("marker = 'user-pipeline'\n")
            old_path = list(sys.path)
            old_config = sys.modules.pop("config", None)
            old_pipeline = sys.modules.pop("pipeline", None)
            try:
                sys.path.insert(0, d)
                if SRC not in sys.path:
                    sys.path.insert(1, SRC)
                import config as cfg2
                import pipeline as pipe2
                self.assertEqual(cfg2.marker, "user-config")
                self.assertEqual(pipe2.marker, "user-pipeline")
                self.assertTrue(hasattr(roma_geo, "verify_pair"))
                self.assertTrue(hasattr(roma_geo, "Settings"))
                self.assertIs(sys.modules.get("roma_geo"), roma_geo)
            finally:
                sys.path[:] = old_path
                sys.modules.pop("config", None)
                sys.modules.pop("pipeline", None)
                if old_config is not None:
                    sys.modules["config"] = old_config
                if old_pipeline is not None:
                    sys.modules["pipeline"] = old_pipeline

    def test_import_does_not_prepend_project_root(self):
        root = os.path.abspath(os.path.join(os.path.dirname(roma_geo.__file__), "..", ".."))
        if sys.path:
            self.assertNotEqual(os.path.normcase(os.path.abspath(sys.path[0])),
                                os.path.normcase(root))

    def test_import_leaves_search_path_unchanged(self):
        """匯入本套件一個目錄都不加：src 也不加（連帶匯入的子模組亦然）。"""
        import subprocess
        import textwrap
        from _support import temp_dir
        code = textwrap.dedent("""
            import json, sys
            before = list(sys.path)
            import roma_geo
            from roma_geo import pipeline
            print(json.dumps([p for p in sys.path if p not in before]))
        """)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        with temp_dir() as d:
            r = subprocess.run([sys.executable, "-c", code], cwd=d, env=env,
                               capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]), [])

    def test_root_shim_leaves_user_config(self):
        """載入專案根目錄相容層之後，使用端的 config 仍是自己的。"""
        import subprocess
        import textwrap
        from _support import ROOT, temp_dir
        with temp_dir() as d:
            open(os.path.join(d, "config.py"), "w", encoding="utf-8").write(
                "marker = 'user-config'\n")
            code = textwrap.dedent("""
                import os, sys
                sys.path.insert(0, %r)
                sys.path.insert(1, %r)
                import roma_geo
                sys.modules.pop("config", None)
                import config
                print(getattr(config, "marker", "STOLEN"))
                print("path0", os.path.normcase(os.path.abspath(sys.path[0] or ".")))
            """ % (d, ROOT))
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            r = subprocess.run(
                [sys.executable, "-c", code],
                cwd=d, env=env, capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("user-config", r.stdout)
            self.assertNotIn("STOLEN", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

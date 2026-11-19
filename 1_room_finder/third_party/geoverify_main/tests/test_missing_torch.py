"""缺少 torch 時，對外進入點須指出缺了什麼，不得只丟底層 ModuleNotFoundError。"""
import os
import subprocess
import sys
import textwrap
import unittest

from _support import ROOT, temp_dir  # noqa: F401

_HOOK = textwrap.dedent(r"""
import builtins
_real = builtins.__import__
def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'")
    return _real(name, globals, locals, fromlist, level)
builtins.__import__ = _import
for k in list(sys.modules):
    if k == "torch" or k.startswith("torch."):
        sys.modules.pop(k, None)
""")


def _script(body):
    return "import sys\n" + _HOOK + "\n" + textwrap.dedent(body)


def _run_code(code, cwd=None):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", _script(code)],
        cwd=cwd or ROOT, env=env, capture_output=True, text=True,
    )


class TestVerifyPairMissingTorch(unittest.TestCase):
    def test_verify_pair_names_torch_and_source(self):
        r = _run_code("""
            import roma_geo
            try:
                roma_geo.verify_pair("a.jpg", "b.jpg")
            except ImportError as e:
                print("ERR", e)
            else:
                raise SystemExit("expected ImportError")
        """)
        self.assertIn("ERR", r.stdout, r.stdout + r.stderr)
        self.assertIn("torch", r.stdout)
        self.assertIn("requirements-core.txt", r.stdout)


class TestAllEntryPointsMissingTorch(unittest.TestCase):
    def test_five_entry_points_and_cli(self):
        r = _run_code(r"""
            import os, traceback
            import roma_geo
            from roma_geo.config import Settings
            from roma_geo.pipeline import match_pairs, run_house

            def show(name, fn):
                try:
                    fn()
                    print(name, "NO_ERROR")
                except ImportError as e:
                    print(name, "ERR", e)
                except Exception:
                    print(name, "OTHER", traceback.format_exc())

            show("verify_pair", lambda: roma_geo.verify_pair("a.jpg", "b.jpg"))
            show("verify_pairs", lambda: next(roma_geo.verify_pairs([("a.jpg", "b.jpg")])))
            show("get_matcher", lambda: roma_geo.get_matcher())
            s = Settings(datasets="/tmp", output="/tmp", cache="/tmp")
            show("match_pairs", lambda: match_pairs(s, "/no/such/pairs.csv"))
            show("run_house", lambda: run_house("H", s, run_match=True))
        """)
        out = r.stdout + r.stderr
        for name in ("verify_pair", "verify_pairs", "get_matcher", "match_pairs", "run_house"):
            self.assertIn(name + " ERR", out, out)
            block = out.split(name + " ERR", 1)[1].split("\n", 1)[0]
            self.assertIn("torch", block)
            self.assertIn("requirements-core.txt", block)

    def test_cli_missing_torch(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        with temp_dir() as d:
            hook = os.path.join(d, "sitecustomize.py")
            with open(hook, "w", encoding="utf-8") as f:
                f.write("import sys\n" + _HOOK)
            env["PYTHONPATH"] = d
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "roma_geo.py"), "a.jpg", "b.jpg"],
                cwd=ROOT, env=env, capture_output=True, text=True,
            )
        text = r.stdout + "\n" + r.stderr
        self.assertNotEqual(r.returncode, 0, text)
        self.assertIn("torch", text)
        self.assertIn("requirements-core.txt", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

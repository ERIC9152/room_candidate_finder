"""在專案根目錄、未改模組搜尋路徑時，import roma_geo 須得到對外 API。"""
import os
import subprocess
import sys
import unittest

from _support import ROOT, temp_dir  # noqa: F401

_API = ("verify_pair", "to_public", "DEFAULTS", "PUBLIC_FIELDS")

_CODE = r"""
import roma_geo
for name in %r:
    assert hasattr(roma_geo, name), name
print("file", roma_geo.__file__)
print("fn", roma_geo.verify_pair.__code__.co_filename)
""" % (_API,)


def _run(cwd):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", _CODE],
        cwd=cwd, env=env, capture_output=True, text=True,
    )


class TestImportFromProjectRoot(unittest.TestCase):
    def test_project_root_exposes_public_api(self):
        r = _run(ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + "\n" + r.stderr)

    def test_outside_dir_uses_same_implementation(self):
        at_root = _run(ROOT)
        self.assertEqual(at_root.returncode, 0, at_root.stdout + "\n" + at_root.stderr)
        with temp_dir() as d:
            outside = _run(d)
        self.assertEqual(outside.returncode, 0, outside.stdout + "\n" + outside.stderr)
        self.assertEqual(_fn_line(at_root.stdout), _fn_line(outside.stdout))

    def test_cli_entry_still_runs(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        script = os.path.join(ROOT, "roma_geo.py")
        r = subprocess.run(
            [sys.executable, script, "-h"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + "\n" + r.stderr)
        self.assertIn("img_a", r.stdout)


class TestMissingDependencyMessage(unittest.TestCase):
    def test_message_names_package_and_source(self):
        from roma_geo import format_dependency_error, require_modules
        msg = format_dependency_error(["numpy"], "做幾何驗證")
        self.assertIn("numpy", msg)
        self.assertIn("requirements-core.txt", msg)
        self.assertIn("setup_env", msg)
        with self.assertRaises(ImportError) as ctx:
            require_modules((("no_such_roma_geo_mod_xyz", "numpy"),), "做幾何驗證")
        self.assertIn("numpy", str(ctx.exception))
        self.assertIn("requirements-core.txt", str(ctx.exception))


def _fn_line(stdout):
    for line in stdout.splitlines():
        if line.startswith("fn "):
            return os.path.normcase(os.path.abspath(line[3:].strip()))
    return stdout


if __name__ == "__main__":
    unittest.main(verbosity=2)

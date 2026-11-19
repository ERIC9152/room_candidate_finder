"""Skip conditions for tests that need things the package cannot ship.

    ZInD data     ZIND_ROOT (default C:\\zind_data): the dataset may not be redistributed
    ZInD code     ZIND_CODE (default C:\\zind\\code): Zillow's own transformations.py
    ImageNet      torchvision's pretrained ResNet, downloaded on first use (needs internet)

Without them the tests are reported as skipped instead of failing, under pytest and when a test file
is run as a script.
"""
import functools
import importlib.util
import os
import sys
import urllib.error

ZIND_ROOT = os.environ.get("ZIND_ROOT", r"C:\zind_data")
ZIND_CODE = os.environ.get("ZIND_CODE", r"C:\zind\code")
if os.path.isdir(ZIND_CODE) and ZIND_CODE not in sys.path:
    sys.path.insert(0, ZIND_CODE)


def _skip(name, reason):
    print(f"  skip {name} ({reason})")
    if os.environ.get("PYTEST_CURRENT_TEST"):          # running under pytest: report it as skipped
        import pytest
        pytest.skip(reason)


def _requires(ok, reason):
    def deco(fn):
        if ok:
            return fn

        @functools.wraps(fn)
        def skipped(*args, **kwargs):
            _skip(fn.__name__, reason)
        return skipped
    return deco


needs_zind = _requires(os.path.isdir(os.path.join(ZIND_ROOT, "0000")),
                       f"ZInD data not found at {ZIND_ROOT}; set ZIND_ROOT")
needs_zind_code = _requires(importlib.util.find_spec("transformations") is not None,
                            f"ZInD code (transformations.py) not found at {ZIND_CODE}; set ZIND_CODE")


def needs_download(fn):
    """Skip when pretrained weights are not cached and cannot be downloaded (no internet)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except urllib.error.URLError as e:
            _skip(fn.__name__, f"pretrained weights could not be downloaded: {e.reason}")
    return wrapper

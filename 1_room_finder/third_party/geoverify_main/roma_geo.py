#!/usr/bin/env python3
"""命令列進入點，同時是專案根目錄下 `import roma_geo` 的相容層。

    python roma_geo.py a.jpg b.jpg
    from roma_geo import verify_pair
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")


def _src_first():
    if _SRC in sys.path:
        sys.path.remove(_SRC)
    sys.path.insert(0, _SRC)


def _load_package():
    """把 src/roma_geo 套件接到本模組名上，讓根目錄的 import 與安裝後相同。"""
    _src_first()
    stub = sys.modules.get("roma_geo")
    stub_file = os.path.abspath(getattr(stub, "__file__", "") or "")
    if stub is not None and stub_file == os.path.abspath(__file__):
        del sys.modules["roma_geo"]
        for key in list(sys.modules):
            if key.startswith("roma_geo."):
                del sys.modules[key]
    import roma_geo as pkg
    sys.modules["roma_geo"] = pkg
    return pkg


if __name__ == "__main__":
    _src_first()
    from roma_geo.__main__ import main
    raise SystemExit(main())

_pkg = _load_package()
globals().update({k: getattr(_pkg, k) for k in dir(_pkg) if not k.startswith("_")})
__all__ = list(getattr(_pkg, "__all__", []))
__doc__ = _pkg.__doc__
__file__ = _pkg.__file__
__path__ = getattr(_pkg, "__path__", [])
__package__ = "roma_geo"

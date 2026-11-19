"""相容層。新程式請用 from roma_geo.errors import RomaGeoError。"""
import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
from roma_geo.errors import RomaGeoError  # noqa: F401

"""相容層。新程式請用 from roma_geo.names import house_of。"""
import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
from roma_geo.names import house_of, image_stem  # noqa: F401

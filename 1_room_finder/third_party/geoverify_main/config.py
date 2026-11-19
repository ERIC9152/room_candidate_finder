"""相容層。新程式請用 from roma_geo.config import Settings。"""
import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
from roma_geo.config import *  # noqa: F401,F403
from roma_geo.config import Settings, PROJECT_ROOT, datasets_root, matrices_root  # noqa: F401
from roma_geo.config import regions_root, cache_root, semantic_root, pair_threshold  # noqa: F401
from roma_geo.config import describe  # noqa: F401

if __name__ == "__main__":
    print(describe())

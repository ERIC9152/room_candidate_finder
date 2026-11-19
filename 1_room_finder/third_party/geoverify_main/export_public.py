"""相容層與命令列：python export_public.py"""
import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
from roma_geo.export_public import *  # noqa: F401,F403
from roma_geo.export_public import main  # noqa: F401

if __name__ == "__main__":
    main()

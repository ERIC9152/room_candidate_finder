"""相容層與命令列：python ingest_connections.py"""
import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
from roma_geo.ingest_connections import *  # noqa: F401,F403
from roma_geo.ingest_connections import main  # noqa: F401

if __name__ == "__main__":
    main()

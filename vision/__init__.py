"""Vision side of the J5 project: capture, detect, map, send."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    # protocol.py lives at the repo root because the phone imports it too
    sys.path.insert(0, _ROOT)

__all__ = ["calibrate", "detect", "record", "run", "sender", "source"]
"""Use the vendored FM-SRM modules; loader methods retain Base behavior."""
import sys
from pathlib import Path

def activate():
    vendor = str(Path(__file__).resolve().parents[1] / 'vendor')
    if vendor not in sys.path: sys.path.insert(0, vendor)

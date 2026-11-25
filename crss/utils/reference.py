"""Activate the frozen loader and noise helpers bundled inside the CRSS package."""
import sys
from pathlib import Path


def activate():
    vendor = str(Path(__file__).resolve().parents[1] / '_vendor')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)

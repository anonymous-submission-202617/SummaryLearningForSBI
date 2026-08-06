"""Pytest fixtures: make the experiment packages and repo root importable."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "Gaussian"),
    os.path.join(REPO_ROOT, "LotkaVoletrra"),
    os.path.join(REPO_ROOT, "Radio"),
    os.path.join(REPO_ROOT, "SIR"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

"""Atomic writes for the per-prompt caches and the metric files. Standard library only."""
from __future__ import annotations

import os


def save(path: str, write, mode: str = "wb") -> None:
    """write(file) into path + '.tmp', then rename, so `path` is either complete or absent. The
    temporary name never matches the p*.npy / p*.npz cache patterns."""
    with open(path + ".tmp", mode) as f:
        write(f)
    os.replace(path + ".tmp", path)

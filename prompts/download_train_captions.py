#!/usr/bin/env python
"""Fetch the 100 ReDiverse-FT training captions into prompts/train_captions.txt.

The captions are the first 100 lines (lines 0-99) of data/diffusiondb.txt in the
DP-DMD repository (DiffusionDB prompts, CC0-1.0), pinned to the commit that added
the file. Both the downloaded file and the written caption file are verified by
SHA-256, so the result is byte-identical to the captions the released adapters
were trained on.

    python prompts/download_train_captions.py
    python prompts/download_train_captions.py --source /path/to/diffusiondb.txt   # local copy, no download
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

REPO = "Multimedia-Analytics-Laboratory/dpdmd"
COMMIT = "2e084a67341eded1d9b48cdce21276cf7047cfcc"
PATH = "data/diffusiondb.txt"
URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{PATH}"
SOURCE_SHA256 = "b544efca19051c0a05ba6373807f6c4ce5b93fa60b69dcec6a635a65caef80c7"   # full 200,000-line file
N_CAPTIONS = 100
OUTPUT_SHA256 = "8174de975f0e61e64c01354e4de66b4a8fe2d027807ae4749694b3095edce8fa"   # lines 0-99
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_captions.txt")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rediverse-download/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def first_lines(data: bytes, n: int) -> bytes:
    """The first n newline-terminated lines, bytes unchanged."""
    lines = data.split(b"\n")
    assert len(lines) > n, f"source has only {len(lines) - 1} lines"
    return b"\n".join(lines[:n]) + b"\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help="output file (default prompts/train_captions.txt)")
    ap.add_argument("--source", default=None,
                    help="read a local copy of DP-DMD's data/diffusiondb.txt instead of downloading it")
    a = ap.parse_args()

    if a.source:
        with open(a.source, "rb") as f:
            data = f.read()
        origin = a.source
    else:
        print(f"downloading {URL}", flush=True)
        data = fetch(URL)
        origin = URL
    got = sha256(data)
    if got != SOURCE_SHA256:
        sys.exit(f"{origin} has sha256 {got}, expected {SOURCE_SHA256}")

    out = first_lines(data, N_CAPTIONS)
    got = sha256(out)
    if got != OUTPUT_SHA256:
        sys.exit(f"the first {N_CAPTIONS} lines have sha256 {got}, expected {OUTPUT_SHA256}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, a.out)
    print(f"wrote {N_CAPTIONS} captions to {a.out} (sha256 {got})", flush=True)


if __name__ == "__main__":
    main()

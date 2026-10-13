"""Video files: the H.264 writer of the generators and the frame reader of the metrics.

Both go through the system ffmpeg (built with libx264) over a rawvideo pipe. Clips use
VBench's naming, "<prompt>-<k>.mp4" in a flat directory, with "/" in a prompt replaced by "_".
"""
from __future__ import annotations

import os
import subprocess

import numpy as np


def vbench_name(prompt: str, i: int = 0) -> str:
    """VBench reads '{prompt}-{index}.mp4'; only a literal '/' is replaced."""
    return f"{prompt.replace('/', '_')}-{i}.mp4"


def clip_path(videos_dir: str, prompt: str, i: int) -> str:
    return os.path.join(videos_dir, vbench_name(prompt, i))


def write_mp4(frames: np.ndarray, path: str, fps: int):
    """uint8 [T,H,W,3] -> H.264 (libx264, yuv420p, CRF 17) via an ffmpeg rawvideo pipe,
    written to a temporary name and renamed. Wan 2.1 clips use fps 16, MiniMax-H3 clips 24."""
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", path + ".tmp.mp4"]
    p = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    assert p.returncode == 0, p.stderr.decode()[:400]
    os.replace(path + ".tmp.mp4", path)


def probe(path: str):
    """(width, height, number of frames) of the first video stream."""
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames",
                        "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip().split(",")
    return int(o[0]), int(o[1]), int(o[2])


def read_frames(path: str, n_frames: int | None = None, size: tuple[int, int] | None = None, indices=None) -> np.ndarray:
    """Decode a clip with ffmpeg -> uint8 [T, H, W, 3]. `n_frames` samples that many frames uniformly
    (first and last included); `indices` picks explicit frame indices; `size` = (W, H) resize via ffmpeg."""
    w, h, T = probe(path)
    vf = []
    if size is not None:
        vf.append(f"scale={size[0]}:{size[1]}")
        w, h = size
    cmd = ["ffmpeg", "-v", "error", "-i", path] + (["-vf", ",".join(vf)] if vf else []) + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    if indices is not None:
        idx = [min(i, arr.shape[0] - 1) for i in indices]
        return arr[idx]
    if n_frames is not None and n_frames < arr.shape[0]:
        idx = np.linspace(0, arr.shape[0] - 1, n_frames).round().astype(int)
        return arr[idx]
    return arr

"""Camera-trajectory diversity with ViPE (main comparison table: Cam; per-aspect TPIPS table: CamEnd).
Runs in the vipe environment (nvidia-vipe 1.2.0 with the pose-only pipeline, see envs/vipe.md).

For every prompt of the subset (default prompts/vbench_subset30.txt) each of the 10 clips goes through
`vipe infer <clip> -p pose_only`: SLAM camera-to-world poses [T, 4, 4] with GeoCalib intrinsics, no instance
segmentation and no dense depth alignment. Each trajectory is expressed in its own first-camera frame,
    q_t = R_0^T (p_t - p_0),
its path length L is the sum of the step norms over all T frames, and it is resampled linearly to 32 points on
normalised time. Over the 45 clip pairs of a prompt:
    trajectory_distance = mean_pairs mean_t || q_i(t) - q_j(t) ||        endpoint_distance = mean_pairs || q_i(1) - q_j(1) ||
and the same on the length-normalised trajectories q / (L + 1e-6): trajectory_norm (Cam) and endpoint_norm
(CamEnd). Only the normalised statistics are reported: ViPE's scale is not metric across clips.

Output: <out>/camtraj/p<idx>.npz {endpoint_distance, trajectory_distance, endpoint_norm, trajectory_norm,
endpoint_pairs [10, 10], traj_pairs [10, 10], path_length [10], prompt}; ViPE's poses and intrinsics stay under
<out>/vipe/p<idx>/. Workers claim prompts through <out>/camtraj/.claims.

ViPE is called as $VIPE_BIN (default: the `vipe` script next to this Python interpreter, else `vipe` on PATH).
If $VIPE_CUDA_HOME is set, its bin/ and lib64/ are prepended to PATH / LD_LIBRARY_PATH of the ViPE process
(the CUDA 12.6 toolkit ViPE was built against).

  python -m evaluation.video.camtraj_vipe --videos-dir <set>/videos --out <set>/feats
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from evaluation.common.atomic import save
from evaluation.video.sets import groups, prompt_ids
from rediverse.claims import claim_tag, release_tag

T_RESAMPLE = 32
PIPELINE = "pose_only"


def vipe_bin():
    if os.environ.get("VIPE_BIN"):
        return os.environ["VIPE_BIN"]
    sibling = os.path.join(os.path.dirname(sys.executable), "vipe")
    return sibling if os.path.exists(sibling) else (shutil.which("vipe") or "vipe")


def vipe_env():
    env = dict(os.environ)
    cuda = os.environ.get("VIPE_CUDA_HOME")
    if cuda:
        env["PATH"] = os.path.join(cuda, "bin") + ":" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = os.path.join(cuda, "lib64") + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def link_name(clip):
    """ViPE names its outputs after the input file; clip names contain spaces and punctuation, so ViPE runs
    on a symlink with a sanitised name."""
    stem = os.path.splitext(os.path.basename(clip))[0]
    return stem.replace(" ", "_").replace(",", "").replace("'", "") + ".mp4"


def run_vipe(clip, out_dir, log, pipeline=PIPELINE):
    """vipe infer on one clip -> path of its pose npz (<out_dir>/pose/<sanitised stem>.npz)."""
    link = os.path.join(out_dir, "in", link_name(clip))
    pose = os.path.join(out_dir, "pose", os.path.splitext(os.path.basename(link))[0] + ".npz")
    if os.path.exists(pose):
        return pose
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.exists(link):
        os.symlink(os.path.abspath(clip), link)
    with open(log, "a") as lf:
        r = subprocess.run([vipe_bin(), "infer", link, "-p", pipeline, "-o", out_dir], stdout=lf,
                           stderr=subprocess.STDOUT, env=vipe_env())
    if r.returncode != 0 or not os.path.exists(pose):
        raise RuntimeError(f"vipe failed on {clip} (rc={r.returncode}); see {log}")
    for sub in ("depth", "mask", "rgb", "flow"):                     # keep only pose + intrinsics
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    return pose


def traj_from_pose(pose_npz):
    """Pose npz (data [T, 4, 4] camera-to-world) -> (trajectory in the first-camera frame resampled to
    [T_RESAMPLE, 3], path length over all T frames)."""
    d = np.load(pose_npz); P = d["data"].astype(np.float64)
    R0 = P[0, :3, :3]; p0 = P[0, :3, 3]
    q = (P[:, :3, 3] - p0) @ R0                                                  # rows: R0^T (p_t - p0)
    t = np.linspace(0, 1, q.shape[0]); ts = np.linspace(0, 1, T_RESAMPLE)
    qs = np.stack([np.interp(ts, t, q[:, k]) for k in range(3)], 1)
    return qs, float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def prompt_stats(pose_files):
    """Per-prompt trajectory statistics from the pose files of its clips (in clip order)."""
    trajs, lengths = [], []
    for f in pose_files:
        q, L = traj_from_pose(f); trajs.append(q); lengths.append(L)
    Q = np.stack(trajs)                                                          # [10, T_RESAMPLE, 3]
    m = Q.shape[0]
    end = np.linalg.norm(Q[:, None, -1] - Q[None, :, -1], axis=-1)              # [10, 10]
    trj = np.linalg.norm(Q[:, None] - Q[None, :], axis=-1).mean(-1)             # [10, 10]
    L = np.array(lengths); Qn = Q / (L[:, None, None] + 1e-6)
    endn = np.linalg.norm(Qn[:, None, -1] - Qn[None, :, -1], axis=-1)
    trjn = np.linalg.norm(Qn[:, None] - Qn[None, :], axis=-1).mean(-1)
    iu = np.triu_indices(m, 1)
    return dict(endpoint_distance=end[iu].mean(), trajectory_distance=trj[iu].mean(),
                endpoint_norm=endn[iu].mean(), trajectory_norm=trjn[iu].mean(),
                endpoint_pairs=end, traj_pairs=trj, path_length=L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--out", required=True, help="feature directory of the set (<set>/feats)")
    ap.add_argument("--prompt-ids", default="subset30", help="default: prompts/vbench_subset30.txt")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only")
    a = ap.parse_args()
    gs = groups(a.videos_dir, prompt_ids(a.prompt_ids))
    if a.limit:
        gs = gs[:a.limit]
    od = os.path.join(a.out, "camtraj"); os.makedirs(od, exist_ok=True)
    vroot = os.path.join(a.out, "vipe"); os.makedirs(vroot, exist_ok=True)
    claims = os.path.join(od, ".claims")
    pending = [g for g in gs if not os.path.exists(os.path.join(od, f"p{g[0]:03d}.npz"))]
    print(f"[camtraj] {len(pending)}/{len(gs)} prompts, pipeline {PIPELINE}", flush=True)
    t0, n = time.time(), 0
    for (k, prompt, paths) in pending:
        tag = f"p{k:03d}"
        if os.path.exists(os.path.join(od, tag + ".npz")) or not claim_tag(claims, tag):
            continue
        out_dir = os.path.join(vroot, tag); os.makedirs(out_dir, exist_ok=True)
        poses = [run_vipe(p, out_dir, os.path.join(out_dir, "vipe.log")) for p in paths]
        s = prompt_stats(poses)
        save(os.path.join(od, tag + ".npz"), lambda fh: np.savez(fh, **s, prompt=prompt))
        release_tag(claims, tag)
        n += 1; el = time.time() - t0
        print(f"[camtraj] {n}/{len(pending)} {tag}  traj_norm {s['trajectory_norm']:.4f}  end_norm {s['endpoint_norm']:.4f}  "
              f"{el / n / 60:.1f} min/prompt  eta {(len(pending) - n) * el / n / 3600:.1f} h", flush=True)
    print(f"[camtraj] done {n} prompts in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

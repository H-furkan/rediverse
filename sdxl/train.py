#!/usr/bin/env python
"""Train the SDXL ReDiverse-FT adapter with FastGen (paper Sec. 4.1, App. D).

Runs FastGen's train.py (third_party/FastGen, prepared by sdxl/setup_fastgen.sh) with
fastgen/configs/experiments/SDXL/config_rediverse_ft.py, then extracts the adapter from the final
checkpoint. One H100 80 GB: about 28 s per step, 200 steps, peak memory about 75 GB.

Inputs (sdxl/README.md): data/sdxl/train_ctx/p000.pt..p099.pt (sdxl/encode_captions.py) and
data/sdxl/dmd2_sdxl_init.pth (sdxl/build_init_ckpt.py). FastGen loads the SDXL-base UNet with
cache_dir=$HF_HOME, so HF_HOME must be set.

  python -m sdxl.train                                   # -> outputs/sdxl/train/
  python -m sdxl.train --out-dir run2 -- trainer.seed=1  # extra FastGen overrides after --

Output in --out-dir: config.yaml (FastGen), config.json (recipe), train.log,
checkpoints/<iteration:07d>.pth (FastGen) and adapters/rediverse_ft_<iteration:07d>.pth, which
`python -m sdxl.convert_weights` turns into the safetensors that sdxl.generate loads.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from rediverse import data as D
from sdxl import config as C

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FASTGEN_CONFIG = "fastgen/configs/experiments/SDXL/config_rediverse_ft.py"


def fastgen_location(run_dir: str):
    """FASTGEN_OUTPUT_ROOT and the log_config overrides for which FastGen's run directory,
    <root>/<project>/<group>/<name>, is `run_dir`."""
    group_dir, name = os.path.split(os.path.abspath(run_dir))
    project_dir, group = os.path.split(group_dir)
    root, project = os.path.split(project_dir)
    assert name and group and project, f"--out-dir needs three path components: {run_dir}"
    return root, [f"log_config.project={project}", f"log_config.group={group}", f"log_config.name={name}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help=f"run directory (default {C.train_dir()})")
    ap.add_argument("--steps", type=int, default=D.STEPS, help="optimizer steps (checkpoint at the last one)")
    ap.add_argument("--fastgen", default=os.path.join(REPO_ROOT, "third_party", "FastGen"))
    ap.add_argument("--no-extract", action="store_true", help="keep only the FastGen checkpoint")
    ap.add_argument("overrides", nargs="*", help="FastGen config overrides (key=value), after --")
    a = ap.parse_args()

    fastgen = os.path.abspath(a.fastgen)
    if not os.path.isfile(os.path.join(fastgen, FASTGEN_CONFIG)):
        sys.exit(f"{fastgen}/{FASTGEN_CONFIG} not found; run: bash sdxl/setup_fastgen.sh")
    if "HF_HOME" not in os.environ:
        sys.exit("FastGen reads the SDXL-base UNet from cache_dir=$HF_HOME; export HF_HOME first "
                 "(e.g. export HF_HOME=~/.cache/huggingface)")
    data_root = os.path.abspath(D.data_root())
    run_dir = os.path.abspath(a.out_dir or C.train_dir())
    for sub in ("checkpoints", "adapters"):
        d = os.path.join(run_dir, sub)
        if os.path.isdir(d) and os.listdir(d):
            sys.exit(f"{d} is not empty; the data sampler does not resume. Remove {run_dir} or choose --out-dir.")
    os.makedirs(run_dir, exist_ok=True)
    fastgen_root, location = fastgen_location(run_dir)

    D.write_run_meta(os.path.join(run_dir, "config.json"), D.run_meta(C.BACKBONE, dict(
        collapsed_block=C.ANCHOR_MODULE, teacher_target_forward=C.TEACHER_TARGET_STEP,
        teacher_guidance=C.TEACHER_GUIDANCE, target_rows="all",
        lora=f"r{D.LORA_RANK} a{D.LORA_ALPHA}, {C.N_LORA_MODULES} Linear modules ({C.LORA_TARGET_REGEX})",
        loss=f"{D.REG_LOSS_WEIGHT} * MSE(h_student step 1, h_teacher forward {C.TEACHER_TARGET_STEP} of a "
             f"CFG-{C.TEACHER_GUIDANCE:g} Euler trajectory)",
        steps=a.steps, fastgen_config=FASTGEN_CONFIG, overrides=list(a.overrides))))

    env = dict(os.environ)
    env["REDIVERSE_DATA"] = data_root
    env["FASTGEN_OUTPUT_ROOT"] = fastgen_root
    env["PYTHONPATH"] = os.pathsep.join([REPO_ROOT] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [sys.executable, "train.py", "--config", FASTGEN_CONFIG, "-", *location,
           f"trainer.max_iter={a.steps + 1}", f"trainer.save_ckpt_iter={a.steps}", *a.overrides]
    print(" ".join(cmd), flush=True)
    with open(os.path.join(run_dir, "train.log"), "a") as log:
        p = subprocess.Popen(cmd, cwd=fastgen, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        rc = p.wait()
    if rc != 0:
        sys.exit(rc)
    if not a.no_extract:
        from sdxl.extract_adapter import extract
        extract(os.path.join(run_dir, "checkpoints", f"{a.steps:07d}.pth"), os.path.join(run_dir, "adapters"))


if __name__ == "__main__":
    main()

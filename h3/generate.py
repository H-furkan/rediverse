#!/usr/bin/env python
"""MiniMax-H3 generation for the five arms: teacher, student, rediverse_ft, rediverse_tf, rediverse.

  teacher       MiniMax-H3, 50 scheduler steps (49 forwards), no CFG
  student       Turbo 4-step LoRA (3 forwards)
  rediverse_ft  student + the ReDiverse-FT adapter, active in the first forward only
  rediverse_tf  student + ReDiverse-TF: the block-45 video rows of the first forward get
                sum_k alpha_k s_k u_k, alpha_k ~ U[-2, 2] (8 teacher directions of the prompt)
  rediverse     both: the ReDiverse-FT adapter and the TF offset in the first forward

The ReDiverse-FT adapter of MiniMax-H3 is downloaded from the Hugging Face Hub (minimax-h3/), or
taken from a local path with --adapter (e.g. one trained with h3.train and converted with
h3.convert_weights). It is added as a second
peft adapter beside the unfused Turbo adapter, its float32 weights cast to bfloat16; forward 1
runs both adapters, forwards 2-3 Turbo alone.

Benchmark mode (VBench-300, 10 clips per prompt, 1344x768, 124 frames, 24 fps) runs in three
phases, because the text encoder, the transformer and the video VAE do not fit one 80 GB GPU
together. Every phase resumes and can run as several workers (clips are claimed atomically):

  python -m h3.generate --encode --device cuda:0                    # text contexts (h3.encode_prompts)
  python -m h3.generate --arm rediverse --device cuda:0
  python -m h3.generate --arm rediverse --decode --device cuda:0    # mp4s

Clip k of VBench prompt i starts from randn(seed 1_000_000 + (10000 + i) * 100003 + k) in every
arm, and receives alpha = U[-2, 2]^8 from seed 7_000_000 + i * 1009 + k in both TF arms.

Single-prompt mode runs the three phases in one process and writes
<outputs>/prompt/<slug>/<arm>/{k:02d}.mp4, clip k following the rules above with i = --prompt-index.
The TF arms take the prompt's directions from --directions, else from
<outputs>/prompt/<slug>/directions.pt when it holds this prompt at this --prompt-index; otherwise
they are estimated first with the teacher (h3.estimate_directions, about one GPU-hour) and saved
there for later runs:

  python -m h3.generate --prompt "a corgi running on the beach" --arm rediverse_tf --num-samples 4 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

from h3 import config as C
from h3 import model as H
from rediverse import data, hub, seeds
from rediverse.claims import claim
from rediverse.directions import N_DIR, direction_file, load_directions, matches, tf_offset
from rediverse.video_io import vbench_name, write_mp4

CLAIM_EXPIRY_S = 3600


# ------------------------------------------------------------- ReDiverse-FT --

def load_rediverse_ft(device, adapter: str):
    """Student (Turbo bound as peft adapter) + the ReDiverse-FT adapter, both active."""
    from peft import LoraConfig
    m = H.load_arm("student", device)
    sd, meta = hub.load_lora_safetensors(adapter)
    ranks = {min(v.shape) for v in sd.values()}
    assert len(ranks) == 1, ranks
    r = ranks.pop()
    alpha = int(meta.get("alpha", r))
    targets = sorted({re.sub(r"\.lora_[AB]\..*", "", k) for k in sd})
    turbo = list(m.peft_config)[0]
    m.add_adapter(LoraConfig(r=r, lora_alpha=alpha, target_modules=targets), adapter_name=C.ADAPTER_NAME)
    own = dict(m.named_parameters())
    names = {k: re.sub(r"\.(lora_[AB])\.weight$", rf".\1.{C.ADAPTER_NAME}.weight", k) for k in sd}
    missing = [n for n in names.values() if n not in own]
    assert not missing, missing[:3]
    assert {n for n in own if f".{C.ADAPTER_NAME}." in n} == set(names.values()), "adapter tensors missing from the file"
    with torch.no_grad():
        for k, v in sd.items():
            own[names[k]].copy_(v.to(own[names[k]].dtype))
    m.set_adapters([turbo, C.ADAPTER_NAME], weights=[1.0, 1.0])
    nz = sum(1 for k, v in sd.items() if "lora_B" in k and float(v.abs().sum()) > 0)
    assert nz > 0, "adapter has an all-zero lora_B"
    print(f"[h3/rediverse_ft] adapter r={r} alpha={alpha} on {len(targets)} modules, {nz} non-zero lora_B; "
          f"active={m.active_adapters()}", flush=True)
    m.eval().requires_grad_(False)
    return m, turbo


# ------------------------------------------------------------- ReDiverse-TF --

class TFInjection:
    """Adds sum_k alpha_k s_k u_k to the block-45 video rows of the student's first forward."""

    def __init__(self, model, band: float = C.TF_BAND):
        self.U = self.s = self.alpha = None
        self.step, self.band = 0, band
        model.transformer_blocks[H.BLOCK].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if self.step != 0 or self.U is None:
            return None
        o = out[0] if isinstance(out, tuple) else out
        o = o.clone()
        delta = tf_offset(self.alpha, self.s, self.U)
        o[:, -H.NUM_VIDEO_ROWS:, :] += delta.reshape(1, H.NUM_VIDEO_ROWS, C.HIDDEN).to(o.dtype)
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o

    def set_directions(self, U, s):
        """U [K, D] bfloat16 and s [K] float32, already on the model's device."""
        self.U, self.s = U, s

    def clear(self):
        self.U = self.s = None
        torch.cuda.empty_cache()

    def set_sample(self, prompt_idx: int, k: int, device) -> list:
        al = seeds.tf_alphas(prompt_idx, k, k=self.U.shape[0], a=self.band)
        self.alpha = al.to(device)
        return seeds.alpha_record(al)


def build_arm(arm: str, device, adapter: str = None, band: float = C.TF_BAND):
    """(model, step_hook, TFInjection or None) of one arm."""
    assert arm in C.ARMS, arm
    gates = []
    if arm in C.FT_ARMS:
        model, turbo = load_rediverse_ft(device, adapter)

        def ft_gate(k):
            model.set_adapters([turbo, C.ADAPTER_NAME] if k == 0 else [turbo],
                               weights=[1.0, 1.0] if k == 0 else [1.0])
        gates.append(ft_gate)
    else:
        model = H.load_arm("teacher" if arm == "teacher" else "student", device)
    tf = None
    if arm in C.TF_ARMS:
        tf = TFInjection(model, band)

        def tf_gate(k):
            tf.step = k
        gates.insert(0, tf_gate)
    if not gates:
        return model, None, None

    def gate(k):
        for g in gates:
            g(k)
    return model, gate, tf


def steps_of(arm: str) -> int:
    return C.STEPS["teacher"] if arm == "teacher" else C.STEPS["student"]


# ---------------------------------------------------------------- benchmark --

def run_benchmark(a):
    texts = data.read_prompts(data.VBENCH_300)
    sel = data.parse_ids(a.prompt_ids) if a.prompt_ids else list(range(len(texts)))
    assert sel and max(sel) < len(texts), sel
    ctx_dir = a.ctx_dir or C.vbench_ctx_dir()
    if a.encode:
        from h3.encode_prompts import encode, vbench_entries
        encode(vbench_entries(sel, ctx_dir), a.device)
        return
    si, sn = a.shard, a.num_shards
    idxs = sel[si::sn]
    if a.reverse:
        idxs = idxs[::-1]
    arm_dir = a.out_dir or C.output_dir(a.arm)
    lat_dir = os.path.join(arm_dir, "latents")
    vid_dir = os.path.join(arm_dir, "videos")
    os.makedirs(lat_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    dev = a.device

    if a.decode:
        dec = H.Decoder(dev)
        n = 0
        for i in idxs:
            for k in range(a.num_samples):
                f = os.path.join(vid_dir, vbench_name(texts[i], k))
                lf = os.path.join(lat_dir, f"v{i:03d}_s{k}.pt")
                if os.path.exists(f) or not os.path.exists(lf) or not claim(f, CLAIM_EXPIRY_S):
                    continue
                write_mp4(dec(torch.load(lf).unsqueeze(0).to(dev)), f, C.FPS)
                n += 1
        print(f"[h3/{a.arm}] decoded {n} clips", flush=True)
        return

    dir_bank = a.directions_dir or C.direction_dir()
    if a.arm in C.TF_ARMS:
        pending = [i for i in idxs if not all(os.path.exists(os.path.join(lat_dir, f"v{i:03d}_s{k}.pt"))
                                              for k in range(a.num_samples))]
        absent = [i for i in pending if not os.path.exists(direction_file(dir_bank, i, "v"))]
        if absent:
            sys.exit(f"{len(absent)} direction files missing in {dir_bank}, e.g. v{absent[0]:03d}.pt; "
                     "run python -m h3.estimate_directions first")
    model, gate, tf = build_arm(a.arm, dev, a.adapter_file, a.tf_band)
    steps = steps_of(a.arm)
    scales = {}
    t0, done = time.time(), 0
    for i in idxs:
        ctx_path = os.path.join(ctx_dir, f"v{i:03d}.pt")
        assert os.path.exists(ctx_path), f"{ctx_path} missing; run python -m h3.generate --encode first"
        if all(os.path.exists(os.path.join(lat_dir, f"v{i:03d}_s{k}.pt")) for k in range(a.num_samples)):
            continue
        ctx = H.load_ctx(ctx_path)
        if tf is not None:
            tf.clear()            # free the previous prompt's directions first
            tf.set_directions(*load_directions(direction_file(dir_bank, i, "v"), a.n_dir, dev, prompt=texts[i]))
        for k in range(a.num_samples):
            lf = os.path.join(lat_dir, f"v{i:03d}_s{k}.pt")
            if os.path.exists(lf) or not claim(lf, CLAIM_EXPIRY_S):
                continue
            x0 = H.eval_latent(i, k).unsqueeze(0)
            if tf is not None:
                scales[f"v{i:03d}_s{k}"] = tf.set_sample(i, k, dev)
            lat = H.sample(model, x0, ctx, dev, steps, step_hook=gate)
            torch.save(lat[0].cpu(), lf + ".tmp"); os.replace(lf + ".tmp", lf)
            done += 1; el = time.time() - t0
            print(f"  [{done}] {el/done:.0f}s/clip eta {(len(idxs)*a.num_samples-done)*el/done/3600:.1f}h "
                  f"{texts[i][:44]!r}", flush=True)
    if done:
        meta = dict(arm=a.arm, backbone=C.BACKBONE, prompts="prompts/vbench_300.txt", num_samples=a.num_samples,
                    adapter=a.adapter_file, frames=C.NUM_FRAMES, fps=C.FPS, steps=steps,
                    resolution=f"{C.WIDTH}x{C.HEIGHT}", seed_rule="1_000_000 + (10000+idx)*100003 + k",
                    tf=(dict(directions=dir_bank, band=[-a.tf_band, a.tf_band], n_dir=min(a.n_dir, N_DIR),
                             rule="h += sum_k alpha_k s_k u_k, alpha ~ U[-a,a] iid per (clip, k)",
                             alpha_seed="7_000_000 + idx*1009 + k") if a.arm in C.TF_ARMS else None))
        data.write_run_meta(os.path.join(arm_dir, "meta.json"), meta)
    if scales:
        sp = os.path.join(arm_dir, f"scales_{os.getpid()}.json")
        with open(sp + ".tmp", "w") as f:
            json.dump(scales, f, indent=0)
        os.replace(sp + ".tmp", sp)
    print(f"[h3/{a.arm}] sampling shard {si}/{sn} done", flush=True)


# ------------------------------------------------------------- single prompt --

def prompt_directions(a, ctx) -> str:
    """Direction file of the prompt: --directions as given, else <outputs>/prompt/<slug>/directions.pt
    when it holds this prompt at --prompt-index; otherwise they are estimated into that path first
    (teacher loaded and released)."""
    from h3 import estimate_directions as E
    if a.directions:
        return a.directions
    path = E.prompt_direction_file(a.prompt)
    if not matches(path, a.prompt, a.prompt_index):
        if os.path.exists(path):
            print(f"[h3/prompt] {path} holds another prompt or --prompt-index; estimating it again", flush=True)
        print(f"[h3/prompt] estimating ReDiverse-TF directions from {C.N_DIRECTION_SAMPLES} teacher rollouts",
              flush=True)
        E.estimate_prompt(ctx, a.prompt, a.prompt_index, a.device, path)
        print(f"[h3/prompt] directions saved to {path}", flush=True)
    return path


def run_prompt(a):
    from h3.encode_prompts import prompt_context
    dev = a.device
    pidx = a.prompt_index
    out_dir = a.out_dir or data.prompt_dir(C.BACKBONE, a.prompt, a.arm)
    os.makedirs(out_dir, exist_ok=True)

    ctx = prompt_context(a.prompt, pidx, dev)
    dir_file = prompt_directions(a, ctx) if a.arm in C.TF_ARMS else None

    model, gate, tf = build_arm(a.arm, dev, a.adapter_file, a.tf_band)
    if tf is not None:
        print(f"[h3/prompt] ReDiverse-TF directions {dir_file}", flush=True)
        tf.set_directions(*load_directions(dir_file, a.n_dir, dev))
    lats = []
    for k in range(a.num_samples):
        x0 = H.eval_latent(pidx, k).unsqueeze(0)
        if tf is not None:
            tf.set_sample(pidx, k, dev)
        lats.append(H.sample(model, x0, ctx, dev, steps_of(a.arm), step_hook=gate)[0].cpu())
        print(f"  [{k + 1}/{a.num_samples}] sampled", flush=True)
    del model, gate, tf
    H.free()

    dec = H.Decoder(dev)
    for k, lat in enumerate(lats):
        f = os.path.join(out_dir, f"{k:02d}.mp4")
        write_mp4(dec(lat.unsqueeze(0).to(dev)), f, C.FPS)
        print(f"  wrote {f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=C.ARMS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--adapter", default=None,
                    help="ReDiverse-FT adapter: a .safetensors file or a directory holding rediverse_ft.safetensors "
                         "(directly or under minimax-h3/); default: download from the Hugging Face Hub")
    ap.add_argument("--n-dir", type=int, default=C.N_DIR, help="K directions injected")
    ap.add_argument("--tf-band", type=float, default=C.TF_BAND, help="alpha_k ~ U[-a, a]")
    ap.add_argument("--out-dir", default="",
                    help="output directory of the arm (default <outputs>/<arm>, or <outputs>/prompt/<slug>/<arm> "
                         "with --prompt)")
    ap.add_argument("--num-samples", type=int, default=None,
                    help=f"clips per prompt (default {C.CLIPS_PER_PROMPT}, or 4 with --prompt)")
    # benchmark mode
    ap.add_argument("--encode", action="store_true", help="encode the VBench text contexts")
    ap.add_argument("--decode", action="store_true", help="decode the arm's latents to mp4")
    ap.add_argument("--prompt-ids", default="", help="subset of VBench indices, e.g. '0,5' or '0-29' (default: all 300)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    ap.add_argument("--num-shards", type=int, default=1, help="number of shards; shard i takes every N-th prompt")
    ap.add_argument("--reverse", action="store_true", help="walk this worker's prompts in reverse order")
    ap.add_argument("--ctx-dir", default="", help="VBench context directory (default <data>/ctx/vbench)")
    ap.add_argument("--directions-dir", default="", help="direction bank (default <data>/directions/b45_k16)")
    # single-prompt mode
    ap.add_argument("--prompt", default="", help="generate clips for this prompt (single-prompt mode)")
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="--prompt: prompt index of the seed rules (a VBench index reproduces that prompt's clips)")
    ap.add_argument("--directions", default="",
                    help="--prompt: direction file (default <outputs>/prompt/<slug>/directions.pt, estimated "
                         "there unless it holds this prompt at --prompt-index)")
    a = ap.parse_args()
    if a.num_samples is None:
        a.num_samples = 4 if a.prompt else C.CLIPS_PER_PROMPT
    if (a.prompt or not a.encode) and not a.arm:
        ap.error("--arm is required")
    if not 0 <= a.shard < a.num_shards:
        ap.error("--shard must be in [0, --num-shards)")
    a.adapter_file = None
    if a.arm in C.FT_ARMS and not (a.encode or a.decode):
        if a.adapter and a.adapter.endswith(".pth"):
            ap.error("--adapter takes the adapter converted by h3.convert_weights, not a training checkpoint")
        a.adapter_file = hub.resolve_adapter(a.adapter, "h3")
    if a.prompt and a.directions and not os.path.isfile(a.directions):
        ap.error(f"--directions {a.directions}: no such file")
    if a.prompt:
        run_prompt(a)
    else:
        run_benchmark(a)


if __name__ == "__main__":
    main()

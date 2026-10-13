#!/usr/bin/env python
"""Wan 2.1 generation for the five arms (run in the `video` environment).

  teacher       Wan2.1-T2V-14B, UniPC 50 steps (repo scheduler, flow shift 3), CFG 5 against the
                negative prompt (two forwards per step)
  student       lightx2v 4-step StepDistill-CfgDistill, guidance 1
  rediverse_ft  student + the ReDiverse-FT LoRA, active at the first step only
  rediverse_tf  student + ReDiverse-TF: at step 1, blocks.28 output h += sum_k alpha_k s_k u_k with the
                prompt's K=8 teacher directions and alpha_k ~ U[-2, 2] (rediverse.seeds.tf_alphas)
  rediverse     both

81 frames at 832x480, decoded by the float32 VAE and written as H.264 (CRF 17, 16 fps). Clip k of
VBench prompt i starts from rediverse.seeds.wan_eval_seed(i, k) in every arm and draws its TF
coefficients from tf_alphas(i, k), so the arms are paired clip by clip.

Benchmark mode (VBench-300 x 10 clips, VBench naming "<prompt>-<k>.mp4"; several workers may share
one output directory):
  python -m wan.generate --arm rediverse --device cuda:0 [--shard 0 --num-shards 4] [--reverse]
Single prompt (clip k uses the seed rules of prompt index --prompt-index, default 0, so the text of
VBench prompt i with --prompt-index i reproduces that prompt's benchmark clips):
  python -m wan.generate --arm rediverse --prompt "a corgi surfing a wave" --num-samples 4
  -> <REDIVERSE_OUT>/wan2.1/prompt/<slug>/<arm>/00.mp4, 01.mp4, ...
The TF arms read the prompt's directions from --directions, or else from
<REDIVERSE_OUT>/wan2.1/prompt/<slug>/directions.pt; unless that file holds this prompt's
directions for this --prompt-index, the base teacher estimates them into it first (as
`python -m wan.estimate_directions --prompt`, ~15 min on an H100) and is freed before the student
is loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from rediverse import claims, data, directions, hub, seeds, video_io
from wan import config as C
from wan import encode_prompts, estimate_directions
from wan import model as M

CLAIM_EXPIRY_S = 3600         # a clip takes at most ~450 s; a claim this old without its mp4 is retaken


def build_model(arm: str, adapter: str, device):
    """The arm's transformer, with the TF hook state `pc` (None for arms without TF)."""
    if arm == "teacher":
        return M.load_transformer("teacher", device), None
    model = M.load_transformer("student", device)
    if arm in C.FT_ARMS:
        path = hub.resolve_adapter(adapter, "wan")
        sd, meta = hub.load_lora_safetensors(path)
        print(f"[wan] adapter {path} ({meta.get('backbone', '?')}, {meta.get('anchor_block', '?')})", flush=True)
        M.attach_adapter(model, sd, alpha=meta.get("alpha"))
    pc = None
    if arm in C.TF_ARMS:
        pc = dict(U=None, s=None, alpha=None, step=0)

        def tf_hook(_m, _i, out):
            if pc["step"] != 0 or pc["U"] is None:
                return None
            o = out[0] if isinstance(out, tuple) else out
            o = o + directions.tf_offset(pc["alpha"], pc["s"], pc["U"]).reshape(o.shape).to(o.dtype)
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        model.blocks[C.ANCHOR_BLOCK].register_forward_hook(tf_hook)
    return model, pc


def sample(arm: str, model, pc, lat, emb, neg, device):
    """One clip's final latent from the fp32 initial latent `lat`."""
    if arm == "teacher":
        return M.sample_teacher(model, lat, emb, neg, device)
    step_hook = None
    if pc is not None:
        def step_hook(k):
            pc["step"] = k
    if arm in C.FT_ARMS:
        model.enable_adapters()
    return M.sample_student(model, lat, emb, device, step_hook=step_hook, adapter_gate=arm in C.FT_ARMS)


def prompt_directions(a, emb, device) -> str:
    """Direction file of a --prompt run of a TF arm: --directions as given, else the prompt's
    directions.pt, (re-)estimated with the base teacher unless it holds the directions of this
    prompt and prompt index."""
    if a.directions:
        return a.directions
    path = estimate_directions.prompt_file(a.prompt)
    if not directions.matches(path, a.prompt, a.prompt_index):
        print(f"[wan/{a.arm}] estimating the prompt's ReDiverse-TF directions with the teacher "
              f"(prompt index {a.prompt_index})", flush=True)
        estimate_directions.estimate_prompt(a.prompt, emb, a.prompt_index, device, path)
    return path


def render(arm, model, pc, seed_idx: int, clip: int, emb, neg, device, path):
    """One clip: seed-rule latent (and TF coefficients), sampler, decode, mp4. Returns the alphas."""
    lat = seeds.wan_latent(seeds.wan_eval_seed(seed_idx, clip), C.FRAMES).to(device, torch.float32)
    al = None
    if pc is not None:
        al = seeds.tf_alphas(seed_idx, clip, k=pc["U"].shape[0], a=C.TF_BAND)
        pc["alpha"] = al.to(device)
        pc["step"] = 0
    with torch.no_grad():
        z = sample(arm, model, pc, lat, emb, neg, device)
        v = np.asarray(M.decode(z, device))
    v = v[0] if v.ndim == 5 else v
    assert v.shape[0] == C.FRAMES, f"decoded {v.shape[0]} frames"
    video_io.write_mp4(v, path, C.FPS)
    return al


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=C.ARMS)
    ap.add_argument("--adapter", default=None,
                    help="ReDiverse-FT adapter: a .safetensors file or a directory holding rediverse_ft.safetensors "
                         "(directly or under wan2.1/); default: download from the Hugging Face Hub")
    ap.add_argument("--n-dir", type=int, default=C.N_DIR)
    ap.add_argument("--num-samples", type=int, default=None,
                    help=f"clips per prompt (default {C.CLIPS_PER_PROMPT} in benchmark mode, 4 with --prompt)")
    ap.add_argument("--out-dir", default=None,
                    help=f"default {C.output_dir('<arm>', 'videos')}, or "
                         f"{C.output_dir('prompt', '<slug>', '<arm>')} with --prompt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt-ids", default=None, help="benchmark mode: VBench indices, e.g. '0-299' (default all)")
    ap.add_argument("--shard", type=int, default=0,
                    help="benchmark mode: this worker takes the prompts with index %% num-shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--reverse", action="store_true", help="benchmark mode: walk the prompts back to front")
    ap.add_argument("--directions-dir", default=None,
                    help=f"benchmark mode: ReDiverse-TF directions (default {C.direction_dir()})")
    ap.add_argument("--prompt", default=None, help="single-prompt mode: the text")
    ap.add_argument("--prompt-index", type=int, default=0, help="single-prompt mode: prompt index of the seed rules")
    ap.add_argument("--directions", default=None,
                    help="single-prompt mode: direction file of the TF arms (default: directions.pt of the "
                         "prompt's directory, estimated with the teacher unless it holds this prompt and "
                         "--prompt-index)")
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in 0..{a.num_shards - 1}")
    if a.directions and not os.path.isfile(a.directions):
        ap.error(f"--directions {a.directions}: no such file")
    dev = a.device
    prompt_mode = a.prompt is not None

    if prompt_mode:
        n = 4 if a.num_samples is None else a.num_samples
        out = a.out_dir or data.prompt_dir(C.BACKBONE, a.prompt, a.arm)
        texts, seed_of = {a.prompt_index: a.prompt}, {a.prompt_index: a.prompt_index}
        if a.arm == "teacher":
            encode_prompts.ensure(["neg"], dev)
        tok, enc = M.text_encoder(dev)
        emb = M.embed(tok, enc, a.prompt, dev)[0].cpu()
        del tok, enc
        torch.cuda.empty_cache()
        emb_of, dir_of = {a.prompt_index: emb}, {}
        if a.arm in C.TF_ARMS:
            dfile = prompt_directions(a, emb, dev)
            dir_of[a.prompt_index] = dfile
            print(f"[wan/{a.arm}] directions {dfile}", flush=True)
        work = {a.prompt_index: [(k, os.path.join(out, f"{k:02d}.mp4")) for k in range(n)]}
    else:
        n = C.CLIPS_PER_PROMPT if a.num_samples is None else a.num_samples
        allt = data.read_prompts(data.VBENCH_300)
        sel = data.parse_ids(a.prompt_ids) if a.prompt_ids else list(range(len(allt)))
        ids = [i for i in sel if i % a.num_shards == a.shard][::-1 if a.reverse else 1]
        out = a.out_dir or C.output_dir(a.arm, "videos")
        texts, seed_of = {i: allt[i] for i in ids}, {i: i for i in ids}
        ddir = a.directions_dir or C.direction_dir()
        dir_of = {i: directions.direction_file(ddir, i, "v") for i in ids}
        encode_prompts.ensure(["vbench"] + (["neg"] if a.arm == "teacher" else []), dev)
        emb_of = {i: os.path.join(C.vbench_ctx_dir(), f"v{i:03d}.pt") for i in ids}
        work = {i: [(k, video_io.clip_path(out, texts[i], k)) for k in range(n)] for i in ids}
        work = {i: [(k, f) for k, f in kf if not os.path.exists(f)] for i, kf in work.items()}
        if a.arm in C.TF_ARMS:
            absent = [dir_of[i] for i, kf in work.items() if kf and not os.path.isfile(dir_of[i])]
            if absent:
                sys.exit(f"{len(absent)} direction files missing, e.g. {absent[0]}; "
                         "run python -m wan.estimate_directions first")
    os.makedirs(out, exist_ok=True)
    n_pending = sum(len(v) for v in work.values())
    print(f"[wan/{a.arm}] {len(texts)} prompts, {n_pending} clips to render -> {out}", flush=True)

    model, pc = build_model(a.arm, a.adapter, dev)
    neg = M.load_ctx(C.neg_ctx_path())["encoder_hidden_states"].to(dev, M.DTYPE) if a.arm == "teacher" else None
    scales, t0, done = {}, time.time(), 0
    for i, kf in work.items():
        if not kf:
            continue
        e = emb_of[i]
        emb = (M.load_ctx(e)["encoder_hidden_states"] if isinstance(e, str) else e).to(dev, M.DTYPE)
        if pc is not None:
            pc["U"] = pc["s"] = None
            pc["U"], pc["s"] = directions.load_directions(dir_of[i], a.n_dir, dev,
                                                          prompt=None if prompt_mode else texts[i])
        for k, f in kf:
            if not prompt_mode and (os.path.exists(f) or not claims.claim(f, CLAIM_EXPIRY_S)):
                continue
            al = render(a.arm, model, pc, seed_of[i], k, emb, neg, dev, f)
            if al is not None:
                scales[f"{k:02d}" if prompt_mode else f"v{seed_of[i]:03d}_s{k}"] = seeds.alpha_record(al)
            done += 1
            el = time.time() - t0
            print(f"  [{done}/{n_pending}] {el / done:.0f} s/clip  eta {(n_pending - done) * el / done / 3600:.1f} h  "
                  f"{texts[i][:50]!r} #{k}", flush=True)
    root = out if prompt_mode else os.path.dirname(os.path.abspath(out))
    if scales:
        sp = os.path.join(root, "scales.json" if prompt_mode else f"scales_{os.getpid()}.json")
        with open(sp + ".tmp", "w") as fh:
            json.dump(scales, fh, indent=0)
        os.replace(sp + ".tmp", sp)
    meta = dict(arm=a.arm, backbone=C.BACKBONE, n_prompts=1 if prompt_mode else len(sel),
                frames=C.FRAMES, fps=C.FPS, resolution=f"{C.WIDTH}x{C.HEIGHT}",
                steps=C.TEACHER_STEPS if a.arm == "teacher" else C.STUDENT_STEPS,
                guidance=C.TEACHER_GUIDANCE if a.arm == "teacher" else 1.0,
                seed_rule="1000000 + prompt_index*100003 + clip",
                tf=(dict(n_dir=a.n_dir, band=[-C.TF_BAND, C.TF_BAND], site=f"blocks.{C.ANCHOR_BLOCK}, step 1",
                         alpha_seed="7000000 + prompt_index*1009 + clip") if a.arm in C.TF_ARMS else None),
                ft=(dict(gate="step 1 only") if a.arm in C.FT_ARMS else None),
                naming="'{clip:02d}.mp4'" if prompt_mode else "VBench '{prompt}-{clip}.mp4'")
    if prompt_mode:
        meta.update(prompt=a.prompt, prompt_index=a.prompt_index, n_clips=n)
    with open(os.path.join(root, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[wan/{a.arm}] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()

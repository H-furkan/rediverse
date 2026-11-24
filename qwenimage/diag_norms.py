#!/usr/bin/env python
"""Does the ReDiverse-FT adapter inflate activations? (quality diagnostics)

  python -m qwenimage.diag_norms --student flash20 --adapter <ft.safetensors> --block 57 --tau 9 [--captions 0,1,2,3]

For a few training captions x noises (the bank's own pairs), reports at block l (image rows):
  rms     sqrt(mean h^2) over tokens and channels
  spread  per-sample distance from the across-noise mean, sqrt(mean_i ||h_i - mean||^2 / D)
for the student's step-1 forward without the FT adapter, with it at scale 0.5 and 1, and for the teacher targets in the
bank (teacher forward tau). It also reports the full first-step velocity RMS and, after sampling the student's whole
schedule from GenEval eval noises, the final packed-latent RMS / mean for the same three adapter settings.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from qwenimage import config as C
from qwenimage import model as M
from qwenimage.build_targets import bank_dir, load_train_context
from qwenimage.generate import Generator, ctx_dir, ensure_contexts, eval_latent, load_context
from rediverse import data as D
from rediverse.seeds import train_noise


def rms(h):
    return float(h.float().pow(2).mean().sqrt())


def spread(h):
    h = h.float().flatten(1)
    return float((h - h.mean(0, keepdim=True)).pow(2).sum(1).mean().div(h.shape[1]).sqrt())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--tau", type=int, required=True)
    ap.add_argument("--captions", default="0,1,2,3")
    ap.add_argument("--noises", type=int, default=4)
    ap.add_argument("--prompt-ids", default="0,27")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    fam = C.family_of(a.student)
    f = C.family(fam)
    caps = D.captions()
    cis = [int(c) for c in a.captions.split(",")]
    bdir = os.path.join(bank_dir(fam, a.tau), f"b{a.block:02d}")
    gen = Generator(a.student, a.device, adapter=a.adapter, decode=False)
    model, dev = gen.model, a.device
    settings = [("student", 0.0), ("ft x0.5", 0.5), ("ft x1", 1.0)]
    res = {k: dict(rms=[], spread=[], v_rms=[], final_rms=[], final_mean=[]) for k, _ in settings}
    res["teacher (bank)"] = dict(rms=[], spread=[])
    t1 = M.model_timestep(a.student, 0, a.noises, dev)
    with torch.no_grad():
        for ci in cis:
            ctx = load_train_context(fam, ci, caps[ci]).to(dev)
            x = torch.stack([train_noise(ci, ni, f["latent_shape"]) for ni in range(a.noises)]).to(dev, C.DTYPE)
            tgt = torch.load(os.path.join(bdir, f"c{ci:03d}.pt"), map_location="cpu", mmap=True)["h"][:a.noises]
            res["teacher (bank)"]["rms"].append(rms(tgt))
            res["teacher (bank)"]["spread"].append(spread(tgt))
            for name, sc in settings:
                M.set_active(model, gen.base + ([M.FT_ADAPTER] if sc > 0 else []), {M.FT_ADAPTER: sc})
                h = M.forward_to_blocks(fam, model, x, ctx, t1, [a.block])[a.block]
                v = M.forward(fam, model, x, ctx, t1)
                res[name]["rms"].append(rms(h))
                res[name]["spread"].append(spread(h))
                res[name]["v_rms"].append(rms(v))
        texts = D.read_prompts(D.GENEVAL_553)
        ids = D.parse_ids(a.prompt_ids)
        ensure_contexts(fam, {i: texts[i] for i in ids}, ctx_dir(fam), dev)
        gen.use_ft = True
        for i in ids:
            ctx = load_context(ctx_dir(fam), i, texts[i])
            lat = torch.stack([eval_latent(fam, i, s) for s in range(a.samples)])
            for name, sc in settings:
                gen.use_ft, gen.ft_scale = sc > 0, sc
                xf = gen.sample(ctx, lat).float()
                res[name]["final_rms"].append(float(xf.pow(2).mean().sqrt()))
                res[name]["final_mean"].append(float(xf.mean()))
    M.set_active(model, gen.base)
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    summary = {k: {kk: mean(vv) for kk, vv in d.items()} for k, d in res.items()}
    lines = [f"diag_norms {a.student} block {a.block} tau {a.tau} adapter {a.adapter}",
             "| setting | block rms | block spread | first-step v rms | final latent rms | final latent mean |",
             "|---|---:|---:|---:|---:|---:|"]
    for k, d in summary.items():
        g = lambda key: "-" if key not in d else f"{d[key]:.4g}"
        lines.append(f"| {k} | {g('rms')} | {g('spread')} | {g('v_rms')} | {g('final_rms')} | {g('final_mean')} |")
    print("\n".join(lines), flush=True)
    if a.out:
        json.dump(dict(summary=summary, raw=res, args=vars(a)), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

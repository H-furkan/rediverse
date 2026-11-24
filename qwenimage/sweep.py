#!/usr/bin/env python
"""Inference sweep for one Qwen-Image student (one model load, many ReDiverse configurations).

  python -m qwenimage.sweep --student lightning --adapter <ft.safetensors> --block 20 --tau 10 \
      --configs ft,tf2,tf4,rd2,rd4 [--prompt-ids 0,11,17,27,28,34 --num-samples 8] [--name-suffix _s200]

Configurations (names are the output folders <REDIVERSE_OUT>/<backbone>/<student>/sweep/<name><suffix>/images):
  ft      FT adapter in the first forward
  tfB     TF only, alpha ~ U[-B, B]          (e.g. tf2, tf4)
  rdB     FT + TF, alpha ~ U[-B, B]          (e.g. rd2, rd4)
  ftN     FT adapter in the first N forwards (e.g. ft2)
  ...xS   any of the above with the FT adapter at scale S (= lora_B x S), e.g. ftx0.5, rd2x0.5
Every configuration is seed-matched with the student, teacher and preview sets (generate.py's seed rules).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch

from qwenimage import config as C
from qwenimage import model as M
from qwenimage.generate import Generator, ctx_dir, ensure_contexts, eval_latent, load_context, save_png
from rediverse import data as D
from rediverse.directions import direction_file, load_direction_file
from rediverse.seeds import tf_alphas_batch


def parse(cfg: str) -> dict:
    """ft[N] | tfB | rdB, optionally followed by xS = FT adapter scale (lora_B x S), e.g. ftx0.5, rd2x0.5."""
    m = re.fullmatch(r"(ft|tf|rd)(\d*(?:\.\d+)?)(?:x(\d+(?:\.\d+)?))?", cfg)
    assert m, f"bad config {cfg}"
    kind, num, scale = m.group(1), m.group(2), float(m.group(3)) if m.group(3) else 1.0
    if kind == "ft":
        return dict(name=cfg, ft=True, tf=False, ft_steps=int(num) if num else 1, band=None, ft_scale=scale)
    return dict(name=cfg, ft=kind == "rd", tf=True, ft_steps=1, band=float(num) if num else 2.0, ft_scale=scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--tau", type=int, required=True)
    ap.add_argument("--allow-tau-mismatch", action="store_true")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--prompts", default=D.GENEVAL_553)
    ap.add_argument("--prompt-ids", default="0,11,17,27,28,34")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--n-dir", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    fam = C.family_of(a.student)
    cfgs = [parse(c) for c in a.configs.split(",")]
    assert a.adapter or not any(c["ft"] for c in cfgs), "FT configurations need --adapter"
    texts = D.read_prompts(a.prompts)
    ids = D.parse_ids(a.prompt_ids)
    S = a.num_samples
    root = C.output_dir(fam, a.student, "sweep")
    cdir = ctx_dir(fam)
    ensure_contexts(fam, {i: texts[i] for i in ids}, cdir, a.device)
    ddir = C.data_dir(fam, f"directions_k{a.tau}", f"b{a.block:02d}")
    tsig = M.teacher_target_sigmas(a.student, a.tau)
    assert a.allow_tau_mismatch or a.tau == tsig["nearest_k"], (a.tau, tsig)
    gen = Generator(a.student, a.device, adapter=a.adapter, block=a.block)
    print(f"[qwen-sweep] {a.student}: {M.provenance(a.student)}, block {a.block}, tau {a.tau}, adapter {a.adapter}", flush=True)
    for c in cfgs:
        name = c["name"] + a.name_suffix
        out = os.path.join(root, name, "images")
        todo = [i for i in ids if not os.path.exists(os.path.join(out, f"{i:03d}_s{S - 1}.png"))]
        if not todo:
            continue
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(root, name, "config.json"), "w") as fh:
            json.dump(dict(c, adapter=a.adapter, block=a.block, tau=a.tau, n_dir=a.n_dir, student=a.student,
                           **M.provenance(a.student)), fh, indent=1)
        gen.use_ft = c["ft"]
        gen.ft_scale = c.get("ft_scale", 1.0)
        t0 = time.time()
        for i in todo:
            ctx = load_context(cdir, i, texts[i])
            if c["tf"]:
                rec = load_direction_file(direction_file(ddir, i))
                assert rec["prompt"] == texts[i] and rec["block"] == a.block and rec["k"] == a.tau
                assert abs(float(rec["sigma_out"]) - tsig["sigma_out"]) < 1e-6, (rec["sigma_out"], tsig)
                M.check_provenance(rec, M.provenance(f"teacher:{fam}", guided=False), direction_file(ddir, i))
                gen.set_directions(rec["U"][:a.n_dir], rec["s"][:a.n_dir].float())
                del rec
            else:
                gen.set_directions(None, None)
            for s0 in range(0, S, a.batch):
                ss = list(range(s0, min(S, s0 + a.batch)))
                lat = torch.stack([eval_latent(fam, i, s) for s in ss])
                alpha = tf_alphas_batch(i, ss, k=a.n_dir, a=c["band"]) if c["tf"] else None
                for s, im in zip(ss, gen.decoder(gen.sample(ctx, lat, alpha, ft_steps=c["ft_steps"]))):
                    save_png(im, os.path.join(out, f"{i:03d}_s{s}.png"))
        print(f"[qwen-sweep] {a.student} {name}: {len(todo)} prompts in {time.time() - t0:.0f}s", flush=True)
    gaps = [(c["name"] + a.name_suffix, i) for c in cfgs for i in ids
            if not os.path.exists(os.path.join(root, c["name"] + a.name_suffix, "images", f"{i:03d}_s{S - 1}.png"))]
    if gaps:
        raise SystemExit(f"[qwen-sweep] INCOMPLETE: {len(gaps)} (config, prompt) pairs missing, e.g. {gaps[:3]}")
    print("[qwen-sweep] done", flush=True)


if __name__ == "__main__":
    main()

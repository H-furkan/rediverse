#!/usr/bin/env python
"""Inference-knob sweep for ReDiverse on Krea 2 (one model load, many configurations).

Each configuration is a dict with any of:
  name        output folder: <REDIVERSE_OUT>/krea2/sweep/<name>/images/{i:03d}_s{s}.png
  adapter     ReDiverse-FT .safetensors (null = no adapter)
  scale       LoRA weight multiplier (default 1.0; the paper uses 1.0)
  ft_steps    number of leading sampling steps with the adapter on (paper: 1)
  tf_block    block of the TF injection (null = no TF)
  band        alpha ~ U[-band, band] (paper: 2)
  tf_steps    number of leading steps with the TF offset (paper: 1)
  n_dir       directions used (<= 8 stored)
Every configuration uses the same prompts and the same eval noises (seed i * 1000 + s) and the same alpha seeds,
so rows are seed-matched with the student / teacher / preview sets.

  python -m krea2.sweep --spec sweep.json --shard 0 --num-shards 2 --prompt-ids 0,11,17,27,28,34,63 --num-samples 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from krea2 import config as C
from krea2 import model as M
from krea2.estimate_directions import ctx_dir, ensure_contexts, load_context
from krea2.generate import eval_latent, save_png
from rediverse import data as D
from rediverse.directions import direction_file, load_direction_file, tf_offset
from rediverse.seeds import tf_alphas_batch


class Sweeper:
    def __init__(self, device):
        from krea2.generate import Generator
        self.g = Generator("student", device)          # Turbo + VAE + decode
        self.model, self.device = self.g.model, device
        self.sigmas = self.g.sigmas
        self.adapter = None                            # path currently loaded
        self.lora_last = None
        self.cfg = {}
        self.tf = dict(U=None, s=None, alpha=None, step=-1)
        self.hooks = {}

    # ---------------- adapter ----------------
    def set_adapter(self, path, scale):
        if path is None:
            if self.adapter is not None:
                self.model.disable_adapters()
            return
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(path, "pt") as fh:
            md = fh.metadata() or {}
        last = int(md.get("lora_last", -1))
        if self.adapter is None:
            M.load_lora(self.model, path)
            self.lora_last = last
        elif path != self.adapter:
            assert last == self.lora_last, f"adapter {path} covers blocks 0..{last}, loaded one 0..{self.lora_last}"
            res = self.model.load_state_dict(load_file(path), strict=False)
            assert not res.unexpected_keys, res.unexpected_keys[:3]
        self.adapter = path
        self.model.set_adapters([M.FT_ADAPTER], weights=[float(scale)])

    # ---------------- TF hook ----------------
    def ensure_hook(self, block):
        for b, h in list(self.hooks.items()):
            if b != block:
                h.remove()
                del self.hooks[b]
        if block is None or block in self.hooks:
            return

        def hook(_m, _i, out):
            if self.tf["U"] is None or self.tf["step"] < 0 or self.tf["step"] >= self.cfg.get("tf_steps", 1):
                return None
            delta = tf_offset(self.tf["alpha"], self.tf["s"], self.tf["U"])
            o = out.clone()
            o[:, -C.N_IMG_TOKENS:] += delta.reshape(out.shape[0], C.N_IMG_TOKENS, C.D_MODEL).to(out.dtype)
            return o
        self.hooks[block] = self.model.transformer_blocks[block].register_forward_hook(hook)

    @torch.no_grad()
    def sample(self, ctx, lat, alpha):
        x = lat.to(self.device, C.DTYPE)
        self.tf["alpha"] = alpha.to(self.device) if alpha is not None else None
        sg, ft_steps = self.sigmas, self.cfg.get("ft_steps", 1)
        for j in range(len(sg) - 1):
            self.tf["step"] = j
            if self.adapter is not None and self.cfg.get("adapter"):
                (self.model.enable_adapters if j < ft_steps else self.model.disable_adapters)()
            v = M.forward(self.model, x, ctx, sg[j])
            x = M.euler_step(x, v, sg[j], sg[j + 1])
        self.tf["step"] = -1
        if self.adapter is not None:
            self.model.disable_adapters()
        return x

    def run(self, cfg, texts, ids, S, B, root):
        self.cfg = cfg
        out = os.path.join(root, cfg["name"], "images")
        todo = [i for i in ids if not os.path.exists(os.path.join(out, f"{i:03d}_s{S - 1}.png"))]
        if not todo:
            return 0
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(root, cfg["name"], "config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
        self.set_adapter(cfg.get("adapter"), cfg.get("scale", 1.0))
        blk = cfg.get("tf_block")
        self.ensure_hook(blk)
        K, band = int(cfg.get("n_dir", 8)), float(cfg.get("band", 2.0))
        n = 0
        for i in todo:
            ctx = load_context(ctx_dir(), i, texts[i]).to(self.device)
            if blk is not None:
                rec = load_direction_file(direction_file(C.data_dir(f"directions_k{C.DIRECTION_STEP}", f"b{blk:02d}"), i))
                assert rec["prompt"] == texts[i] and rec["block"] == blk
                self.tf["U"], self.tf["s"] = rec["U"][:K].to(self.device), rec["s"][:K].float().to(self.device)
                del rec
            else:
                self.tf["U"] = None
            for s0 in range(0, S, B):
                ss = list(range(s0, min(S, s0 + B)))
                lat = torch.stack([eval_latent(i, s) for s in ss])
                alpha = tf_alphas_batch(i, ss, k=K, a=band) if blk is not None else None
                for s, im in zip(ss, self.g.decode(self.sample(ctx, lat, alpha))):
                    save_png(im, os.path.join(out, f"{i:03d}_s{s}.png"))
                n += len(ss)
        return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="JSON list of configurations")
    ap.add_argument("--prompts", default=D.GENEVAL_553)
    ap.add_argument("--prompt-ids", default="0,11,17,27,28,34,63")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out-root", default=None, help="default <REDIVERSE_OUT>/krea2/sweep")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    with open(a.spec) as f:
        cfgs = json.load(f)
    # group by adapter so one shard reloads as few checkpoints as possible
    mine = [c for k, c in enumerate(cfgs) if k % a.num_shards == a.shard]
    mine.sort(key=lambda c: (c.get("adapter") or "", c["name"]))
    texts = D.read_prompts(a.prompts)
    ids = D.parse_ids(a.prompt_ids)
    root = a.out_root or C.output_dir("sweep")
    ensure_contexts({i: texts[i] for i in ids}, ctx_dir(), a.device)
    sw = Sweeper(a.device)
    print(f"[sweep] shard {a.shard}/{a.num_shards}: {len(mine)} configs x {len(ids)} prompts x {a.num_samples}", flush=True)
    for c in mine:
        t0 = time.time()
        n = sw.run(c, texts, ids, a.num_samples, a.batch, root)
        print(f"[sweep] {c['name']}: {n} images in {time.time() - t0:.0f}s  {json.dumps(c)}", flush=True)
    gaps = [(c["name"], i) for c in mine for i in ids
            if not os.path.exists(os.path.join(root, c["name"], "images", f"{i:03d}_s{a.num_samples - 1}.png"))]
    if gaps:
        raise SystemExit(f"[sweep] INCOMPLETE: {gaps[:5]}")
    print(f"[sweep] shard {a.shard}/{a.num_shards} done", flush=True)


if __name__ == "__main__":
    main()

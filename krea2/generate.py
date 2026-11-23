#!/usr/bin/env python
"""Krea 2 generation for the five arms: teacher, student, rediverse_ft, rediverse_tf, rediverse.

  teacher       Krea-2-Raw, 52 FlowMatchEuler steps (mu from resolution), Krea guidance 3.5 (= CFG 4.5)
  student       Krea-2-Turbo, 8 steps, mu 1.15, no guidance
  rediverse_ft  student + the ReDiverse-FT adapter, active in the first forward only (as in the H3 port, whose
                joint text/video stream is the closest analog of Krea's single stream)
  rediverse_tf  student + ReDiverse-TF: the block-l image rows of the first forward get sum_k alpha_k s_k u_k,
                alpha_k ~ U[-band, band] (8 teacher directions of the prompt)
  rediverse     both, in the first forward

A sampling loop equivalent to Krea2Pipeline (same sigmas, Euler updates in fp32, same VAE decode), run on the
compacted text contexts of krea2.model. Sample s of GenEval prompt i starts from the packed latent
randn([4096, 64], CPU generator seeded i * 1000 + s) in every arm and gets alpha from seed 7_000_000 + i * 1009 + s
in both TF arms (recorded in <out-dir>/scales/p{i}.json). Images: <out-dir>/images/{i:03d}_s{s}.png; workers take
disjoint shards (i % num_shards); a requeued worker redoes its own missing images and exits non-zero on gaps.

  python -m krea2.generate --arm student --prompt-ids prompts/geneval_subset100.txt --num-samples 16
  python -m krea2.generate --arm rediverse_tf --block 14 --prompt-ids 0-9 --num-samples 8 --out-dir ...
  python -m krea2.generate --check-pipeline           # custom loop vs Krea2Pipeline on one prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from krea2 import config as C
from krea2 import model as M
from krea2.estimate_directions import ensure_contexts, load_context, ctx_dir
from rediverse import data as D
from rediverse.directions import direction_file, load_direction_file, tf_offset
from rediverse.seeds import alpha_record, cpu_randn, image_eval_seed, tf_alphas_batch

ARMS = ("teacher", "student", "rediverse_ft", "rediverse_tf", "rediverse")
FT_ARMS = ("rediverse_ft", "rediverse")
TF_ARMS = ("rediverse_tf", "rediverse")
BATCH = dict(teacher=4, student=8, rediverse_ft=8, rediverse_tf=8, rediverse=8)


def eval_latent(i: int, s: int) -> torch.Tensor:
    return cpu_randn(image_eval_seed(i, s), C.LATENT_SHAPE)


def uncond_context(device) -> torch.Tensor:
    f = C.data_dir("uncond.pt")
    if not os.path.exists(f):
        os.makedirs(os.path.dirname(f), exist_ok=True)
        c = M.encode_many([""], device)[0]
        torch.save(dict(ctx=c, prompt=""), f + f".tmp{os.getpid()}")
        os.replace(f + f".tmp{os.getpid()}", f)
    return torch.load(f, map_location="cpu")["ctx"]


class Generator:
    def __init__(self, arm: str, device, adapter: str = None, block: int = C.ANCHOR_BLOCK, lora_last: int = None,
                 guidance: float = None, steps: int = None, rank: int = 32):
        assert arm in ARMS, arm
        self.arm, self.device = arm, device
        self.role = "teacher" if arm == "teacher" else "student"
        self.guidance = C.GUIDANCE[self.role] if guidance is None else guidance
        self.sigmas = M.schedule(self.role)["sigmas"]
        if steps is not None:
            assert steps == C.STEPS[self.role], "custom step counts not supported"
        self.model = M.load_transformer(self.role, device)
        self.vae = M.load_vae(device)
        self.ft = arm in FT_ARMS
        if self.ft:
            assert adapter, "--adapter is required for the FT arms"
            self.lora_info = M.load_lora(self.model, adapter, lora_last)
        self.tf = dict(U=None, s=None, alpha=None, step=-1)
        if arm in TF_ARMS:
            assert 0 <= block < C.N_BLOCKS, "TF arms need --block"
            self.block = block

            def hook(_m, _i, out):
                if self.tf["step"] != 0 or self.tf["U"] is None:
                    return None
                delta = tf_offset(self.tf["alpha"], self.tf["s"], self.tf["U"])            # [B, D]
                o = out.clone()
                o[:, -C.N_IMG_TOKENS:] += delta.reshape(out.shape[0], C.N_IMG_TOKENS, C.D_MODEL).to(out.dtype)
                return o
            self.model.transformer_blocks[block].register_forward_hook(hook)
        lm = torch.tensor(self.vae.config.latents_mean).view(1, 16, 1, 1, 1)
        ls = torch.tensor(self.vae.config.latents_std).view(1, 16, 1, 1, 1)
        self._lat_mean, self._lat_std = lm.to(device, C.DTYPE), ls.to(device, C.DTYPE)

    def set_directions(self, U, s):
        self.tf["U"], self.tf["s"] = U.to(self.device), s.to(self.device)

    @torch.no_grad()
    def sample(self, ctx, uctx, latents, alpha=None) -> torch.Tensor:
        """Packed final latents [B, 4096, 64] from initial latents of one prompt."""
        x = latents.to(self.device, C.DTYPE)
        if self.tf["U"] is not None:
            assert alpha is not None and alpha.shape[0] == x.shape[0]
            self.tf["alpha"] = alpha.to(self.device)
        sg = self.sigmas
        for j in range(len(sg) - 1):
            self.tf["step"] = j
            if self.ft:
                (self.model.enable_adapters if j == 0 else self.model.disable_adapters)()
            v = M.forward(self.model, x, ctx, sg[j])
            if self.guidance > 0:
                vu = M.forward(self.model, x, uctx, sg[j])
                v = v + self.guidance * (v - vu)
            x = M.euler_step(x, v, sg[j], sg[j + 1])
        self.tf["step"] = -1
        return x

    @torch.no_grad()
    def decode(self, x) -> list[Image.Image]:
        """Krea2Pipeline._unpack_latents + de-normalisation + VAE decode, one image at a time."""
        b, p, g = x.shape[0], C.PATCH, C.GRID
        lat = x.view(b, g, g, 16, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, 16, 1, g * p, g * p)
        lat = lat * self._lat_std + self._lat_mean
        out = []
        for k in range(b):
            img = self.vae.decode(lat[k:k + 1].to(self.vae.dtype), return_dict=False)[0][:, :, 0]
            arr = (img[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            out.append(Image.fromarray((arr * 255).round().astype("uint8")))
        return out


def save_png(img: Image.Image, path: str):
    tmp = path + f".tmp{os.getpid()}.png"
    img.save(tmp, "PNG")
    os.replace(tmp, path)


def check_pipeline(device):
    """Custom loop (student, compacted context) vs Krea2Pipeline (masked context) from the same latents."""
    from diffusers import Krea2Pipeline
    prompt = "a photo of a dog"
    pipe = Krea2Pipeline.from_pretrained(M.snapshot(C.STUDENT_ID, C.STUDENT_REVISION), torch_dtype=C.DTYPE).to(device)
    lat = torch.stack([eval_latent(27, s) for s in range(2)]).to(device, C.DTYPE)
    ref = pipe([prompt] * lat.shape[0], latents=lat, num_inference_steps=8, guidance_scale=0.0,
               output_type="latent").images
    # strict check: the same Euler loop on the pipeline's own masked (uncompacted) inputs, batch 1
    with torch.no_grad():
        ref1 = pipe([prompt], latents=lat[:1], num_inference_steps=8, guidance_scale=0.0, output_type="latent").images
        pe, pm = pipe.encode_prompt(prompt=[prompt], device=device, max_sequence_length=C.MAX_SEQ_LEN)
        pos = pipe.prepare_position_ids(pe.shape[1], C.GRID, C.GRID, device)
        sg, x = M.schedule("student")["sigmas"], lat[:1].clone()
        for j in range(len(sg) - 1):
            t = torch.tensor([sg[j]], device=device).to(x.dtype)
            v = pipe.transformer(hidden_states=x, encoder_hidden_states=pe, timestep=t, position_ids=pos,
                                 encoder_attention_mask=pm, return_dict=False)[0]
            dt = torch.tensor(sg[j + 1], dtype=torch.float32) - torch.tensor(sg[j], dtype=torch.float32)
            x = (x.float() + dt.to(device) * v).to(x.dtype)          # scheduler arithmetic (bf16 product)
    rel1 = ((ref1.float() - x.float()).norm() / ref1.float().norm()).item()
    floor = ((ref[:1].float() - ref1.float()).norm() / ref1.float().norm()).item()
    print(f"[check] noise floor, pipeline B=2 vs B=1 on the same latent: rel diff {floor:.3e}", flush=True)
    print(f"[check] masked custom loop vs pipeline (B=1): rel diff {rel1:.3e} (expect ~0: same inputs, same kernels)",
          flush=True)
    ctx = M.encode(pipe, prompt, device)
    del pipe
    torch.cuda.empty_cache()
    g = Generator("student", device)
    mine = g.sample(ctx, None, lat)
    rel = ((ref.float() - mine.float()).norm() / ref.float().norm()).item()
    print(f"[check] final latents rel diff custom-loop vs pipeline: {rel:.3e}", flush=True)
    imgs = g.decode(torch.cat([ref, mine]))
    w = imgs[0].width // 2
    grid = Image.new("RGB", (w * len(imgs), w))
    for k, im in enumerate(imgs):
        grid.paste(im.resize((w, w)), (k * w, 0))
    os.makedirs(C.output_dir("checks"), exist_ok=True)
    grid.save(C.output_dir("checks", "pipeline_vs_loop.jpg"), quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, default="student")
    ap.add_argument("--prompts", default=D.GENEVAL_553)
    ap.add_argument("--prompt-ids", default=D.GENEVAL_SUBSET100)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--out-dir", default=None,
                    help="default <REDIVERSE_OUT>/krea2/<arm>[_b<block>][_<adapter run>]")
    ap.add_argument("--adapter", default=None, help="ReDiverse-FT adapter .safetensors (FT arms)")
    ap.add_argument("--lora-last", type=int, default=None, help="last LoRA block (default --block)")
    ap.add_argument("--block", type=int, default=C.ANCHOR_BLOCK, help="collapsed block l (TF injection site)")
    ap.add_argument("--directions-dir", default=None, help="default <REDIVERSE_DATA>/krea2/directions_k5/b{block}")
    ap.add_argument("--band", type=float, default=2.0, help="alpha ~ U[-band, band]")
    ap.add_argument("--n-dir", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=None, help="Krea convention (teacher default 3.5)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--check-pipeline", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.check_pipeline:
        check_pipeline(a.device)
        return
    texts = D.read_prompts(a.prompts)
    ids = [i for i in D.parse_ids(a.prompt_ids) if i % a.num_shards == a.shard]
    name = a.arm
    if a.arm in TF_ARMS:
        name += f"_b{a.block:02d}"
    if a.arm in FT_ARMS and a.adapter:
        name += "_" + os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(a.adapter))))
    out_dir = a.out_dir or C.output_dir(name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    S = a.num_samples
    todo = [i for i in ids if not os.path.exists(os.path.join(img_dir, f"{i:03d}_s{S - 1}.png"))]
    print(f"[krea2-gen] arm {a.arm}: {len(todo)}/{len(ids)} prompts x {S} -> {out_dir}", flush=True)
    if not todo:
        return
    ddir = a.directions_dir or C.data_dir(f"directions_k{C.DIRECTION_STEP}", f"b{a.block:02d}")
    if a.arm in TF_ARMS:
        miss = [i for i in todo if not os.path.exists(direction_file(ddir, i))]
        if miss:
            sys.exit(f"{len(miss)} direction files missing in {ddir} (first p{miss[0]:03d}); run krea2.estimate_directions")
    ensure_contexts({i: texts[i] for i in todo}, ctx_dir(), a.device)
    uctx = uncond_context(a.device).to(a.device) if a.arm == "teacher" else None
    gen = Generator(a.arm, a.device, adapter=a.adapter, block=a.block, lora_last=a.lora_last, guidance=a.guidance)
    meta = dict(arm=a.arm, guidance=gen.guidance, sigmas=gen.sigmas, block=a.block, adapter=a.adapter,
                band=a.band, n_dir=a.n_dir, directions_dir=ddir if a.arm in TF_ARMS else None, res=C.RES)
    with open(os.path.join(out_dir, f"meta_{os.getpid()}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    B = a.batch or BATCH[a.arm]
    t0, n_img = time.time(), 0
    sc_dir = os.path.join(out_dir, "scales")
    for i in todo:          # shards are disjoint (i % num_shards), so no claims: a requeued task redoes its own gaps
        scales = {}
        ctx = load_context(ctx_dir(), i, texts[i]).to(a.device)
        if a.arm in TF_ARMS:
            rec = load_direction_file(direction_file(ddir, i))
            assert rec["prompt"] == texts[i] and rec["block"] == a.block and rec["k"] == C.DIRECTION_STEP \
                and tuple(rec["shape"]) == (C.N_IMG_TOKENS, C.D_MODEL) and rec["U"].shape[0] >= a.n_dir, \
                (i, rec["prompt"], rec["block"], rec["k"], rec["shape"])
            gen.set_directions(rec["U"][:a.n_dir], rec["s"][:a.n_dir].float())
            del rec
        for s0 in range(0, S, B):
            ss = list(range(s0, min(S, s0 + B)))
            if all(os.path.exists(os.path.join(img_dir, f"{i:03d}_s{s}.png")) for s in ss):
                continue
            lat = torch.stack([eval_latent(i, s) for s in ss])
            alpha = tf_alphas_batch(i, ss, k=a.n_dir, a=a.band) if a.arm in TF_ARMS else None
            imgs = gen.decode(gen.sample(ctx, uctx, lat, alpha))
            for s, im in zip(ss, imgs):
                save_png(im, os.path.join(img_dir, f"{i:03d}_s{s}.png"))
                if alpha is not None:
                    scales[f"{i:03d}_s{s}"] = alpha_record(alpha[ss.index(s)])
            n_img += len(ss)
        if a.arm in TF_ARMS:        # coefficients of every sample (deterministic seed rule), per prompt, atomic
            os.makedirs(sc_dir, exist_ok=True)
            al = tf_alphas_batch(i, list(range(S)), k=a.n_dir, a=a.band)
            f = os.path.join(sc_dir, f"p{i:03d}.json")
            with open(f + f".tmp{os.getpid()}", "w") as fh:
                json.dump({f"{i:03d}_s{s}": alpha_record(al[s]) for s in range(S)}, fh, indent=0)
            os.replace(f + f".tmp{os.getpid()}", f)
        el = time.time() - t0
        print(f"  p{i:03d} done | {el / max(n_img, 1):.2f} s/img | {n_img} imgs", flush=True)
    gaps = [f"{i:03d}_s{s}" for i in ids for s in range(S) if not os.path.exists(os.path.join(img_dir, f"{i:03d}_s{s}.png"))]
    if gaps:
        sys.exit(f"[krea2-gen] shard {a.shard}/{a.num_shards} INCOMPLETE: {len(gaps)} images missing, e.g. {gaps[:3]}")
    print(f"[krea2-gen] shard {a.shard}/{a.num_shards} done", flush=True)


if __name__ == "__main__":
    main()

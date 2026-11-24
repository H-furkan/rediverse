#!/usr/bin/env python
"""Qwen-Image generation for the five arms: teacher, student, rediverse_ft, rediverse_tf, rediverse.

  teacher       the family's base model on its own schedule (q21: 40 steps, no CFG; q20: 50 steps, true CFG 4)
  student       --student <name> on its card schedule (config.STUDENTS)
  rediverse_ft  student + the ReDiverse-FT adapter, active in the first forward only
  rediverse_tf  student + ReDiverse-TF: block-l image rows of the first forward get sum_k alpha_k s_k u_k,
                alpha_k ~ U[-band, band] (8 teacher directions of the prompt, forward tau)
  rediverse     both, in the first forward

A sampling loop equivalent to the diffusers pipelines (same sigmas and timesteps, the scheduler-exact Euler update by
default (model.EULER), no prefix KV cache; check.py shows it is bit-identical to the pipeline given the same contexts). Sample s of GenEval prompt i starts from randn([4096, 64], CPU
generator seeded i * 1000 + s) in every arm and family, and gets alpha from seed 7_000_000 + i * 1009 + s in the TF
arms. Images: <out-dir>/images/{i:03d}_s{s}.png; workers take disjoint shards (i % num_shards).

  python -m qwenimage.generate --family q21 --arm teacher --prompt-ids 0,11,17,27,28,34 --num-samples 8
  python -m qwenimage.generate --student lightning --arm student --prompt-ids 0,11,17,27,28,34 --num-samples 8
  python -m qwenimage.generate --student lightning --arm rediverse --adapter <ft.safetensors> --block 20 --tau 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from qwenimage import config as C
from qwenimage import model as M
from rediverse import data as D
from rediverse.directions import direction_file, load_direction_file, tf_offset
from rediverse.seeds import alpha_record, cpu_randn, image_eval_seed, tf_alphas_batch

ARMS = ("teacher", "student", "rediverse_ft", "rediverse_tf", "rediverse")
FT_ARMS = ("rediverse_ft", "rediverse")
TF_ARMS = ("rediverse_tf", "rediverse")


def eval_latent(fam: str, i: int, s: int) -> torch.Tensor:
    return cpu_randn(image_eval_seed(i, s), C.family(fam)["latent_shape"])


# ------------------------------------------------------------------ contexts --
def ctx_dir(fam: str) -> str:
    """Cached contexts by prompt index (QWEN_CTX_DIR overrides it for another prompt file)."""
    return os.environ.get("QWEN_CTX_DIR") or C.data_dir(fam, "geneval_ctx")


def ensure_contexts(fam: str, texts: dict, cdir: str, device) -> None:
    os.makedirs(cdir, exist_ok=True)
    missing = [i for i in sorted(texts) if not os.path.exists(os.path.join(cdir, f"p{i:03d}.pt"))]
    if not missing:
        return
    for i, c in zip(missing, M.encode_many(fam, [texts[i] for i in missing], device)):
        f = os.path.join(cdir, f"p{i:03d}.pt")
        torch.save(dict(ctx=c, prompt=texts[i], index=i, family=fam), f + f".tmp{os.getpid()}")
        os.replace(f + f".tmp{os.getpid()}", f)


def load_context(cdir: str, i: int, text: str) -> torch.Tensor:
    c = torch.load(os.path.join(cdir, f"p{i:03d}.pt"), map_location="cpu")
    assert c["prompt"] == text, f"ctx p{i:03d} holds {c['prompt']!r}, expected {text!r}"
    return c["ctx"]


def negative_context(fam: str, device) -> torch.Tensor:
    """The CFG teacher's negative prompt context (q20: " "; q2512: the card's negative prompt; q21, which samples
    without guidance, gets the pipelines' empty-negative convention " " when guidance is forced on)."""
    neg = C.family(fam)["teacher"].get("negative", " ")
    f = C.data_dir(fam, "negative.pt")
    if not os.path.exists(f):
        os.makedirs(os.path.dirname(f), exist_ok=True)
        torch.save(dict(ctx=M.encode_many(fam, [neg], device)[0], prompt=neg), f + f".tmp{os.getpid()}")
        os.replace(f + f".tmp{os.getpid()}", f)
    rec = torch.load(f, map_location="cpu")
    assert rec["prompt"] == neg
    return rec["ctx"]


# ------------------------------------------------------------------ sampling --
class Generator:
    """One loaded transformer (+ VAE) that samples any arm of one model: the teacher, or a student with optional FT
    adapter and TF hook."""

    def __init__(self, who: str, device, adapter: str = None, block: int = None, decode: bool = True,
                 teacher_cfg: float = None):
        self.who, self.device = who, device
        self.fam = M.who_family(who)
        self.f = C.family(self.fam)
        self.is_teacher = who.startswith("teacher")
        self.cfg = self.f["teacher"]["cfg"] if self.is_teacher else C.student(who)["cfg"]
        if self.is_teacher and teacher_cfg is not None:
            self.cfg = float(teacher_cfg)
        self.sigmas = M.schedule(who)["sigmas"]
        self.model = M.load_transformer(who, device)
        self.base = M.base_adapters(self.model)
        self.decoder = M.Decoder(self.fam, device) if decode else None
        self.ft = None
        if adapter:
            assert not self.is_teacher
            self.ft = M.load_lora(self.fam, self.model, adapter)
        self.use_ft = False
        self.ft_scale = 1.0
        self.tf = dict(U=None, s=None, alpha=None, step=-1, steps=1)
        self.block = block
        if block is not None:
            assert 0 <= block < self.f["n_blocks"]

            def hook(_m, _i, out):
                if self.tf["U"] is None or not (0 <= self.tf["step"] < self.tf["steps"]):
                    return None
                delta = tf_offset(self.tf["alpha"], self.tf["s"], self.tf["U"])
                return M.add_to_image_rows(self.fam, out, delta.reshape(-1, self.f["n_img"], self.f["d_model"]))
            self.model.transformer_blocks[block].register_forward_hook(hook)
        self.neg = negative_context(self.fam, device).to(device) if self.is_teacher and self.cfg > 1 else None
        self.provenance = M.provenance(who)
        if self.is_teacher and teacher_cfg is not None:          # an overridden teacher guidance is recorded as run
            self.provenance.update(cfg=self.cfg, negative=C.family(self.fam)["teacher"].get("negative", " ")
                                   if self.cfg > 1 else None)

    def set_directions(self, U, s):
        self.tf["U"], self.tf["s"] = (None, None) if U is None else (U.to(self.device), s.to(self.device))

    @torch.no_grad()
    def sample(self, ctx, latents, alpha=None, ft_steps: int = 1, tf_steps: int = 1, euler: str = None,
               trace: list = None) -> torch.Tensor:
        """Final latents from initial latents of one prompt. `trace` (a list) receives the first forward's velocity
        and the latents after every step (parity checks); `euler` overrides model.EULER."""
        x = latents.to(self.device, C.DTYPE)
        b = x.shape[0]
        if self.tf["U"] is not None:
            assert alpha is not None and alpha.shape[0] == b
            self.tf["alpha"] = alpha.to(self.device)
        self.tf["steps"] = tf_steps
        ctx = ctx.to(self.device)
        for j in range(len(self.sigmas) - 1):
            self.tf["step"] = j
            if self.ft is not None:
                M.set_active(self.model, self.base + ([M.FT_ADAPTER] if (self.use_ft and j < ft_steps) else []),
                             {M.FT_ADAPTER: self.ft_scale})
            t = M.model_timestep(self.who, j, b, self.device)
            v = M.velocity(self.fam, self.model, x, ctx, t, self.neg, self.cfg)
            x = M.step(x, v, self.who, j, euler)
            if trace is not None:
                if j == 0:
                    trace.append(("v0", v.detach().clone()))
                trace.append((f"x{j + 1}", x.detach().clone()))
        self.tf["step"] = -1
        if self.ft is not None:
            M.set_active(self.model, self.base)
        return x


def save_png(img, path: str):
    tmp = path + f".tmp{os.getpid()}.png"
    img.save(tmp, "PNG")
    os.replace(tmp, path)


def default_out(fam: str, who: str, arm: str, tag: str = "") -> str:
    return C.output_dir(fam, "teacher" if arm == "teacher" else who, arm + tag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default=None, help="q21 | q20 (teacher arm; else taken from --student)")
    ap.add_argument("--student", default=None, help="config.STUDENTS key")
    ap.add_argument("--arm", choices=ARMS, default="student")
    ap.add_argument("--prompts", default=D.GENEVAL_553)
    ap.add_argument("--prompt-ids", default=D.GENEVAL_SUBSET100)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="", help="suffix of the default output folder")
    ap.add_argument("--adapter", default=None, help="ReDiverse-FT adapter .safetensors (FT arms)")
    ap.add_argument("--block", type=int, default=None, help="collapsed block l (TF site)")
    ap.add_argument("--tau", type=int, default=None, help="teacher forward of the TF directions")
    ap.add_argument("--allow-tau-mismatch", action="store_true", help="ablations: tau other than the sigma match")
    ap.add_argument("--teacher-cfg", type=float, default=None, help="diagnostics: override the teacher's true CFG")
    ap.add_argument("--ft-scale", type=float, default=1.0, help="FT adapter scale (= lora_B x scale)")
    ap.add_argument("--directions-dir", default=None)
    ap.add_argument("--band", type=float, default=2.0)
    ap.add_argument("--n-dir", type=int, default=8)
    ap.add_argument("--ft-steps", type=int, default=1)
    ap.add_argument("--tf-steps", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.arm == "teacher":
        fam = a.family or C.family_of(a.student)
        who = f"teacher:{fam}"
    else:
        assert a.student, "--student is required for the student arms"
        fam, who = C.family_of(a.student), a.student
    site = C.site(a.student) if a.student else {}
    block = a.block if a.block is not None else site.get("block")
    tau = a.tau if a.tau is not None else site.get("tau")
    if a.arm in TF_ARMS:
        assert block is not None and tau is not None, "TF arms need --block and --tau (or config.SITES)"
        tsig = M.teacher_target_sigmas(a.student, tau)
        assert a.allow_tau_mismatch or tau == tsig["nearest_k"], \
            f"--tau {tau} but the teacher forward landing nearest {a.student}'s sigma after step 1 is {tsig['nearest_k']}"
    if a.arm in FT_ARMS:
        assert a.adapter, "--adapter is required for the FT arms"
    texts = D.read_prompts(a.prompts)
    ids = [i for i in D.parse_ids(a.prompt_ids) if i % a.num_shards == a.shard]
    out_dir = a.out_dir or default_out(fam, who, a.arm, a.tag)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    S = a.num_samples
    todo = [i for i in ids if not os.path.exists(os.path.join(img_dir, f"{i:03d}_s{S - 1}.png"))]
    print(f"[qwen-gen] {who} arm {a.arm}: {len(todo)}/{len(ids)} prompts x {S} -> {out_dir}", flush=True)
    if not todo:
        return
    ddir = a.directions_dir or (C.data_dir(fam, f"directions_k{tau}", f"b{block:02d}") if a.arm in TF_ARMS else None)
    if a.arm in TF_ARMS:
        miss = [i for i in todo if not os.path.exists(direction_file(ddir, i))]
        if miss:
            sys.exit(f"{len(miss)} direction files missing in {ddir} (first p{miss[0]:03d})")
    cdir = ctx_dir(fam)
    ensure_contexts(fam, {i: texts[i] for i in todo}, cdir, a.device)
    gen = Generator(who, a.device, adapter=a.adapter if a.arm in FT_ARMS else None,
                    block=block if a.arm in TF_ARMS else None, teacher_cfg=a.teacher_cfg)
    gen.use_ft = a.arm in FT_ARMS
    gen.ft_scale = a.ft_scale
    meta = dict(who=who, arm=a.arm, sigmas=gen.sigmas, block=block, tau=tau, adapter=a.adapter, **gen.provenance,
                ft_scale=a.ft_scale,
                band=a.band, n_dir=a.n_dir, ft_steps=a.ft_steps, tf_steps=a.tf_steps, directions_dir=ddir,
                res=C.RES, student_lora=gen.model._qwen_info.get("student_lora"))
    with open(os.path.join(out_dir, f"meta_{os.getpid()}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    B = a.batch
    t0, n_img = time.time(), 0
    sc_dir = os.path.join(out_dir, "scales")
    for i in todo:
        ctx = load_context(cdir, i, texts[i])
        if a.arm in TF_ARMS:
            rec = load_direction_file(direction_file(ddir, i))
            assert rec["prompt"] == texts[i] and rec["block"] == block and rec["k"] == tau and \
                rec["U"].shape[0] >= a.n_dir, (i, rec["prompt"], rec["block"], rec["k"])
            assert abs(float(rec["sigma_out"]) - tsig["sigma_out"]) < 1e-6, (rec["sigma_out"], tsig)
            M.check_provenance(rec, M.provenance(f"teacher:{fam}", guided=False), direction_file(ddir, i))
            gen.set_directions(rec["U"][:a.n_dir], rec["s"][:a.n_dir].float())
            del rec
        for s0 in range(0, S, B):
            ss = list(range(s0, min(S, s0 + B)))
            if all(os.path.exists(os.path.join(img_dir, f"{i:03d}_s{s}.png")) for s in ss):
                continue
            lat = torch.stack([eval_latent(fam, i, s) for s in ss])
            alpha = tf_alphas_batch(i, ss, k=a.n_dir, a=a.band) if a.arm in TF_ARMS else None
            imgs = gen.decoder(gen.sample(ctx, lat, alpha, a.ft_steps, a.tf_steps))
            for s, im in zip(ss, imgs):
                save_png(im, os.path.join(img_dir, f"{i:03d}_s{s}.png"))
            n_img += len(ss)
        if a.arm in TF_ARMS:
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
        sys.exit(f"[qwen-gen] shard {a.shard}/{a.num_shards} INCOMPLETE: {len(gaps)} images missing, e.g. {gaps[:3]}")
    print(f"[qwen-gen] shard {a.shard}/{a.num_shards} done", flush=True)


if __name__ == "__main__":
    main()

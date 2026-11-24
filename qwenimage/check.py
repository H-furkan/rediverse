#!/usr/bin/env python
"""Pre-flight checks for a Qwen-Image family (one GPU).

  python -m qwenimage.check --family q21 [--students lightning,viggle,turbo,pq,v2rl] [--pipeline] [--te-equal]

  schedules      sigmas / timesteps of the teacher and each student, and tau = the teacher forward landing closest
                 to the student's sigma after step 1
  LoRA           every student adapter loads (modules touched vs adapter keys)
  truncation     forward_to_blocks == hooks on the full forward
  pipeline       the custom sampling loop vs the diffusers pipeline (card settings, with its KV cache) from the same
                 latents, for the teacher and each student: the pipeline's batch-1 vs batch-2 noise floor, then our
                 loop with the fp32 and the scheduler-exact Euler update (first forward, every step, final latents)
  te-equal       q21: Qwen-Image-2.1-Turbo's text encoder vs the base one, tensor by tensor
  timing         seconds per forward at batch 1 and 4
Writes <out>/<backbone>/checks/check_<family>.json and pipeline_vs_loop_<who>.jpg.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from qwenimage import config as C
from qwenimage import model as M

PROMPT = "a photo of a dog"


def schedules(fam, students):
    out = {"teacher": M.schedule(f"teacher:{fam}")}
    for s in students:
        out[s] = M.schedule(s)
        out[s]["tau"] = M.teacher_target_sigmas(s)
    return {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_t")} for k, v in out.items()}


def te_equal(fam):
    from safetensors import safe_open
    import glob
    base = M.base_dir(fam)
    tur = M.snapshot(C.student("turbo")["repo"], C.student("turbo")["revision"])

    def index(d):
        idx = {}
        for f in sorted(glob.glob(os.path.join(d, "text_encoder", "*.safetensors"))):
            with safe_open(f, "pt") as fh:
                for k in fh.keys():
                    idx[k] = f
        return idx
    ib, it = index(base), index(tur)
    common = sorted(set(ib) & set(it))
    diff, maxd = [], 0.0
    handles = {}

    def get(f, k):
        if f not in handles:
            handles[f] = safe_open(f, "pt")
        return handles[f].get_tensor(k)
    for k in common:
        a, b = get(ib[k], k), get(it[k], k)
        if a.shape != b.shape or not torch.equal(a.to(torch.float32), b.to(torch.float32)):
            d = float((a.float() - b.float()).abs().max()) if a.shape == b.shape else float("inf")
            diff.append(k)
            maxd = max(maxd, d)
    return dict(n_base=len(ib), n_turbo=len(it), n_common=len(common), only_base=sorted(set(ib) - set(it))[:5],
                only_turbo=sorted(set(it) - set(ib))[:5], n_diff=len(diff), diff=diff[:10], max_abs_diff=maxd)


def full_weights_check(fam: str, s: str) -> dict:
    """A full-weights student against its family base, from the safetensors headers and a few tensors."""
    import glob
    from safetensors import safe_open
    repo, sub = C.student_weights(s)
    sd = M.snapshot(repo, C.student(s)["revision"]) + (f"/{sub}" if sub else "")
    bd = M.base_dir(fam) + "/transformer"

    def keys(d):
        """Tensor names of the diffusers weights in d (the index's shards, or diffusion_pytorch_model*.safetensors), so
        other files in a repo root (e.g. v2-RL's LoRA export) are not mixed in."""
        out = {}
        idx = glob.glob(f"{d}/diffusion_pytorch_model*.safetensors.index.json")
        files = sorted({f"{d}/{v}" for v in json.load(open(idx[0]))["weight_map"].values()}) if idx else \
            sorted(glob.glob(f"{d}/diffusion_pytorch_model*.safetensors"))
        assert files, f"no diffusers weights in {d}"
        for f in files:
            with safe_open(f, "pt") as fh:
                for k in fh.keys():
                    out[k] = f
        return out
    ks, kb = keys(sd), keys(bd)
    cs, cb = (json.load(open(f"{d}/config.json")) for d in (sd, bd))
    strip = lambda c: {k: v for k, v in c.items() if not k.startswith("_") and k != "pooled_projection_dim"}
    n = C.family(fam)["n_blocks"]
    probe = [k for k in (f"transformer_blocks.{i}.attn.to_q.weight" for i in (0, n // 2, n - 1)) if k in ks and k in kb]
    diffs = {}
    for k in probe:
        with safe_open(ks[k], "pt") as f1, safe_open(kb[k], "pt") as f2:
            x, y = f1.get_tensor(k).float(), f2.get_tensor(k).float()
        diffs[k] = float((x - y).norm() / y.norm())
    return dict(dir=sd, same_config=strip(cs) == strip(cb), config_diff={k: (cs.get(k), cb.get(k)) for k in
                set(strip(cs)) | set(strip(cb)) if cs.get(k) != cb.get(k)}, same_keys=set(ks) == set(kb),
                n_keys=len(ks), only_student=sorted(set(ks) - set(kb))[:5], only_base=sorted(set(kb) - set(ks))[:5],
                rel_diff_vs_base=diffs, n_differ=sum(v > 0 for v in diffs.values()))


def _rel(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


class _FirstOutput:
    """Forward hook keeping the first output (image rows) of a transformer."""

    def __init__(self, fam, model):
        self.fam, self.out = fam, None
        self.h = model.register_forward_hook(self)

    def __call__(self, _m, _i, o):
        if self.out is None:
            o = o[0] if isinstance(o, tuple) else (o.sample if hasattr(o, "sample") else o)
            self.out = o[:, -C.family(self.fam)["n_img"]:].detach().clone()

    def remove(self):
        self.h.remove()


@torch.no_grad()
def pipeline_parity(fam, who, device, ctx, lat):
    """The diffusers pipeline (card call, batch 2) vs our loop from the same latents: the pipeline's own batch-1 vs
    batch-2 noise floor, then our loop with the fp32 Euler update and with the scheduler-exact update; for each, the
    first-forward output and the latents after every step, as relative differences."""
    from qwenimage.generate import Generator
    P = M.pipe_cls(fam)
    kw = {}
    if who.startswith("teacher"):
        pipe = P.from_pretrained(M.base_dir(fam), torch_dtype=C.DTYPE).to(device)
        call = dict(num_inference_steps=C.family(fam)["teacher"]["steps"])
        if C.family(fam)["teacher"]["cfg"] > 1:
            call.update(true_cfg_scale=C.family(fam)["teacher"]["cfg"], negative_prompt=C.family(fam)["teacher"]["negative"])
    else:
        s = C.student(who)
        if s["kind"] == "full":
            repo, sub = C.student_weights(who)
            tr = M.transformer_cls(fam).from_pretrained(M.snapshot(repo, s["revision"]), subfolder=sub, torch_dtype=C.DTYPE)
            pipe = P.from_pretrained(M.base_dir(fam), transformer=tr, torch_dtype=C.DTYPE).to(device)
        else:
            pipe = P.from_pretrained(M.base_dir(fam), torch_dtype=C.DTYPE).to(device)
            pipe.load_lora_weights(M.snapshot(s["repo"], s["revision"]), weight_name=s["file"], adapter_name="card")
        pipe.scheduler = M.make_scheduler(who)
        call = dict(num_inference_steps=len(s["sigmas"]), sigmas=list(s["sigmas"]), true_cfg_scale=1.0)
    if C.arch(fam) == "q21":
        kw["use_kv_cache"] = True

    def run(latents, embeds=None, **extra):
        """Pipeline call; embeds=(ctx, neg) feeds our cached contexts instead of encoding the prompt in a batch."""
        steps = []

        def cb(_p, _i, _t, k):
            steps.append(k["latents"].detach().clone())
            return k
        fo = _FirstOutput(fam, pipe.transformer)
        b = latents.shape[0]
        if embeds is None:
            pk = dict(prompt=[PROMPT] * b)
        else:
            c, n = embeds
            pk = dict(prompt=None, prompt_embeds=c.to(device).expand(b, -1, -1).contiguous())
            if "negative_prompt" in call:
                pk["negative_prompt_embeds"] = n.to(device).expand(b, -1, -1).contiguous()
        cl = {k: v for k, v in call.items() if not (embeds is not None and k == "negative_prompt")}
        try:
            out = pipe(latents=latents.to(device, C.DTYPE), height=C.RES, width=C.RES, output_type="latent",
                       callback_on_step_end=cb, **pk, **cl, **{**kw, **extra}).images
        finally:
            fo.remove()
        return out, steps, fo.out
    res = {}
    t0 = time.time()
    ref, ref_steps, ref_v0 = run(lat)
    res["pipe_s"] = time.time() - t0
    ref1, ref1_steps, ref1_v0 = run(lat[:1])
    res["floor_pipe_b1_vs_b2"] = dict(first_forward=_rel(ref1_v0, ref_v0[:1]),
                                     per_step=[_rel(a, b[:1]) for a, b in zip(ref1_steps, ref_steps)],
                                     final=_rel(ref1, ref[:1]))
    if C.arch(fam) == "q21":
        ref_nc, _, _ = run(lat, use_kv_cache=False)
        res["kv_cache_vs_none"] = _rel(ref_nc, ref)
    # the same call fed our cached contexts (batch-1 encodings), which removes batched text-encoding differences
    neg = None
    if "negative_prompt" in call:
        from qwenimage.generate import negative_context
        neg = negative_context(fam, device)
    ref_ctx, ref_ctx_steps, ref_ctx_v0 = run(lat, embeds=(ctx, neg))
    res["pipe_own_embeds_vs_ours"] = _rel(ref_ctx, ref)
    res["pipe_timesteps"] = [float(t) for t in pipe.scheduler.timesteps]
    res["timesteps_equal"] = res["pipe_timesteps"] == M.schedule(who)["timesteps"][:len(res["pipe_timesteps"])]
    # our batched text encoding for comparison
    with torch.no_grad():
        if C.arch(fam) == "q21":
            pe2 = pipe.encode_prompt(prompt=[PROMPT, PROMPT], device=device)[0]
        else:
            pe2 = pipe.encode_prompt(prompt=[PROMPT, PROMPT], device=device)[0]
    res["ctx_batch2_vs_batch1"] = _rel(pe2[:1].cpu(), ctx) if pe2.shape[1] == ctx.shape[1] else "length differs"
    ctx_pipe = M.encode(pipe, fam, PROMPT, device)
    res["ctx_vs_cached"] = _rel(ctx_pipe.cpu(), ctx) if ctx_pipe.shape == ctx.shape else \
        f"shape {tuple(ctx_pipe.shape)} vs {tuple(ctx.shape)}"
    del pipe
    torch.cuda.empty_cache()
    g = Generator(who, device)
    finals = {}
    for mode in ("fp32", "scheduler"):
        fo = _FirstOutput(fam, g.model)
        trace = []
        t0 = time.time()
        try:
            mine = g.sample(ctx, lat, euler=mode, trace=trace)
        finally:
            fo.remove()
        res[f"loop_s_{mode}"] = time.time() - t0
        xs = [x for k, x in trace if k.startswith("x")]
        res[f"loop_vs_pipe_{mode}"] = dict(first_forward=_rel(fo.out, ref_v0),
                                           per_step=[_rel(a, b) for a, b in zip(xs, ref_steps)],
                                           final=_rel(mine, ref))
        res[f"loop_vs_pipe_ctx_{mode}"] = dict(first_forward=_rel(fo.out, ref_ctx_v0),
                                               per_step=[_rel(a, b) for a, b in zip(xs, ref_ctx_steps)],
                                               final=_rel(mine, ref_ctx))
        finals[mode] = mine
    # our loop at batch 1 vs the pipeline at batch 1
    mine1 = g.sample(ctx, lat[:1], euler="scheduler")
    res["loop_b1_vs_pipe_b1_scheduler"] = _rel(mine1, ref1)
    # headlines: with the pipeline fed our contexts (the like-for-like comparison) and against its prompt path
    res["final_vs_pipe_own_embeds"] = res[f"loop_vs_pipe_ctx_{M.EULER}"]["final"]
    res["final_vs_pipe_prompt_path"] = res[f"loop_vs_pipe_{M.EULER}"]["final"]
    imgs = g.decoder(torch.cat([ref.to(C.DTYPE), finals["fp32"], finals["scheduler"]]))
    from PIL import Image
    w = 256
    grid = Image.new("RGB", (w * len(imgs), w), "white")
    for k, im in enumerate(imgs):
        grid.paste(im.resize((w, w)), (k * w, 0))
    os.makedirs(C.output_dir(fam, "checks"), exist_ok=True)
    grid.save(C.output_dir(fam, "checks", f"pipeline_vs_loop_{who.replace(':', '_')}.jpg"), quality=90)
    x1 = lat[:1].to(device, C.DTYPE)
    x4 = torch.cat([lat] * (4 // lat.shape[0])).to(device, C.DTYPE) if lat.shape[0] <= 4 else lat[:4].to(device, C.DTYPE)
    for name, x in (("b1", x1), ("b4", x4)):
        t = M.model_timestep(who, 0, x.shape[0], device)
        M.forward(fam, g.model, x, ctx.to(device), t)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            M.forward(fam, g.model, x, ctx.to(device), t)
        torch.cuda.synchronize()
        res[f"forward_s_{name}"] = (time.time() - t0) / 3
    res["trunc_rel"] = M.check_truncation(fam, g.model, ctx.to(device), device)
    res["student_lora"] = g.model._qwen_info.get("student_lora")
    res["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    del g
    torch.cuda.empty_cache()
    return res


def _merge(old: dict, new: dict) -> dict:
    """Recursive dict merge (new wins), so a partial rerun keeps the earlier sections of check_<family>.json."""
    out = dict(old)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", default="q21")
    ap.add_argument("--students", default=None, help="comma list (default: every student of the family)")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--te-equal", action="store_true")
    ap.add_argument("--skip-teacher", action="store_true")
    ap.add_argument("--loras", action="store_true", help="load every LoRA student and count matched modules")
    ap.add_argument("--truncation", action="store_true", help="truncated forward vs hooks (teacher)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    fam = a.family
    studs = a.students.split(",") if a.students else [k for k, v in C.STUDENTS.items() if v["family"] == fam]
    out = dict(family=fam, students=studs)
    import diffusers
    import transformers
    import peft
    out["versions"] = dict(torch=torch.__version__, diffusers=diffusers.__version__,
                           transformers=transformers.__version__, peft=peft.__version__)
    out["schedules"] = schedules(fam, studs)
    for s in studs:
        print(f"[check] {s}: sigmas {[round(x, 4) for x in out['schedules'][s]['sigmas']]} tau {out['schedules'][s]['tau']}",
              flush=True)
    print(f"[check] teacher sigmas[:12] {[round(x, 4) for x in out['schedules']['teacher']['sigmas'][:12]]}", flush=True)
    path = C.output_dir(fam, "checks", f"check_{fam}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def dump():
        prev = json.load(open(path)) if os.path.exists(path) else {}       # re-read: another check may have written
        with open(path + f".tmp{os.getpid()}", "w") as fh:
            json.dump(_merge(prev, out), fh, indent=1, default=str)
        os.replace(path + f".tmp{os.getpid()}", path)
    dump()
    if a.te_equal and fam == "q21":
        out["turbo_text_encoder"] = te_equal(fam)
        print(f"[check] turbo TE vs base: {out['turbo_text_encoder']}", flush=True)
        dump()
    if a.loras:
        # every student loads: LoRA adapters onto the base (module / file-header / non-zero-B counts), merged adapters
        # into the base (pair / replacement counts), full weights against the base (same config and tensor names,
        # weights that actually differ)
        for s in studs:
            kind = C.student(s)["kind"]
            if kind == "full":
                info = full_weights_check(fam, s)
                assert info["same_config"] and info["same_keys"] and info["n_differ"] > 0, info
            m = M.load_transformer(s, a.device)
            if kind != "full":
                info = dict(m._qwen_info["student_lora"], adapters=M.adapter_names(m))
            out.setdefault("loras", {})[s] = dict(info, kind=kind)
            print(f"[check] {kind} {s}: {info}", flush=True)
            del m
            dump()
    if a.truncation:
        ctx = M.encode_many(fam, [PROMPT], a.device)[0]
        m = M.load_transformer(f"teacher:{fam}", a.device)
        out["truncation_teacher"] = M.check_truncation(fam, m, ctx.to(a.device), a.device)
        print(f"[check] truncation (teacher): {out['truncation_teacher']:.3e}", flush=True)
        del m
        dump()
    if a.pipeline:
        ctx = M.encode_many(fam, [PROMPT], a.device)[0]
        from qwenimage.generate import eval_latent
        lat = torch.stack([eval_latent(fam, 27, s) for s in range(2)])
        whos = ([] if a.skip_teacher else [f"teacher:{fam}"]) + studs
        for who in whos:
            try:
                out.setdefault("pipeline", {})[who] = r = pipeline_parity(fam, who, a.device, ctx, lat)
                print(f"[check] {who}: {r}", flush=True)
            except Exception as e:                       # keep going: one broken student must not hide the others
                import traceback
                traceback.print_exc()
                out.setdefault("pipeline", {})[who] = dict(error=f"{type(e).__name__}: {e}")
            dump()
    print(f"[check] wrote {path}", flush=True)


if __name__ == "__main__":
    main()

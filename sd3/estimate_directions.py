"""ReDiverse-TF directions for SD3 (paper Sec. 4.2).

Per prompt, the top-8 principal directions of the teacher's block-12 image stream at teacher
forward 9.

    python -m sd3.estimate_directions                              # all 553 GenEval prompts
    python -m sd3.estimate_directions --shard 0 --num-shards 4     # one of four workers
    python -m sd3.estimate_directions --prompt "a photo of a fox"  # one custom prompt

For GenEval prompt i the teacher is rolled CONDITIONAL-ONLY (no CFG), batch 1, from 25 latents
torch.randn(1, 16, 128, 128) on a CPU generator seeded i * 1000 + j, j = 100..124 (disjoint from the
evaluation samples 0..99): 8 full forwards, then forward 9 hooked at transformer_blocks.12 and
truncated after it. The image stream [4096, 1536] of each latent is flattened (D = 6,291,456) and
rediverse.directions.directions_from_samples returns U [8, D] (bf16), s [8] and the eigenvalues.
Files p{i:03d}.pt (about 100 MB each) go to <REDIVERSE_DATA>/sd3/directions. The conditioning is
the conditional encode_prompt output of each prompt alone, cached in <REDIVERSE_DATA>/sd3/geneval_ctx.
A custom prompt (--prompt) uses the latent seeds of prompt index 0; its file goes to
<REDIVERSE_OUT>/sd3/prompt/<slug>/directions.pt, where sd3.generate --prompt looks for it.
An existing p{i:03d}.pt counts as done only when it holds the directions of prompt i of --prompts.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch

from rediverse import data as D
from rediverse.claims import claim
from rediverse.directions import N_DIR, direction_file, directions_from_samples, matches, save_directions
from rediverse.seeds import N_DIRECTION_SAMPLES, cpu_randn, direction_sample_indices, image_direction_seed
from sd3.config import (ANCHOR_BLOCK, BACKBONE, DIRECTION_STEP, DTYPE, LATENT_SHAPE, N_BLOCKS,
                        PROMPT_SEED_INDEX, STEPS, directions_dir, geneval_ctx_dir)
from sd3.encode_prompts import load_text_pipe
from sd3.model import load_scheduler, load_teacher


def latent(prompt_idx: int, j: int) -> torch.Tensor:
    """Estimation latent j (100..124) of prompt `prompt_idx`: CPU float32 [1, 16, 128, 128]."""
    return cpu_randn(image_direction_seed(prompt_idx, j), (1,) + LATENT_SHAPE)


@torch.no_grad()
def teacher_state_at(teacher, x0, pe, pooled, k: int, device):
    """Block-12 (text, image) output of the teacher's k-th forward from x0, conditional only.

    Forwards 1..k-1 are full forwards on the teacher's 28-step schedule with Euler updates;
    forward k is truncated after the collapsed block."""
    sch = load_scheduler()
    sch.set_timesteps(STEPS["teacher"])
    sig = [float(s) for s in sch.sigmas]

    grab = {}
    blk = teacher.transformer_blocks[ANCHOR_BLOCK]
    h = blk.register_forward_hook(lambda _m, _i, out: grab.__setitem__("o", (out[0].detach(), out[1].detach())))
    x = x0.clone()
    try:
        for i in range(k):
            t = sch.timesteps[i].to(device).expand(x.shape[0])
            last = (i == k - 1)
            kw = {} if not last else {"skip_layers": list(range(ANCHOR_BLOCK + 1, N_BLOCKS))}
            v = teacher(hidden_states=x, timestep=t, encoder_hidden_states=pe,
                        pooled_projections=pooled, return_dict=False, **kw)[0]
            if not last:
                x = x + (sig[i + 1] - sig[i]) * v
    finally:
        h.remove()
    assert "o" in grab, "hook never fired"
    return grab["o"], sig[k - 1], sig[k]


@torch.no_grad()
def encode(pipe, text: str, device: str) -> dict:
    """Conditional encode_prompt of one prompt (batch 1): pe [1, 333, 4096], pooled [1, 2048]."""
    pe, _, pooled, _ = pipe.encode_prompt(prompt=text, prompt_2=None, prompt_3=None,
                                          device=device, num_images_per_prompt=1,
                                          do_classifier_free_guidance=False)
    return dict(pe=pe.cpu(), pooled=pooled.cpu(), prompt=text)


def encode_missing(texts: dict, ctx_dir: str, device: str):
    """Write <ctx_dir>/p{idx:03d}.pt for every prompt in `texts` ({idx: text}) not cached yet."""
    os.makedirs(ctx_dir, exist_ok=True)
    missing = [i for i in sorted(texts) if not os.path.exists(os.path.join(ctx_dir, f"p{i:03d}.pt"))]
    if not missing:
        return
    pipe = load_text_pipe(device)
    for i in missing:
        f = os.path.join(ctx_dir, f"p{i:03d}.pt")
        torch.save(encode(pipe, texts[i], device), f + f".tmp{os.getpid()}")
        os.replace(f + f".tmp{os.getpid()}", f)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[sd3-dirs] encoded {len(missing)} prompts -> {ctx_dir}", flush=True)


@torch.no_grad()
def estimate(teacher, pe, pooled, prompt_idx: int, device: str, n_lat: int = N_DIRECTION_SAMPLES,
             n_dir: int = N_DIR, k: int = DIRECTION_STEP):
    """(U, s, eigvals, seeds, shape) for one prompt from n_lat teacher rollouts."""
    seeds = direction_sample_indices(n_lat)
    pe, pooled = pe.to(device, DTYPE), pooled.to(device, DTYPE)
    X, shape = None, None
    for j, s in enumerate(seeds):
        (_txt, img), _, _ = teacher_state_at(teacher, latent(prompt_idx, s).to(device, DTYPE),
                                             pe, pooled, k, device)
        h = img[0]                                        # [4096, 1536]
        shape = tuple(h.shape)
        h = h.reshape(-1).to(torch.bfloat16).cpu()
        if X is None:
            X = torch.empty(n_lat, h.numel(), dtype=torch.bfloat16)
        X[j] = h
    U, Sv, evals = directions_from_samples(X, n_dir)
    return U, Sv, evals, seeds, shape


def save(path, U, Sv, evals, prompt_idx, prompt, seeds, shape, k=DIRECTION_STEP):
    return save_directions(path, U, Sv, evals, prompt_idx=prompt_idx, prompt=prompt, k=k,
                           block=ANCHOR_BLOCK, stream="image", arm="teacher", n=len(seeds),
                           seeds=seeds, shape=shape)


def prompt_directions_path(text: str) -> str:
    """<REDIVERSE_OUT>/sd3/prompt/<slug>/directions.pt of a custom prompt."""
    return os.path.join(D.prompt_dir(BACKBONE, text), "directions.pt")


def estimate_prompt(text: str, path: str, device: str) -> dict:
    """Directions of one custom prompt (latent seeds of prompt index 0) written to `path`. The text
    encoders and the teacher are loaded here and freed before returning."""
    pipe = load_text_pipe(device)
    c = encode(pipe, text, device)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    teacher = load_teacher(device)
    U, Sv, evals, seeds, shape = estimate(teacher, c["pe"], c["pooled"], PROMPT_SEED_INDEX, device)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rec = save(path, U, Sv, evals, PROMPT_SEED_INDEX, text, seeds, shape)
    print(f"[sd3-dirs] {path}: s1={float(Sv[0]):.2f} pc1_share={rec['pc1_share']:.3f}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompts", default=D.GENEVAL_553,
                    help="prompt file, line i = prompt index i (default GenEval-553); for another file also "
                         "set --out-dir and --ctx-dir")
    ap.add_argument("--prompt-ids", default=None, help="subset, e.g. '27' or '0-99' (default: all)")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker takes the prompts whose index modulo --num-shards equals --shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=None, help="default <REDIVERSE_DATA>/sd3/directions")
    ap.add_argument("--ctx-dir", default=None, help="default <REDIVERSE_DATA>/sd3/geneval_ctx")
    ap.add_argument("--prompt", default=None, help="a custom prompt instead of the prompt file")
    ap.add_argument("--out", default=None,
                    help="--prompt: output file (default <REDIVERSE_OUT>/sd3/prompt/<slug>/directions.pt)")
    ap.add_argument("--no-claim", action="store_true", help="skip the per-prompt claim locks")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    if a.prompt is not None:
        estimate_prompt(a.prompt, a.out or prompt_directions_path(a.prompt), dev)
        return

    texts = D.read_prompts(a.prompts)
    si, sn = a.shard, a.num_shards
    assert 0 <= si < sn, f"--shard {si} outside 0..{sn - 1}"
    keep = None if a.prompt_ids is None else set(D.parse_ids(a.prompt_ids))
    out_dir = a.out_dir or directions_dir()
    ctx = a.ctx_dir or geneval_ctx_dir()
    os.makedirs(out_dir, exist_ok=True)
    idxs = [i for i in range(len(texts)) if i % sn == si and (keep is None or i in keep)]
    missing = [i for i in idxs if not os.path.exists(direction_file(out_dir, i))]
    foreign = [i for i in sorted(set(idxs) - set(missing)) if not matches(direction_file(out_dir, i), texts[i])]
    if foreign:
        sys.exit(f"{len(foreign)} files in {out_dir} (first: p{foreign[0]:03d}.pt) hold the directions of "
                 f"other prompts than those of {a.prompts}; pass --out-dir and --ctx-dir to build a separate "
                 "direction bank for a different prompt file")
    print(f"[sd3-dirs] teacher block {ANCHOR_BLOCK} image stream, forward {DIRECTION_STEP}, "
          f"{N_DIRECTION_SAMPLES} latents, {N_DIR} directions: {len(missing)}/{len(idxs)} prompts -> {out_dir}",
          flush=True)
    if not missing:
        return
    encode_missing({i: texts[i] for i in missing}, ctx, dev)
    teacher = load_teacher(dev)
    t0, done = time.time(), 0
    for i in missing:
        f_out = direction_file(out_dir, i)
        if os.path.exists(f_out) or (not a.no_claim and not claim(f_out)):
            continue
        c = torch.load(os.path.join(ctx, f"p{i:03d}.pt"), map_location="cpu")
        assert c["prompt"] == texts[i], (f"ctx p{i:03d} holds {c['prompt']!r}, prompt {i} is {texts[i]!r}; "
                                         "pass --ctx-dir for a different prompt file")
        U, Sv, evals, seeds, shape = estimate(teacher, c["pe"], c["pooled"], i, dev)
        rec = save(f_out, U, Sv, evals, i, texts[i], seeds, shape)
        done += 1
        el = time.time() - t0
        if done % 10 == 0 or done == 1:
            print(f"  [{done}/{len(missing)}] p{i:03d} s1={float(Sv[0]):.2f} pc1_share={rec['pc1_share']:.3f} "
                  f"{el / done:.1f}s/prompt eta {(len(missing) - done) * el / done / 60:.0f}m", flush=True)
    print(f"[sd3-dirs] shard {si}/{sn} done", flush=True)


if __name__ == "__main__":
    main()

# ReDiverse on Krea 2

Teacher: [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (52 FlowMatchEuler steps, resolution-dependent shift
μ = 0.906 at 1024², Krea guidance 3.5 = conventional CFG 4.5). Student: [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo)
(TDM-distilled and post-trained full weights, 8 steps, μ = 1.15, no guidance). Both are gated on the Hub: accept Krea's license
and log in (`hf auth login`). They share the text encoder (Qwen3-VL-4B), tokenizer, VAE (Qwen-Image) and transformer config;
only the `Krea2Transformer2DModel` weights differ. Revisions are pinned in `krea2/config.py`. Images are 1024×1024
(packed latent 4,096 × 64).

| | value |
|---|---|
| collapsed block l | `transformer_blocks.26` of 28, image rows (the last 4,096 rows of the joint [text; image] sequence × 6,144) |
| probe at l (Turbo / Raw, step 1) | participation ratio 0.49, total variance 3.44, concentration 7.05 (N = 200, 6 GenEval prompts); concentration steps up from 2.35 at block 25 |
| teacher step tau | forward 5 of 52 (reads σ 0.96743, lands at 0.95883; Turbo's first step lands at 0.95672) |
| ReDiverse-FT target | Raw rolled 4 steps with Krea guidance 3.5 from the student's initial noise, then the conditional forward 5 |
| ReDiverse-FT adapter | LoRA r = 32, α = 32 on `attn.to_q/k/v/to_gate/to_out.0`, `ff.gate/up/down` of blocks 0..26 (216 modules, 103.5M parameters); active in sampling step 1 only |
| ReDiverse-TF | K = 8 directions from N = 25 conditional-only Raw rollouts per prompt (block 26 image rows at forward 5), α_k ~ U[−band, band] added in step 1; band 2 in the paper, 4 recommended for Krea 2 |

Everything runs in an environment with `diffusers` ≥ 0.39 (`Krea2Pipeline`), `transformers` 5.x (Krea's tokenizer and
Qwen3-VL configs use the 5.x format) and `peft`. Data go to `<REDIVERSE_DATA>/krea2/`, outputs to `<REDIVERSE_OUT>/krea2/`.
GPU figures below are for RTX PRO 6000 (96 GB); an 80 GB H100 is about 1.5–2× faster.

## Two implementation notes

- **Text compaction.** `Krea2Pipeline` pads every prompt to 512 text tokens and masks the padding. With a mask, PyTorch SDPA
  falls back to its math kernel for grouped-query attention (48 query / 12 key-value heads), which runs out of memory at
  batch 4. Every krea2 stage therefore drops the padded rows once at encoding time (`krea2.model.encode`). This is exact for
  this architecture (text rotary ids are zero and the text-fusion blocks have no positional encoding). On the pipeline's own
  masked inputs the custom loop's schedule, timesteps and Euler updates reproduce `Krea2Pipeline` bit for bit; with compaction
  (and the Euler product in fp32) the final latents differ only by kernel-level noise, below the pipeline's own batch-1 vs
  batch-2 difference, and the images match (`python -m krea2.generate --check-pipeline`).
- **fp32 norms.** `Krea2Transformer2DModel` keeps its RMSNorm scales in fp32 (`_keep_in_fp32_modules`); `krea2.model.load_transformer`
  moves the model to the device without a dtype cast and asserts it.

## Probe

```bash
python -m krea2.probe --arm teacher --n 200 --device cuda:0     # default prompts: prompts/probe_video20.txt
python -m krea2.probe --arm student --n 200 --device cuda:0
python -m rediverse.probe_report --teacher <out>/probe_n200/teacher --student <out>/probe_n200/student --collapsed block26
```

About 4 min per prompt and arm on one GPU at N = 200 (two groups of blocks within 160 GB of host RAM). `--blocks 22-27`
restricts the probe to a window (needed for N = 1000: 50 GB of staged activations per block). `--adapter` probes the
student with a ReDiverse-FT adapter active (paper App. A, "after the fine-tuning").

## ReDiverse-TF directions

```bash
python -m krea2.estimate_directions --blocks 26 --prompt-ids prompts/geneval_subset100.txt --shard 0 --num-shards 4 --device cuda:0
```

About 60 s per prompt (25 rollouts of 5 Raw forwards); one rollout serves several blocks (`--blocks 24,25,26,27`). Files
`<REDIVERSE_DATA>/krea2/directions_k5/b{l}/p{idx:03d}.pt` (U [8, 25,165,824] bf16, s [8], eigenvalues; ~400 MB each). Another
prompt file needs its own `--out-dir` and `KREA2_CTX_DIR` (cached contexts are keyed by prompt index).

## Training ReDiverse-FT

```bash
python prompts/download_train_captions.py                                    # 100 DiffusionDB captions
python -m krea2.build_targets --blocks 26 --shard 0 --num-shards 2 --device cuda:0   # target bank, ~1 min per caption
torchrun --standalone --nproc_per_node 2 -m krea2.train --block 26 --run b26_mse    # ~2.2 h on 2 GPUs (data parallel)
python -m krea2.convert_weights --src <out>/train/b26_mse/adapters/rediverse_ft_0000200.safetensors --out-dir <out>/release
```

The bank holds Raw's block-26 image rows for 100 captions × 16 noises (805 MB per caption and block); training reads it into
host RAM. Recipe constants are the paper's (`rediverse/data.py`): 32 pairs per step, 200 steps, AdamW 1e-4, plain MSE
(`--loss norm` gives MiniMax-H3's variance-normalised loss). `--guidance 0` trains on unguided targets (paper Tab. 11).
`convert_weights` writes the diffusers LoRA file and a ComfyUI/original-format file, and checks the round trip.

## Generation

```bash
python -m krea2.generate --arm rediverse --block 26 --band 4 --adapter <release>/krea-2-rediverse.safetensors \
    --prompt-ids prompts/geneval_subset100.txt --num-samples 16 --shard 0 --num-shards 8 --device cuda:0
```

Arms: `teacher`, `student`, `rediverse_ft`, `rediverse_tf`, `rediverse` (FT + TF). Sample `s` of prompt `i` starts from
`randn([4096, 64], CPU generator seeded i * 1000 + s)` in every arm and gets α from seed `7_000_000 + i * 1009 + s` in both TF
arms. Turbo arms run at ~3.5 s per image, the teacher at ~40 s. `krea2.sweep` renders many inference settings (adapter
checkpoint and strength, adapter steps, TF block, band and steps) with one model load; `krea2/sweep_table.py` tabulates them.

## Results (GenEval, 8 seeds per prompt, 1024²)

| | DINO | Vendi | DreamSim | CLIPScore | Δ HPSv2 | Δ ImageReward |
|---|---:|---:|---:|---:|---:|---:|
| Turbo | 49.4 | 2.38 | 22.6 | 29.7 | 0.00 | 0.000 |
| ReDiverse-TF (band 2) | 50.2 | 2.54 | 24.5 | 29.5 | −0.37 | +0.042 |
| ReDiverse-FT | 69.1 | 3.74 | 39.6 | 29.2 | −0.39 | −0.174 |
| ReDiverse, band 2 (paper) | 68.4 | 3.62 | 37.0 | 29.1 | −0.23 | −0.035 |
| ReDiverse, band 4 | 71.5 | 4.10 | 43.0 | 29.1 | −1.00 | −0.031 |
| Raw (teacher) | 70.0 | 4.10 | 43.2 | 28.9 | −2.23 | −0.243 |

The released adapter is on [ReDiverse/Krea-2-ReDiverse](https://huggingface.co/ReDiverse/Krea-2-ReDiverse) (gated: Krea 2
Community License), with a standalone diffusers script, `generate_krea2.py`.

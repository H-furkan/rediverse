# ReDiverse on MiniMax-H3

Teacher: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (50 scheduler steps, 49
transformer forwards, guidance-distilled, no CFG). Student: the 4-step Turbo LoRA of
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
(`minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`, 3 transformer forwards), bound on the
teacher transformer. Both are pinned to the revisions in `h3/config.py` and are fetched into the
Hugging Face cache (`HF_HOME`) on first use. The MiniMax-H3 weights are distributed under the
MiniMax H3 Community License; read its terms before downloading or using them.

Clips are 1344x768, 124 frames, 24 fps (H.264, CRF 17, no audio track). MiniMax-H3 denoises
video and audio jointly in one packed sequence; the audio stream starts from one constant noise
tensor per prompt (stored in the prompt's text context, shared by all clips and arms) and is not
decoded.

| | value |
|---|---|
| collapsed block l | `transformer_blocks[45]`, video rows of the packed `[text \| audio \| video]` sequence (37,296 x 5,376) |
| teacher step tau | forward 16 (sigma 0.9645 -> 0.9612, matched to the student's first landing at 0.96) |
| ReDiverse-FT adapter | LoRA r = 32, alpha = 32 on `attn.to_q/k/v/out.0`, `ff.net.0.proj`, `ff.net.2` of blocks 0..45 (276 modules); active in the student's first forward only |
| ReDiverse-TF | K = 8 directions from N = 25 teacher rollouts per prompt, alpha_k ~ U[-2, 2] |

All commands run from the repository root in the `video` environment (`envs/requirements-video.txt`),
which needs the system `ffmpeg` with libx264. Data go to `data/minimax-h3/` (`REDIVERSE_DATA`),
outputs to `outputs/minimax-h3/` (`REDIVERSE_OUT`). Every default reproduces the paper's
artifacts. GPU figures below are for 80 GB H100s.

## Generation

The MiniMax-H3 ReDiverse-FT adapter is not distributed on the Hugging Face Hub. Train it and
convert it first (see [Training ReDiverse-FT](#training-rediverse-ft)); the `rediverse_ft` and
`rediverse` arms then take it with `--adapter`, either the directory written by
`h3.convert_weights` or the `rediverse_ft.safetensors` file in it. The `teacher`, `student` and
`rediverse_tf` arms need no adapter.

The text encoder (Qwen3-VL), the transformer and the video VAE each need most of an 80 GB GPU, so
benchmark generation runs in three phases. Each phase resumes where it stopped, and several
workers (one per GPU) can run the same command at once: clips are claimed atomically.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. text contexts of the 300 VBench prompts (prompts/vbench_300.txt)
python -m h3.generate --encode --device cuda:0

# 2. ReDiverse-TF directions of the rediverse_tf and rediverse arms (see below)
python -m h3.estimate_directions --device cuda:0

# 3. latents, 10 clips per prompt: teacher, student, rediverse_ft, rediverse_tf, rediverse
python -m h3.generate --arm rediverse --adapter outputs/minimax-h3/train/rediverse_ft --device cuda:0

# 4. mp4s -> outputs/minimax-h3/<arm>/videos/<prompt>-<k>.mp4
python -m h3.generate --arm rediverse --decode --device cuda:0
```

Clip k of VBench prompt i starts from `randn(seed = 1_000_000 + (10000 + i) * 100003 + k)` in
every arm, and both TF arms draw the same coefficients `U[-2, 2]^8` from seed
`7_000_000 + i * 1009 + k` (recorded in `scales_<pid>.json`). Per clip: about 30 s for the
student-based arms (3 forwards) and 440 s for the teacher. Step 3 keeps each clip's final latent in
`outputs/minimax-h3/<arm>/latents/` (7.2 MB each, about 21.5 GB per arm). They can be deleted once
`--decode` has written the mp4s; running step 3 again would sample them again.

Useful options: `--prompt-ids 0-29` (subset), `--num-samples` (clips per prompt, default 10),
`--shard I --num-shards N` (worker I takes every N-th prompt), `--reverse`, `--out-dir` (default
`outputs/minimax-h3/<arm>`).

Single prompt (all phases in one process), clips written to
`outputs/minimax-h3/prompt/<slug>/<arm>/00.mp4, 01.mp4, ...`, where `<slug>` is the prompt in
lowercase with every run of other characters replaced by `_` and leading/trailing `_` removed; a
slug longer than 80 characters is cut to 71 characters and followed by `_` and an 8-character hash
of the full prompt:

```bash
python -m h3.generate --prompt "a corgi running on the beach at sunset" --arm rediverse_tf --num-samples 4 --device cuda:0
python -m h3.generate --prompt "a corgi running on the beach at sunset" --arm rediverse --num-samples 4 \
    --adapter outputs/minimax-h3/train/rediverse_ft --device cuda:0
```

The TF arms read the prompt's directions from `--directions FILE`, else from
`outputs/minimax-h3/prompt/<slug>/directions.pt` when that file was estimated for the same prompt
and `--prompt-index`. Otherwise the directions are first estimated with the teacher (about one
GPU-hour, the same computation as `python -m h3.estimate_directions --prompt ...`) and saved to that
path, so later runs of the same prompt and index reuse them. Clip k uses the seed rules above with
i = `--prompt-index` (default 0); with a VBench prompt and its index, the clips are those of the
benchmark.

## ReDiverse-TF directions

```bash
python -m h3.estimate_directions --device cuda:0          # one worker per GPU; prompts are claimed
```

For each VBench prompt the teacher is run 16 forwards from 25 latents (sample indices 100..124,
disjoint from the evaluation clips), the block-45 video rows are flattened (D = 200,503,296) and
the top 8 principal directions are written to `data/minimax-h3/directions/b45_k16/v{idx:03d}.pt`
(U in bfloat16, 3.2 GB per prompt, about 960 GB for all 300). About 1 GPU-hour and 30 GB of host
memory per prompt. `--shard I --num-shards N` splits the prompts over workers explicitly.

For a single prompt, the directions go to `outputs/minimax-h3/prompt/<slug>/directions.pt`, where
`h3.generate --prompt` finds them:

```bash
python -m h3.estimate_directions --prompt "a corgi running on the beach at sunset" --device cuda:0
```

## Training ReDiverse-FT

The H3 adapter is trained on its own data: 20 GenEval prompts (`prompts/h3_train_geneval_ids.txt`
over `prompts/geneval_553.txt`) x 20 latents (seed `1_000_000 + p * 100003 + i`), with the teacher's
block-45 video rows at forward 16 precomputed into a target bank. This recipe differs from the shared recipe of the
SDXL, SD3 and Wan 2.1 adapters (100 DiffusionDB captions x 16 noises, batch 32, plain MSE) that the paper describes for
all four pairs; the paper's MiniMax-H3 ReDiverse-FT rows were produced with it.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. text contexts of the 20 training prompts
python -m h3.encode_prompts --set train --device cuda:0

# 2. target bank: 400 entries of about 415 MB (166 GB), about 150 s each; one worker per GPU
python -m h3.build_targets --device cuda:0

# 3. training: 200 steps on two 80 GB GPUs (about 22 h, 64 GB peak per GPU)
#    -> outputs/minimax-h3/train/adapters/rediverse_ft_0000200.pth
python -m h3.train --device cuda:0 --dev1 cuda:1

# 4. adapter file -> outputs/minimax-h3/train/rediverse_ft/{rediverse_ft.safetensors, config.json}
python -m h3.convert_weights --src outputs/minimax-h3/train/adapters/rediverse_ft_0000200.pth \
    --out-dir outputs/minimax-h3/train/rediverse_ft
python -m h3.generate --arm rediverse_ft --adapter outputs/minimax-h3/train/rediverse_ft --device cuda:0
```

Recipe: loss `0.5 * MSE(h_student, h_teacher) / Var(h_teacher)` of each latent; each step has
4 micro-batches of 2 latents of one prompt (prompts round-robin, latent window rotating);
AdamW lr 1e-4, weight decay 0.01, betas (0.9, 0.999), gradient clipping at 10, `torch.manual_seed(0)`.
The trainer checks that the target bank holds all 400 entries before it starts. To fit the
124-frame sequence, Turbo is fused into the base weights for training, blocks 0..22 run on
`--device`, blocks 23..45 on `--dev1`, and blocks 46..49 are dropped (the forward stops at block
45). Checkpoints `rediverse_ft_{iter:07d}.pth` are written every 50 steps and at the last one
(`--steps`, default 200); the paper's adapter is step 200. Per-step metrics go to
`train_log.jsonl`. An `--out-dir` that already holds checkpoints needs `--overwrite`.
`--dry-run 2` checks memory and speed with two single-micro-batch steps: it writes no checkpoint,
logs to `dry_run.jsonl`, and its loss, rel and gradient norm carry the 1/4 weight of one
micro-batch in a full step. Training runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(`scripts/h3.sh` sets it), which keeps the training peak within 80 GB. The bf16 backward pass is
not bitwise deterministic across runs; the first step's loss is reproduced exactly.

## Files

| file | role |
|---|---|
| `config.py` | model ids, pinned revisions, geometry, recipe constants, data paths |
| `model.py` | model loading (teacher, Turbo binding), text contexts, packed layout, sampler, teacher rollout, video decoder |
| `split2.py` | two-GPU block split and the forward up to block 45 used in training |
| `encode_prompts.py` | text contexts of the training and VBench prompts |
| `build_targets.py` | ReDiverse-FT target bank |
| `train.py` | ReDiverse-FT training |
| `estimate_directions.py` | ReDiverse-TF directions (VBench bank or one `--prompt`) |
| `generate.py` | the five arms: encode, sample, decode, or one `--prompt` |
| `convert_weights.py` | training checkpoint -> adapter file (`rediverse_ft.safetensors` + `config.json`) |

`scripts/h3.sh` chains these stages over several GPUs (`bash scripts/h3.sh encode bank train directions
generate`); its `generate` stage reads the adapter written by its `train` stage unless `ADAPTER` points
elsewhere, and `PY_VIDEO` names the interpreter of the video environment (default `python`).

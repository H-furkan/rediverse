# ReDiverse on Wan 2.1

Teacher: [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)
(revision `38ec498c`). Student: the 4-step step- and CFG-distilled
[lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v)
(revision `14734f86`, file `distill_models/distill_model.safetensors`). Clips are 81 frames at
832x480 (latent 16 x 21 x 60 x 104, 32,760 tokens per block). The collapsed block is `blocks.28`.

| | value |
|---|---|
| Teacher sampling (evaluation) | UniPC, 50 steps, repository scheduler (flow shift 3), CFG 5 against the WanPipeline docstring's English negative prompt |
| Student sampling | flow Euler, sigmas 1, 0.9375, 0.8333, 0.625, 0 (timestep = 1000 sigma), guidance 1 |
| ReDiverse-FT target | `blocks.28` output at forward 13 of the CFG-distilled teacher: the base teacher plus [Perflow-Shuai/Wan2.1-T2V-14B-CFG5-to-CFG1-50Step-LoRA-r128-iter500](https://huggingface.co/Perflow-Shuai/Wan2.1-T2V-14B-CFG5-to-CFG1-50Step-LoRA-r128-iter500) (revision `11730fdc`, rank 128, scale 1), UniPC 50 steps at flow shift 5, one conditional forward per step |
| ReDiverse-FT adapter | LoRA rank 32, alpha 32, on `attn1`/`attn2` `to_q`, `to_k`, `to_v`, `to_out.0` and `ffn.net.0.proj`, `ffn.net.2` of blocks 0-28 (290 modules); active at the student's first step only |
| ReDiverse-FT data | 100 training captions x 16 latents = 1,600 pairs; 200 steps of 4 captions x 8 latents; MSE over all video tokens; AdamW 1e-4, weight decay 0.01, clipping 10 |
| ReDiverse-TF directions | base teacher, conditional only, repository schedule (flow shift 3), `blocks.28` at forward 8, N = 25 latents, top K = 8 |
| ReDiverse-TF injection | at the student's first step, `blocks.28` output += sum_k alpha_k s_k u_k, alpha_k ~ U[-2, 2] |

All constants live in [`config.py`](config.py); every command below reproduces the released
artifacts with its default arguments.

## Environments

Two of the environments in [`envs/`](../envs/README.md) are used:

| Stage | Environment |
|---|---|
| `wan.encode_prompts`, `wan.generate`, `wan.estimate_directions` | `video` (diffusers 0.40, transformers 5.x for the UMT5 tokenizer) |
| `wan.build_targets`, `wan.train` | `train` (diffusers 0.37, peft 0.19) |
| `wan.convert_weights` | either |

Models are downloaded into your Hugging Face cache (`HF_HOME`) on first use at the pinned revisions.
Data goes to `data/wan2.1/` and outputs to `outputs/wan2.1/` (override with `REDIVERSE_DATA` /
`REDIVERSE_OUT`). Every command runs from the repository root. Generation needs about 43 GB of GPU
memory (a single-prompt TF call that estimates its directions holds the teacher first, about 34 GB,
and frees it before loading the student), target building 35 GB, training 54 GB per GPU (80 GB H100s
were used throughout).

## Generation with the released adapter

```bash
# ReDiverse-FT, four clips of one prompt (adapter downloaded from the Hugging Face Hub)
python -m wan.generate --arm rediverse_ft --prompt "a corgi surfing a wave at sunset" --num-samples 4

# ReDiverse (FT + TF); the first TF call for a prompt estimates its teacher directions (~15 min on an H100)
python -m wan.generate --arm rediverse --prompt "a corgi surfing a wave at sunset" --num-samples 4
```

Clips are written to `outputs/wan2.1/prompt/<slug>/<arm>/00.mp4`, `01.mp4`, ..., where `<slug>` is
the prompt in lowercase with every run of other characters replaced by `_` and leading/trailing `_`
removed (`a_corgi_surfing_a_wave_at_sunset`); a slug longer than 80 characters is cut to 71
characters and followed by `_` and an 8-character hash of the full prompt. The TF arms
(`rediverse_tf`, `rediverse`) read the prompt's directions from
`outputs/wan2.1/prompt/<slug>/directions.pt`. When that file does not exist, or holds the directions
of another prompt or another `--prompt-index`, the base teacher estimates them first, saves them
there and is freed before the student is loaded, so later calls with the same prompt and index reuse
the file. The same file can be made on its own with
`python -m wan.estimate_directions --prompt "a corgi surfing a wave at sunset"` (with the same
`--prompt-index`); `--directions FILE` points `wan.generate` at another file, which is used as given.

`--prompt-index i` selects the seed rules' prompt index: clip k starts from seed
`1000000 + i*100003 + k` and draws its TF coefficients from seed `7000000 + i*1009 + k`, so the text of
VBench prompt i with `--prompt-index i` reproduces that prompt's benchmark clips. Every call renders
all requested clips, replacing existing files. `--adapter` accepts a `rediverse_ft.safetensors` file
or a directory holding it; `--out-dir` overrides the output directory.

## Full pipeline

### 0. Training captions

```bash
python prompts/download_train_captions.py          # prompts/train_captions.txt (100 DiffusionDB captions)
```

### 1. Text contexts (`video`)

```bash
python -m wan.encode_prompts --set train,vbench,neg --device cuda:0
```

UMT5 contexts of the 100 training captions (`data/wan2.1/ctx/train/p10000.pt` ... `p10099.pt`), the
VBench-300 prompts (`data/wan2.1/ctx/vbench/v000.pt` ... `v299.pt`) and the teacher's negative
prompt (`data/wan2.1/ctx/neg.pt`). A few minutes, dominated by loading UMT5-XXL.

### 2. ReDiverse-FT target bank (`train`)

```bash
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python -m wan.build_targets --device cuda:0 --shard $g --num-shards 8 &
done; wait
```

1,600 entries `data/wan2.1/targets/f81_b28_s13/p{10000+ci}_i{ni:03d}.safetensors` (bf16 target
`[32760, 5120]` + the fp32 initial latent, 328 MiB each, 512 GiB in total) and `manifest.json`.
One entry is 13 teacher forwards, about 50 s on an H100 (23 GPU-hours in total). The builder
resumes where it stopped.

### 3. ReDiverse-FT training (`train`)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node 8 -m wan.train
```

Writes `outputs/wan2.1/train/config.json` and an adapter checkpoint every 50 steps,
`outputs/wan2.1/train/adapters/rediverse_ft_0000050.pth` ... `rediverse_ft_0000200.pth`; the
released adapter is step 200 (`--steps`, default 200). About 3 hours on 8 H100s. Each rank keeps
the bank entries it has read in host RAM (up to 175 GiB per rank at 8 ranks), so epochs 2-4 do not
touch the disk. Before training, the trainer checks that every text context holds its caption
(`--captions` names the caption file when it is not `prompts/train_captions.txt`), that its
truncated student forward equals the full forward at `blocks.28` and that the bank latents follow
the seed rule, and it prints the pre-LoRA residual on caption 0, latent 0 (MSE 0.12245669960975647,
relative L2 0.804051399230957).

Any world size dividing 32 computes the same 32-pair mean gradient per step. At 8 ranks each rank
holds one latent of each of the step's 4 captions, so every forward has batch size 1; at smaller
world sizes `--chunk` (default 2) latents of a caption share a forward, which changes bf16
rounding. The released adapter was trained with 8 ranks in the `train` environment.

### 4. Released adapter format

```bash
python -m wan.convert_weights --src outputs/wan2.1/train/adapters/rediverse_ft_0000200.pth --out-dir outputs/wan2.1/release
python -m wan.generate --arm rediverse_ft --adapter outputs/wan2.1/release --prompt "a corgi surfing a wave at sunset"
```

`outputs/wan2.1/release` then holds `rediverse_ft.safetensors` and `config.json`. The safetensors
file holds the 580 float32 LoRA tensors under `blocks.N.<module>.lora_{A,B}.weight` with string
metadata; `config.json` describes it. At inference peft creates the adapter in the student's bf16,
so the weights are used in bf16.

### 5. ReDiverse-TF directions (`video`)

```bash
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python -m wan.estimate_directions --device cuda:0 --shard $g --num-shards 8 &
done; wait
```

`data/wan2.1/directions/b28_k8_f81/v000.pt` ... `v299.pt`: per prompt, 25 base-teacher rollouts
to forward 8 (seed rule `1000000 + i*100003 + j`, j = 100..124), then the top 8 directions
(`U` bf16 `[8, 167731200]`, `s`, all 25 eigenvalues). 2.7 GB and about 15 minutes per prompt
(about 77 GPU-hours and 805 GB for all 300).

### 6. Generation, VBench-300 x 10 clips (`video`)

```bash
for arm in teacher student rediverse_ft rediverse_tf rediverse; do
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g python -m wan.generate --arm $arm --device cuda:0 --shard $g --num-shards 8 &
  done; wait
done
```

`outputs/wan2.1/<arm>/videos/<prompt>-<k>.mp4` (VBench naming), plus `meta.json` and, for the TF
arms, `scales_<pid>.json` with the coefficients of every clip. Workers may also share one output
directory without sharding (`--reverse` walks the prompts back to front); `--prompt-ids` and
`--num-samples` (default 10) select a subset. Per clip on an H100: teacher about 440 s (100
forwards), the 4-step arms about 21 s. Clip k of prompt i starts from
seed `1000000 + i*100003 + k` in every arm and its TF coefficients come from seed
`7000000 + i*1009 + k`, so the arms are paired clip by clip.

Then evaluate the five output directories with [`evaluation/`](../evaluation/README.md).

## Reproducibility

- The paper's clips were written with ffmpeg 6.1.1 (libx264, CRF 17, yuv420p, 16 fps); another
  ffmpeg or x264 build encodes the same frames to different bytes, so compare decoded frames.
- Latents are drawn on the CPU from explicitly seeded generators (`rediverse/seeds.py`), so they do
  not depend on the GPU or on the order in which clips are rendered.
- The teacher's guidance combination is computed on the bf16 velocities and its UniPC update in
  float32; the student update casts the velocity to float32. The training timestep is 999
  (float32), the sampling timestep of the first step 1000.

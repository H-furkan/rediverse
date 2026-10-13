# ReDiverse on SDXL

Teacher: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
(50-step EulerDiscrete with the repository's scheduler config, CFG 8, no negative prompt). Student:
the DMD2 4-step UNet [tianweiy/DMD2](https://huggingface.co/tianweiy/DMD2) (`dmd2_sdxl_4step_unet_fp16.bin`
for sampling, `dmd2_sdxl_4step_unet.bin` in fp32 for training) in the SDXL-base pipeline,
LCMScheduler at timesteps 999/749/499/249, guidance 0. Both are pinned to the revisions in
`sdxl/config.py` and are fetched into the Hugging Face cache (`HF_HOME`) on first use. Images are
1024x1024 (latent 4x128x128).

| | value |
|---|---|
| collapsed block l | `up_blocks.0.attentions.0` (output of the first Transformer2DModel of the up path, 1280x32x32) |
| teacher step tau | forward 7 of the 50-step Euler schedule (t = 861; forward 1 is t = 981) |
| ReDiverse-FT target | teacher rolled 6 steps with CFG 8 from the student's initial noise, then the conditional forward 7 |
| ReDiverse-FT adapter | LoRA r = 32, alpha = 32 on `attn1`/`attn2` `to_q/to_k/to_v/to_out.0` and `ff.net.0.proj`/`ff.net.2` of every transformer block in `down_blocks`, `mid_block` and `up_blocks.0.attentions.0` (440 Linear modules, 53.9M parameters); active in sampling step 1 only |
| ReDiverse-TF | K = 8 directions from N = 25 conditional-only teacher rollouts per prompt (block l at forward 7), alpha_k ~ U[-2, 2], added in step 1 |

Data go to `data/sdxl/` (`REDIVERSE_DATA`), outputs to `outputs/sdxl/` (`REDIVERSE_OUT`). All
commands run from the repository root. Generation, the noise bank and the directions use the `image`
environment (`envs/requirements-image.txt`); caption encoding, the initialisation checkpoint and
training use the `train` environment (`envs/requirements-train.txt`). Every default reproduces the
released artifacts. GPU figures below are for 80 GB H100s.

## Generation with the released adapter

```bash
# frozen evaluation latents: GenEval-553 x 100, 7.25 GB (see "Noise bank" below)
python -m sdxl.make_noise_bank --device cuda:0

# ReDiverse-TF directions for rediverse_tf and rediverse, about 2 h (see "ReDiverse-TF directions" below)
python -m sdxl.estimate_directions --device cuda:0

# teacher | student | rediverse_ft | rediverse_tf | rediverse
python -m sdxl.generate --arm rediverse --device cuda:0
for k in 0 1 2 3; do python -m sdxl.generate --arm rediverse --shard $k --num-shards 4 --device cuda:$k & done; wait
```

Images are written to `outputs/sdxl/<arm>/images/{idx:03d}_s{s}.png` (GenEval prompt `idx` of
`prompts/geneval_553.txt`, sample `s` in 0..99); a rerun skips prompts whose images all exist.
One GPU renders the 55,300 images of a student-based arm in about 9 hours and those of the teacher
in about 37 hours. Useful options: `--prompt-ids 0-99`, `--num-samples 10`, `--out-dir`.

`--adapter` defaults to the released `sdxl/rediverse_ft.safetensors` on the Hugging Face Hub; it also
takes a local `.safetensors` file or a directory holding `rediverse_ft.safetensors` (for example the
output of `sdxl.convert_weights`, see below). The TF arms record their coefficients in
`outputs/sdxl/<arm>/scales_shard<k>of<n>.json`; sample `s` of prompt `i` gets `U[-2, 2]^8` from the
CPU seed `7_000_000 + i * 1009 + s` in both `rediverse_tf` and `rediverse`.

Single prompt, any arm:

```bash
python -m sdxl.generate --arm rediverse --prompt "a photo of a fox in the snow" --num-samples 4 --device cuda:0
# -> outputs/sdxl/prompt/a_photo_of_a_fox_in_the_snow/rediverse/{00,01,02,03}.png
```

Sample `k` follows the seed rules of prompt index 0: initial noise from the CUDA generator seeded
`k`, TF coefficients from the CPU seed `7_000_000 + k`. The TF arms read the direction file given
by `--directions`, otherwise `outputs/sdxl/prompt/<prompt>/directions.pt`; when that file is
missing or holds the directions of another prompt it is estimated first with the teacher (about 15 s
plus model loading, the same computation as `python -m sdxl.estimate_directions --prompt "..."`,
direction latents seeded `100..124`), saved there and reused by later calls for the same prompt. The
teacher is released before the student is loaded. `--out-dir` replaces the arm directory.

How the arms run, all in one `StableDiffusionXLPipeline` call (fp16, VAE decoded in fp32) per batch
of 10 samples of one prompt:

- `teacher`: 50 Euler steps, CFG 8. `student`: 4 LCM steps, guidance 0 (the scheduler warns that
  749 and 249 are off its distillation schedule; this is DMD2's schedule).
- `rediverse_ft`: the student with the adapter enabled before the call and disabled by
  `callback_on_step_end` after step 1.
- `rediverse_tf`: the student with `sum_k alpha_k s_k u_k` (computed in bf16, added in fp16) added to
  the output of `up_blocks.0.attentions.0` in step 1, one coefficient vector per image of the batch.
- `rediverse`: `rediverse_ft` with the same directions and coefficients.

The batch size is part of the protocol: fp16 reductions differ between batch shapes, so
`--batch-size 10` (the default) is needed to reproduce the images bit for bit. A batch of 10 peaks
at about 54 GB of allocated GPU memory (the fp32 VAE decode), so the bit-exact default needs an 80 GB
card; there PyTorch's caching allocator may log a failed 12.9 GB allocation, after which it frees its
cached blocks and continues with the same images.

## Noise bank

Sample `s` of prompt `i` starts from `torch.randn((1, 4, 128, 128), generator=torch.Generator("cuda")
.manual_seed(i * 1000 + s), dtype=torch.float16)` in every arm; the bank (`data/sdxl/noise_bank_553x100.npz`,
one key per prompt) freezes these draws. The LCM sampler re-noises between steps from the same
per-sample generator, so with bank latents the generator first consumes the draw the pipeline would
have made itself; bank mode and generator mode therefore give identical images. Without a bank file
`sdxl.generate` draws from the generators directly (CUDA draws may differ between GPU models; the
bank built on an H100 is the reference).

## ReDiverse-TF directions

```bash
python -m sdxl.estimate_directions --device cuda:0                 # all 553 prompts, about 2 h
for k in 0 1 2 3; do python -m sdxl.estimate_directions --shard $k --num-shards 4 --device cuda:$k & done; wait
python -m sdxl.estimate_directions --prompt "a photo of a fox in the snow"   # one free-text prompt
```

Per prompt: 25 latents (CPU `randn`, seeds `i * 1000 + 100..124`, disjoint from the evaluation
samples), the fp16 teacher rolled conditional-only (no CFG) for 6 Euler steps, the output of
`up_blocks.0.attentions.0` at forward 7 (D = 1,310,720, kept in bf16), top-8 principal directions of
the centred samples (Gram trick, `rediverse.directions`). About 14 s per prompt; files
`data/sdxl/directions/p{idx:03d}.pt` (`U` [8, D] bf16, `s` [8], eigenvalues), 21 MB each and 11.6 GB
for all 553. With `--prompt` the latents use the seeds of prompt index 0 (`100..124`) and the file
is `outputs/sdxl/prompt/<prompt>/directions.pt`, the one `sdxl.generate --prompt` reads. Each file
records its prompt, and both commands check it: for another prompt file, pass `--prompts` together
with `--out-dir` here and `--directions-dir` to `sdxl.generate`.

## Training ReDiverse-FT

Training runs on NVIDIA FastGen at a pinned commit, with four files added
(`sdxl/fastgen_overlay`, listed in its `REDIVERSE_OVERLAY.md`).

```bash
bash sdxl/setup_fastgen.sh                          # clone FastGen 060b857 into third_party/FastGen + overlay
python prompts/download_train_captions.py           # prompts/train_captions.txt (100 DiffusionDB captions)
python -m sdxl.encode_captions --device cuda:0      # data/sdxl/train_ctx/p000..p099.pt
python -m sdxl.build_init_ckpt                      # data/sdxl/dmd2_sdxl_init.pth (fp32 DMD2 UNet, 10.3 GB)
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} # FastGen loads the teacher with cache_dir=$HF_HOME
CUDA_VISIBLE_DEVICES=0 python -m sdxl.train         # outputs/sdxl/train, about 95 min on one H100
python -m sdxl.convert_weights --src outputs/sdxl/train/adapters/rediverse_ft_0000200.pth \
    --out-dir outputs/sdxl/release                  # rediverse_ft.safetensors + config.json
python -m sdxl.generate --arm rediverse_ft --adapter outputs/sdxl/release --device cuda:0
```

`sdxl.train` writes `outputs/sdxl/train/config.json` (the recipe), runs FastGen's `train.py` with
`fastgen/configs/experiments/SDXL/config_rediverse_ft.py` (FastGen's `config.yaml` and
`checkpoints/0000200.pth` go to the same directory), logs to `train.log`, and extracts the
generator LoRA of the last checkpoint into `outputs/sdxl/train/adapters/rediverse_ft_0000200.pth`
(`python -m sdxl.extract_adapter --src <checkpoint> --out-dir <dir>` does the same for any
checkpoint). FastGen also downloads the fp32 SDXL-base UNet into
`$HF_HOME/models--stabilityai--stable-diffusion-xl-base-1.0`.

The recipe: 100 captions x 16 paired noises (CPU `randn`, seed `1_000_000 + (10000 + ci) * 100003 + ni`),
4 captions x 8 noises per optimizer step (four gradient-accumulation micro-batches of one caption x
8 noises, `rediverse.data.StepSampler(seed=0)`), 200 steps. The student's step-1 input is
`eps * sigma(t=0.999)` at t = 0.999, and the teacher rollout starts from the same tensor (its
Euler schedule scales it by `init_noise_sigma` and back). Loss: plain MSE over all elements between
the student's and the teacher's `up_blocks.0.attentions.0` outputs, student and teacher in fp32
with TF32 matmuls, gradient checkpointing on. AdamW lr 1e-4 (weight decay 0.01, betas 0.9/0.999,
eps 1e-8, constant), gradients clipped to norm 10, seed 0 (it sets peft's LoRA A initialisation).
Each micro-batch costs one student forward and 13 teacher forwards (6 CFG steps + forward 7); peak
memory is about 75 GB (80 GB GPU; most of it is cuDNN autotuning workspace). The log prints the
first micro-batches as `micro-batch 1: caption 73 noises [4, 8, 12, 5, 3, 13, 7, 1]` and, every 5
iterations, the last micro-batch's loss and the mean across-noise std of the collapsed-block activation
(`spread_student`, `spread_teacher`); the released run logged
`LOSS iter=5 total_loss=23.53036 reg_loss=23.53036 spread_student=3.06363 spread_teacher=3.30932
spread_ratio=0.92576 step1_std=0.65548`. cuDNN autotuning (its choice depends on free memory) and
the GPU backward pass are not bitwise deterministic, so reruns differ in the fourth or fifth
significant digit from the second step on (23.529 to 23.530 at iteration 5 on H100s).
`--steps`, `--out-dir` and FastGen overrides after `--` change only the length, the run directory and
the trainer settings.

## Released weights

`sdxl/rediverse_ft.safetensors` holds the 880 LoRA tensors (fp32) under
`<module>.lora_{A,B}.weight` (module paths inside the UNet, e.g.
`down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q`); the safetensors metadata and
`config.json` list rank, alpha, the adapted modules, the base models and their revisions.
`sdxl.generate.attach_adapter` adds it to the student UNet with `unet.add_adapter(LoraConfig(r=32,
lora_alpha=32, target_modules=...))` in fp16. `sdxl.convert_weights` writes it from a training
adapter `rediverse_ft_<iteration>.pth` (`{"lora": {"unet.<module>.lora_A.default.weight": ...}, "src": "net",
"iter": ...}`), renaming the tensors and leaving their values unchanged.

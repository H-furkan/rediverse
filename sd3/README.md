# ReDiverse on SD3-Medium

Teacher: [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
(28 FlowMatchEuler steps, shift 3, CFG 7). Student: the TDM 4-step LoRA
[Luo-Yihong/TDM_sd3_lora](https://huggingface.co/Luo-Yihong/TDM_sd3_lora) (`tdm_sd3_lora.safetensors`)
on the same transformer at adapter scale 0.125, 4 steps, guidance 1.0 (TDM is CFG-distilled). Both
are pinned to the revisions in `sd3/config.py` and are fetched into the Hugging Face cache
(`HF_HOME`) on first use. SD3-Medium is a gated model: accept its license on the Hub and log in
(`hf auth login`) before the first run.

Images are 1024x1024 (latent 16x128x128).

| | value |
|---|---|
| collapsed block l | `transformer_blocks.12`, image stream (the second output of the MMDiT block; 4,096 tokens x 1,536) |
| teacher step tau | forward 9 of 28 (reads sigma 0.8774, lands at 0.8577 = the student's sigma after step 1) |
| ReDiverse-FT target | teacher rolled 8 steps with CFG 7 from the student's initial noise, then the conditional forward 9 |
| ReDiverse-FT adapter | LoRA r = 32, alpha = 32 on `attn.to_q/k/v/out.0`, `attn.add_q/k/v_proj`, `attn.to_add_out`, `ff.net.{0.proj,2}`, `ff_context.net.{0.proj,2}` of blocks 0..12, without the four modules of block 12 that only write its text output (152 modules, 22.3M parameters); active in sampling step 1 only |
| ReDiverse-TF | K = 8 directions from N = 25 conditional-only teacher rollouts per prompt (block 12 image stream at forward 9), alpha_k ~ U[-2, 2], added in step 1 |

All commands run from the repository root in the `train` environment (`envs/requirements-train.txt`).
Data go to `data/sd3/` (`REDIVERSE_DATA`), outputs to `outputs/sd3/` (`REDIVERSE_OUT`). Every default
reproduces the released artifacts. GPU figures below are for 80 GB H100s.

## Generation with the released adapter

```bash
# frozen evaluation latents: GenEval-553 x 100, about 29 GB (see "Noise bank" below)
python -m sd3.make_noise_bank --device cuda:0

# ReDiverse-TF directions, needed by rediverse_tf and rediverse: about 10 s per prompt, 56 GB (see below)
python -m sd3.estimate_directions --device cuda:0

# teacher | student | rediverse_ft | rediverse_tf | rediverse
python -m sd3.generate --arm rediverse --device cuda:0
```

Images are written to `outputs/sd3/<arm>/images/{idx:03d}_s{s}.png` (GenEval prompt `idx` of
`prompts/geneval_553.txt`, sample `s` in 0..99). Several workers (one per GPU) can run the same
command at once: prompts are claimed atomically under `<out-dir>/.claims`, and a rerun resumes where
it stopped. Throughput per GPU: teacher about 0.5 img/s (about 31 GPU-hours for 55,300 images),
student-based arms 1.7 to 3.4 img/s. Useful options: `--prompt-ids 0-99`, `--num-samples 10`,
`--shard I --num-shards N` (worker `I` takes the prompts with index `i % N == I`), `--out-dir`.
The TF arms stop before loading any model when a direction file is missing.

`--adapter` defaults to the released `sd3/rediverse_ft.safetensors` on the Hugging Face Hub; it
also takes a local `.safetensors` file, a directory holding `rediverse_ft.safetensors` (or laid out
like the Hub repository), or a checkpoint written by `sd3.train`. The TF arms record their
coefficients in `outputs/sd3/<arm>/scales_<pid>.json`; sample `s` of prompt `i` gets
`U[-2, 2]^8` from seed `7_000_000 + i * 1009 + s` in both `rediverse_tf` and `rediverse`.

Single prompt (`--num-samples`, default 4; sample `k` uses latent seed `k` and, in the TF arms,
coefficient seed `7_000_000 + k`, i.e. the prompt takes index 0 of every seed rule):

```bash
python -m sd3.generate --arm rediverse --prompt "a photo of a fox in the snow" --num-samples 4 --device cuda:0
# -> outputs/sd3/prompt/a_photo_of_a_fox_in_the_snow/rediverse/{00..03}.png
```

`<slug>` is the prompt in lowercase with every run of other characters replaced by `_` and
leading/trailing `_` removed; a slug longer than 80 characters is cut to 71 characters and followed
by `_` and an 8-character hash of the full prompt. The TF arms (`rediverse_tf`, `rediverse`) read the
prompt's directions from `--directions`, used as given, else from
`outputs/sd3/prompt/<slug>/directions.pt`. When that file does not exist, or holds the directions of
another prompt, the generator first estimates it exactly as `sd3.estimate_directions --prompt` does
(25 x 9 teacher forwards, about 15 s on an H100), frees the teacher, and keeps the file for later
calls on the same prompt.

How the arms run, all in one `StableDiffusion3Pipeline` call per batch of 10 latents of one prompt:

- `teacher`: 28 steps, CFG 7 (empty negative prompt). `student`: 4 steps, guidance 1.0.
- `rediverse_ft`: the student with both adapters active in step 1. In step 1 the transformer runs
  twice: a forward truncated after block 12 with the ReDiverse-FT adapter off reads the student's
  block-12 text stream, and the adapted forward has its block-12 text output replaced by it, so the
  correction acts on the image stream only (the stream the loss supervised). After step 1 the
  ReDiverse-FT adapter is switched off; the TDM adapter stays on at every step.
- `rediverse_tf`: the student with `sum_k alpha_k s_k u_k` added to the block-12 image stream in step 1.
- `rediverse`: `rediverse_ft` with the same directions and coefficients added after the text-stream
  replacement.

## Noise bank

Sample `s` of prompt `i` starts from `torch.randn((1, 16, 128, 128), generator=torch.Generator("cuda")
.manual_seed(i * 1000 + s), dtype=torch.bfloat16)` in every arm. CUDA random draws depend on the
GPU's launch configuration, so the bank built on an H100 is the reference input; it stores the bf16
bit patterns as int16 (read back with `.view(torch.bfloat16)`). Without a bank file the generator
draws the same rule directly on the current GPU, which matches the bank on H100s.

## ReDiverse-TF directions

```bash
python -m sd3.estimate_directions --device cuda:0                 # all 553 prompts
for k in 0 1 2 3; do python -m sd3.estimate_directions --shard $k --num-shards 4 --device cuda:$k & done; wait
python -m sd3.estimate_directions --prompt "a photo of a fox in the snow" --device cuda:0
# -> outputs/sd3/prompt/a_photo_of_a_fox_in_the_snow/directions.pt
```

Per prompt: 25 latents (CPU `randn`, seeds `i * 1000 + 100..124`, disjoint from the evaluation
samples), the teacher rolled conditional-only (no CFG) for 8 steps, block-12 image stream of forward
9 (D = 6,291,456), top-8 principal directions of the centred samples (Gram trick). About 10 s per
prompt; files `data/sd3/directions/p{idx:03d}.pt` (`U` [8, D] bf16, `s` [8], eigenvalues), about
100 MB each, about 56 GB for all 553. The prompt embeddings are cached in `data/sd3/geneval_ctx/`.
A custom prompt (`--prompt`, optional `--out`) uses the latent seeds of prompt index 0.

Each direction file records its prompt: `sd3.estimate_directions` counts an existing file as done,
and `sd3.generate` uses it, only for that prompt. Another prompt file (`--prompts`) therefore needs
its own `--out-dir` and `--ctx-dir` here, and `--directions-dir` and `--out-dir` in `sd3.generate`.

## Training ReDiverse-FT

```bash
python prompts/download_train_captions.py           # prompts/train_captions.txt (100 DiffusionDB captions)
python -m sd3.encode_prompts --device cuda:0        # data/sd3/ctx/p000000..p000099.pt + data/sd3/uncond.pt
python -m sd3.train --device cuda:0                 # outputs/sd3/train, about 61 min on one H100
python -m sd3.convert_weights --src outputs/sd3/train/adapters/rediverse_ft_0000200.pth \
    --out-dir outputs/sd3/release                   # rediverse_ft.safetensors + config.json
python -m sd3.generate --arm rediverse_ft --adapter outputs/sd3/release/rediverse_ft.safetensors
```

The recipe has no flags: 100 captions x 16 paired noises (CPU `randn`, seed
`1_000_000 + (10000 + ci) * 100003 + ni`), 4 captions x 8 noises per optimizer step (four
gradient-accumulation micro-batches of 8), 200 steps, plain MSE in fp32 between the student's step-1
block-12 image stream and the teacher target, AdamW lr 1e-4 (weight decay 0.01, betas 0.9/0.999,
constant), gradients `nan_to_num`'d and clipped to norm 10, seed 0, adapter weights in fp32 on the
bf16 model. Per micro-batch the teacher runs 8 doubled-batch CFG forwards and one forward truncated
after block 12; the student forward is truncated after block 12 with gradient checkpointing. The
first log line reads `it 1  reg_loss 8.5251 ... h_std s/t 6.039/7.217  spread_ratio 0.7104` (`|g|`
1.968); later iterations vary in the fourth digit between reruns because the backward pass is not
bitwise deterministic.
`--steps`, `--out-dir`, `--ctx-dir`, `--uncond` and `--captions` change only the length and the paths.

## Released weights

`sd3/rediverse_ft.safetensors` holds the 304 LoRA tensors (fp32) under
`transformer_blocks.{b}.{module}.lora_{A,B}.weight`; the safetensors metadata and `config.json`
list rank, alpha, the adapted modules, the base models and their revisions.
`sd3.model.load_student_with_adapter` loads it with peft as a second adapter next to the TDM adapter.

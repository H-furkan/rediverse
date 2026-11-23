# Evaluation

The metrics and table scripts behind the main comparison table of the paper: image diversity and
quality on GenEval-553 x 100 seeds (SDXL, SD3), video diversity and quality on VBench-300 x 10 clips
(Wan 2.1, MiniMax-H3), plus the SD3 ablation table, the design-choice tables (supplementary) and the
per-aspect TPIPS table (supplementary).

```
evaluation/
  common/       helpers shared by the image and video metrics
  image/        image metrics: main comparison table (images), SD3 ablation table, design-choice tables,
                image half of the per-aspect TPIPS table
  video/        video metrics: main comparison table (videos), video half of the per-aspect TPIPS table
  tables/       image_table.py, video_table.py: per-set metric files -> the printed paper rows
```

Run everything from the repository root (`python -m evaluation....`). A *set* is the output
directory of one generation arm (`teacher`, `student`, `rediverse_tf`, `rediverse_ft`, `rediverse`),
as written by the backbone packages under `outputs/<backbone>/<arm>/`.

- [Image metrics](#image-metrics): files, inputs, environments, one set, metric definitions,
  paper rows, SD3 ablation
- [Video metrics](#video-metrics): files, inputs, environments, running, outputs, metric
  definitions, tables

## Image metrics

### Image files

```
evaluation/common/
  image_sets.py        <idx:03d>_s<seed>.png sets: grouping by prompt, strict prompts x seeds grid check
  pairwise.py          1 - mean pairwise cosine over a set of embeddings (DreamSim, TPIPS)
  atomic.py            complete-or-absent writes of the caches and metric files
  tpips_factors.py     the pinned TPIPS-Embed checkpoint and the five TPIPS factor texts
evaluation/image/
  config.py            metric model ids and pinned revisions
  extract_feats.py     stage A (CPU): DINOv2-base tokens, CLIP ViT-B/32 image + prompt features, DreamSim
  run_metrics.py       stage H (GPU): LPIPS, HPSv2.1, ImageReward (PickScore on request)
  tpips_embed.py       stage T (GPU): TPIPS-Embed embeddings, one factor per call
  feature_metrics.py   stage B (CPU): DINO, Vendi, DreamSim, CLIPScore, TPIPS from the caches
  merge_shards.py      merges the outputs of run_metrics.py --shard i --num-shards n
  score_many.py        HPSv2.1 + ImageReward for many sets with one model load (used for the Krea 2 sweep)
  evaluate_set.sh      stages A, H, T, B for one set (main protocol)
  score_ablation.sh    the same stages for one SD3 arm on the ablation grid (prompts/geneval_subset50.txt
                       x 100 seeds), + collect_ablation.py
  diversity_lpips.py, quality_hpsv2.py, quality_imagereward.py, quality_pickscore.py   metric models
evaluation/tables/
  image_table.py       per-set metric files -> printed rows (text, CSV, LaTeX)
```

### Image inputs

An image set is the output of one generation arm:
`outputs/<backbone>/<arm>/images/<idx:03d>_s<seed>.png` for every prompt of
`prompts/geneval_553.txt` (`idx` = 0-based line number) and seeds 0..99, i.e. 55,300 images at
1024x1024, generated from the frozen noise bank (see [`sdxl/`](../sdxl/README.md) and
[`sd3/`](../sd3/README.md)). Every stage checks the grid and refuses a set with a missing or extra
file (pairwise metrics are only comparable at a fixed number of samples per prompt).

### Image environments

Two environments are needed, because ImageReward requires `transformers < 4.57` while TPIPS needs
`transformers 5.x` (see [`envs/README.md`](../envs/README.md)):

| Variable | Environment | Stages |
|---|---|---|
| `PY_IMAGE` | `image` (`envs/requirements-image.txt`) | A features, H LPIPS / HPSv2.1 / ImageReward, B metrics, tables |
| `PY_TPIPS` | `tpips` (`envs/requirements-tpips.txt`, includes `tpips==0.1.2`) | T TPIPS embeddings |

`tpips` is installed from PyPI (`tpips==0.1.2`); its package metadata declares the Adobe Research
License (`LicenseRef-Adobe-Research-License`), which permits noncommercial research use only.

Weights download on first use. `facebook/dinov2-base`, `openai/clip-vit-base-patch32`,
`xswu/HPSv2` and `sywang/TPIPS-Embed-Qwen3VL-8B` go into `HF_HOME` at the revisions pinned in
`evaluation/image/config.py` and `evaluation/common/tpips_factors.py`. `tpips` resolves its
`Qwen/Qwen3-VL-Embedding-8B` backbone itself (paper snapshot `2c4565515e0f265c6511776e7193b22c0968ddc7`);
TPIPS-Embed with its backbone is about 17 GB. The `image-reward` package fetches ImageReward-v1.0 into
`~/.cache/ImageReward` whatever `HF_HOME` is (paper snapshot `5736be03b2652728fb87788c9797b0570450ab72`)
and its `bert-base-uncased` tokenizer into `HF_HOME`. The DreamSim ensemble checkpoint goes into
`$DREAMSIM_CACHE` (default `<torch hub dir>/dreamsim`) and the LPIPS VGG16 trunk into the torch hub
directory.

### One image set

```bash
export PY_IMAGE=venvs/image/bin/python PY_TPIPS=venvs/tpips/bin/python
bash evaluation/image/evaluate_set.sh outputs/sdxl/teacher/images outputs/sdxl/teacher --gpu 0 --workers 4 --threads 32
```

writes `outputs/sdxl/teacher/feats/` (per-prompt caches), `metrics_cpu.json` (DINO, Vendi, DreamSim,
CLIPScore, TPIPS overall and aspects) and `metrics_gpu.json` (LPIPS, HPSv2.1, ImageReward), then
prints the set's row. Stages A (CPU), H and T (one GPU, ~25 GB together) run concurrently. Rough
cost per 55,300-image set on one 80 GB GPU: stage A 1-2 h with 4 workers x 32 threads; ImageReward
~1.5 h (one image per call); LPIPS ~2.5 h; TPIPS 2.5-4 h for the overall factor plus ~2 h for the four
aspects. Stages A and T skip prompts whose cache files are newer than their images, so an
interrupted run resumes; JSON outputs are always rewritten.

To spread a set over several GPUs, run the stages directly:

```bash
S=outputs/sd3/rediverse
for i in 0 1 2 3; do   # stage A, CPU
  $PY_IMAGE -m evaluation.image.extract_feats --images-dir $S/images --out $S/feats --shard $i --num-shards 4 --threads 32 &
done
for i in 0 1 2 3; do   # stage H, one GPU per shard
  CUDA_VISIBLE_DEVICES=$i $PY_IMAGE -m evaluation.image.run_metrics --images-dir $S/images --out $S/metrics_gpu_s${i}of4.json --shard $i --num-shards 4 &
done
wait
$PY_IMAGE -m evaluation.image.merge_shards --out $S/metrics_gpu.json $S/metrics_gpu_s{0,1,2,3}of4.json
# stage T: any number of workers per factor share the prompts through claim files (one host)
CUDA_VISIBLE_DEVICES=0 $PY_TPIPS -m evaluation.image.tpips_embed --images-dir $S/images --out $S/feats --factor overall &
CUDA_VISIBLE_DEVICES=1 $PY_TPIPS -m evaluation.image.tpips_embed --images-dir $S/images --out $S/feats --factor overall &
wait
for f in subject background camera lighting; do
  CUDA_VISIBLE_DEVICES=0 $PY_TPIPS -m evaluation.image.tpips_embed --images-dir $S/images --out $S/feats \
      --factor $f --prompt-ids prompts/geneval_subset100.txt
done
$PY_IMAGE -m evaluation.image.feature_metrics --feats $S/feats --images-dir $S/images --out $S/metrics_cpu.json
```

### Image metric definitions

Paper App. F. Diversity values are computed per prompt over all C(100, 2) = 4950 pairs of its 100
samples and averaged over prompts; quality values are averaged over all images.

| Column | Definition | Model, input |
|---|---|---|
| DINO | 1 - mean over pairs of the mean cosine of the 256 aligned patch tokens (CLS dropped) | DINOv2-base, HF processor (resize 256, center crop 224), CPU fp32, tokens cached in fp16 |
| DS | 1 - mean pairwise cosine of the embeddings (= mean DreamSim distance) | DreamSim 0.2.1 ensemble, 224x224, CPU fp32 |
| LP | mean pairwise LPIPS distance | LPIPS 0.1.4 VGG, 256x256 in [-1, 1], GPU fp32 |
| TP | mean of the five TPIPS cells below, each rounded to one decimal | |
| TPIPS factor | 1 - mean pairwise cosine of the L2-normalised embeddings | TPIPS-Embed (bf16), native resolution, batch 16; `overall` on all 553 prompts, `the main subject`, `the background scene`, `lighting and color`, `the camera viewpoint and framing` on the 100 prompts of `prompts/geneval_subset100.txt` |
| Vendi | exp(entropy) of the eigenvalues of K = C C^T (symmetrised, + 1e-6 I, normalised to sum 1) | DINOv2-base CLS tokens, L2-normalised |
| CS | cos(image, prompt), raw cosine (Hessel's 2.5 max(cos, 0) is stored as `hessel`) | CLIP ViT-B/32, 224 center crop, text truncated to 77 tokens, CPU fp32 |
| HPS | cos(image, prompt) | HPSv2.1 (OpenCLIP ViT-H-14 + `HPS_v2.1_compressed.pt`), fp16, batch 32 |
| IR | ImageReward score | ImageReward-v1.0 (image-reward 1.5), one image per call, fp32 |

The feature caches are computed on CPU in fp32 only; the same models on GPU in fp16 would change
per-prompt DINO in the 4th-5th decimal and Vendi in the 2nd-3rd. LPIPS pairs are scored in chunks
(`--lpips-pair-batch`, default 64) whose sums are divided by the total pair count, so the chunk size
does not change the value.

### Image table

```bash
python -m evaluation.tables.image_table \
    --block SDXL teacher=outputs/sdxl/teacher student=outputs/sdxl/student rediverse_tf=outputs/sdxl/rediverse_tf \
                 rediverse_ft=outputs/sdxl/rediverse_ft rediverse=outputs/sdxl/rediverse \
    --block SD3  teacher=outputs/sd3/teacher student=outputs/sd3/student rediverse_tf=outputs/sd3/rediverse_tf \
                 rediverse_ft=outputs/sd3/rediverse_ft rediverse=outputs/sd3/rediverse \
    --csv outputs/image_table.csv --latex outputs/image_table.tex
```

prints, per row, DINO, DS, LP, TP, Vendi, CS, HPS, IR (the eight columns of the main comparison
table, images) and the five TPIPS values (the per-aspect TPIPS table, supplementary). Cells are x100
with one decimal, except Vendi (two decimals) and IR (three decimals). `--latex` marks the displayed
values of the non-reference rows (`--reference`, default `teacher`): all rows tied at the best are
bold, and the second-best value is underlined only when the best is unique. Further arms (e.g. a
baseline scored with the same scripts) are ranked with the others.

### SD3 ablation

The SD3 ablation table and the design-choice tables (supplementary). Each arm is a directory with
`images/` holding exactly the 50 prompts of `prompts/geneval_subset50.txt` x seeds 0..99. The release
generates the reference arms; the scorer evaluates any SD3 variant from its `images/` directory on the
same 50 x 100 grid. The reference arms (teacher, student, ReDiverse-FT, ReDiverse-TF) are the matching
subsets of the main SD3 sets (same noise bank and generation batch, hence identical images), e.g.

```bash
mkdir -p outputs/sd3_ablation/teacher/images
for i in $(tr , ' ' < prompts/geneval_subset50.txt); do
  ln outputs/sd3/teacher/images/$(printf %03d $i)_s{0..99}.png outputs/sd3_ablation/teacher/images/
done
```

```bash
bash evaluation/image/score_ablation.sh outputs/sd3_ablation/<arm> --gpu 0
python -m evaluation.tables.image_table --protocol ablation \
    --block SD3 teacher=outputs/sd3_ablation/teacher student=outputs/sd3_ablation/student \
                rediverse_ft=outputs/sd3_ablation/rediverse_ft rediverse_tf=outputs/sd3_ablation/rediverse_tf <arm>=...
```

The scorer runs stages A, H (HPSv2.1 and ImageReward; no LPIPS), T (overall and the four aspects, all
on the 50 prompts) and B, then collects `<arm>/metrics_ablate.json`. The ablation table prints DINO,
DS, TPIPS (overall; the column of the design-choice tables), Vendi, CS, HPS, IR and TP (the five-factor
mean of the SD3 ablation table).

## Video metrics

The main comparison table (videos) for Wan 2.1 and MiniMax-H3 (DINO, VJ, TP, Cam, VC, SC, BC, VCS) and
the video rows of the per-aspect TPIPS table (supplementary: per-aspect TPIPS, CamEnd, ViCLIP Vendi).

### Video files

```
evaluation/video/
  sets.py            prompt list, prompt subsets, per-prompt clip groups (every prompt needs all 10 clips)
  video_dino.py      DINOv2 frame CLS (16 frames) + V-JEPA 2 clip embeddings (64 frames)      env: tpips
  tpips_video.py     TPIPS-Embed frame embeddings (8 frames), one aspect per call               env: tpips
  viclip_scores.py   ViCLIP clip + prompt embeddings (VBench 8-frame 'middle' sampling)        env: vbench
  vbench_run.py      VBench subject / background consistency                                  env: vbench
  camtraj_vipe.py    ViPE camera poses -> camera-trajectory statistics                         env: vipe
  video_metrics.py   feature caches -> <set>/metrics_video.json                                env: tpips (numpy)
  evaluate_set.sh    all of the above for one set, stage by stage
  vipe_pose_only.yaml  ViPE pipeline config used for the camera poses
evaluation/tables/
  video_table.py     <set>/metrics_video.json + <set>/vbench/*.json -> the paper rows
```

### Video inputs

A video set is a directory with `videos/<prompt>-<k>.mp4` (VBench naming, `k = 0..9`) for all 300
prompts of `prompts/vbench_300.txt`, i.e. 3000 clips. The generators write the sets to
`outputs/wan2.1/<arm>/` and `outputs/minimax-h3/<arm>/` (`$REDIVERSE_OUT` relocates `outputs/`).
All clips are H.264 (libx264, CRF 17, yuv420p) and every metric decodes them with the system
`ffmpeg`, so a set generated elsewhere must use the same encoder settings.

The prompt index used everywhere is the 0-based line number of `prompts/vbench_300.txt`. Pairwise
diversity grows with the number of clips per prompt, so it is always 10: the extractors stop
with an error if a clip of a requested prompt is missing, and `video_metrics.py` stops if a cache holds
anything other than the expected prompts with 10 clips each.

### Video environments

Three environments from [`envs/`](../envs/README.md) are needed, because the metric models pin
incompatible versions:

| Environment | Used for | Install |
|---|---|---|
| `tpips`  | DINOv2, V-JEPA 2, TPIPS features; aggregation (`video_table.py` needs only the standard library) | `envs/requirements-tpips.txt` |
| `vbench` | ViCLIP features; VBench subject / background consistency | `envs/requirements-vbench.txt` |
| `vipe`   | ViPE camera poses | [`envs/vipe.md`](../envs/vipe.md) |

`ffmpeg` / `ffprobe` (with libx264) must be on `PATH`. For the `vipe` environment, also install the
pose-only pipeline config shipped here (identical to the one in `envs/vipe.md`):

```bash
CFG=$(venvs/vipe/bin/python -c "import os, vipe; print(os.path.join(os.path.dirname(vipe.__file__), '_configs', 'pipeline'))")
cp evaluation/video/vipe_pose_only.yaml "$CFG/pose_only.yaml"
```

Model weights are downloaded on first use:

| Metric | Model | Source (snapshot) | Cache |
|---|---|---|---|
| DINO | DINOv2-B | `facebook/dinov2-base` (`f9e44c814b77203eaa57a6bdbbd535f21ede1415`, pinned in code) | `HF_HOME` |
| VJ | V-JEPA 2 ViT-L | `facebook/vjepa2-vitl-fpc64-256` (`b3c1679b7c34d3255ef3547f27c7b226aefab26f`, pinned in code) | `HF_HOME` |
| TP | TPIPS-Embed | `sywang/TPIPS-Embed-Qwen3VL-8B` (`86a1493c60121e656a1c075bc7aed88f25fcd4d4`, pinned in code) on `Qwen/Qwen3-VL-Embedding-8B` (`2c4565515e0f265c6511776e7193b22c0968ddc7`, resolved by `tpips`) | `HF_HOME` |
| VC, VCS | ViCLIP | `ViClip-InternVid-10M-FLT.pth` from `OpenGVLab/VBench_Used_Models` + CLIP's `bpe_simple_vocab_16e6.txt.gz`, fetched by VBench | `VBENCH_CACHE_DIR` (default `~/.cache/vbench`) |
| SC | DINO ViT-B/16 | `torch.hub` `facebookresearch/dino:main` (`dino_vitbase16_pretrain.pth`) | `~/.cache/torch/hub` |
| BC | CLIP ViT-B/32 | OpenAI `clip` package (`ViT-B-32.pt`) | `~/.cache/clip` |
| Cam | ViPE 1.2.0 | GeoCalib pinhole, DROID-SLAM, UniDepth v2 ViT-L, fetched by ViPE | `~/.cache/torch/hub`, `HF_HOME` |

Environment variables: `HF_HOME` is honoured and never set by the code. The `vbench` stage runs with
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, as in the reference runs (set by `evaluate_set.sh`). ViPE runs as
`$VIPE_BIN` (default: the `vipe` script next to the interpreter that runs `camtraj_vipe.py`); if
`VIPE_CUDA_HOME` is set, its `bin/` and `lib64/` are put in front of `PATH` / `LD_LIBRARY_PATH` of the
ViPE process, which needs the CUDA 12.6 toolkit it was built against at run time.

### One video set

One set, one GPU, all stages:

```bash
CUDA_VISIBLE_DEVICES=0 evaluation/video/evaluate_set.sh outputs/wan2.1/rediverse_ft
```

`PY_TPIPS`, `PY_VBENCH` and `PY_VIPE` point to the interpreters of the three environments (default
`venvs/{tpips,vbench,vipe}/bin/python` under the repository root); the script runs from any directory.
Stages can be run separately, e.g. `evaluate_set.sh <set> vipe` on the machine that has ViPE and
`evaluate_set.sh <set> metrics` at the end. The individual commands are:

```bash
SET=outputs/wan2.1/rediverse_ft
# env tpips
python -m evaluation.video.video_dino   --videos-dir $SET/videos --out $SET/feats
python -m evaluation.video.tpips_video  --videos-dir $SET/videos --out $SET/feats --factor overall      # all 300 prompts
python -m evaluation.video.tpips_video  --videos-dir $SET/videos --out $SET/feats --factor subject      # subset100; also
                                                                                                       # background, lighting, camera
# env vbench (TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1)
python -m evaluation.video.viclip_scores --videos-dir $SET/videos --out $SET/feats
python -m evaluation.video.vbench_run    --set $SET                                                    # both dimensions
# env vipe
python -m evaluation.video.camtraj_vipe  --videos-dir $SET/videos --out $SET/feats                      # subset30
# env tpips
python -m evaluation.video.video_metrics --feats $SET/feats --out $SET/metrics_video.json
```

Several workers can share a set: `video_dino.py` and `viclip_scores.py` take `--shard i --num-shards n`,
while `tpips_video.py` and `camtraj_vipe.py` claim prompts through `<set>/feats/<metric>/.claims/` (start
the same command on several GPUs). A claim whose process has exited on the same host is taken over; remove
the `.claims` directories before restarting workers on a different host. Every cache and json is written to
a temporary file and renamed, so an interrupted run leaves no partial file. Every extractor skips prompts
whose cache exists and `vbench_run.py` skips a dimension whose json exists; delete them to recompute.

Cost: TPIPS-Embed needs about 24 GB of GPU memory per worker. ViPE needs about 10 GB and takes about 1.5 min
per Wan 2.1 clip and 2 min per MiniMax-H3 clip, i.e. 8-11 GPU hours per set for the 300 clips of the subset.

### Video outputs

```
<set>/feats/vdino/dinov2/p<idx>.npz        cls [10, 16, 768] fp16 (L2-normalised CLS per frame), prompt
<set>/feats/vjepa2/p<idx>.npy              [10, 1024] fp16
<set>/feats/tpips_video/<aspect>/p<idx>.npy  [10, 8, 4096] fp16    (overall: 300 prompts; aspects: 100)
<set>/feats/viclip/p<idx>.npz              video [10, 768] fp16, text [768] fp16, prompt
<set>/feats/camtraj/p<idx>.npz             trajectory_norm, endpoint_norm, trajectory_distance, endpoint_distance,
                                           traj_pairs [10, 10], endpoint_pairs [10, 10], path_length [10], prompt
<set>/feats/vipe/p<idx>/                   ViPE poses (pose/*.npz, [T, 4, 4] camera-to-world) and intrinsics
<set>/vbench/{subject,background}_consistency.json   mean, n_prompts, n_videos, per_video scores
<set>/metrics_video.json                   metrics.<key>.{mean, std_over_prompts, n_prompts, per_prompt}
```

### Video metric definitions

`pair_div(X) = 1 - mean off-diagonal cosine` over the rows of `X` (the 10 clips of a prompt, 45 pairs).
Every diversity metric is computed per prompt and averaged over prompts; the tables print it x100.

| Column | Key in `metrics_video.json` | Definition | Prompts |
|---|---|---|---|
| DINO | `dino_frame_dinov2` | 16 frames at `linspace(0, T-1, 16).round()`; DINOv2-B processor (shortest edge 256, bicubic, centre crop 224); CLS token after the final norm, fp16; `pair_div` per frame index, averaged over the 16 indices | 300 |
| VJ | `vjepa2` | 64 frames at `linspace(0, T-1, 64).round()`, scaled to 256x256 by ffmpeg (aspect ratio not kept), V-JEPA 2 processor (shortest edge 292, centre crop 256); mean over all tokens; `pair_div` | 300 |
| TP | `tpips_video_{overall,subject,background,lighting,camera}` | 8 frames at `linspace(0, T-1, 8).round()` at native resolution; TPIPS-Embed with the aspect instruction; `pair_div` per frame index, averaged over the 8 indices | overall 300, aspects 100 (`vbench_subset100`) |
| Cam | `camtraj_traj_norm` | ViPE pose-only camera-to-world poses; positions in the first-camera frame `R_0^T (p_t - p_0)`; resampled to 32 points on normalised time; each trajectory divided by its own path length (+1e-6); mean over clip pairs of the mean pointwise distance | 30 (`vbench_subset30`) |
| CamEnd | `camtraj_endpoint_norm` | as Cam, distance between the endpoints | 30 |
| VC | `viclip_div` | ViCLIP `encode_vision` of 8 frames (middle of 8 equal segments, VBench `clip_transform(224)`); `pair_div` | 300 |
| Vendi | `viclip_vendi` | `exp(-sum w log w)`, `w` = normalised eigenvalues of `X X^T / 10` on the ViCLIP embeddings (unscaled) | 300 |
| SC | `<set>/vbench/subject_consistency.json` `mean` | VBench 0.1.5, custom_input mode, on the prompts VBench_full_info.json assigns to the dimension; mean over clips | 23 (230 clips) |
| BC | `<set>/vbench/background_consistency.json` `mean` | as SC | 27 (270 clips) |
| VCS | `viclip_text` | mean over the 10 clips of the cosine between the ViCLIP clip embedding and the ViCLIP text embedding of the prompt (VBench's overall-consistency computation, on all 300 prompts) | 300 |

TP in the main comparison table (videos) is the mean of the five aspect values *as printed* (each x100
and rounded to one decimal), rounded to one decimal; the overall aspect averages 300 prompts and the four
others the 100-prompt subset.

### Video table

```bash
python -m evaluation.tables.video_table --backbone wan                    # main comparison table (videos), Wan 2.1
python -m evaluation.tables.video_table --backbone h3                     # MiniMax-H3
python -m evaluation.tables.video_table --backbone wan --table aspects    # per-aspect TPIPS table, video rows
```

The script reads `outputs/<backbone>/<arm>/{metrics_video.json, vbench/*.json}` for the five arms (or
`--root DIR`, or `--set ARM=DIR` per arm) and checks that every value covers its full prompt set. It prints
a Markdown table (`--format tex` for LaTeX rows; `--out FILE` to save it). Best and second-best values among
all rows except the teacher are marked bold / underlined on the printed values: all rows tied at the best
are bold, and the second-best value is marked only when the best is unique. `--extra NAME=DIR` adds another
ranked row (e.g. a baseline evaluated with the same scripts) after the student.

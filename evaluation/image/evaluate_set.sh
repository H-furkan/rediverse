#!/usr/bin/env bash
# Evaluate ONE image sample set with the paper's protocol (553 GenEval prompts x 100 seeds).
#
#   PY_IMAGE=venvs/image/bin/python PY_TPIPS=venvs/tpips/bin/python \
#   bash evaluation/image/evaluate_set.sh <images_dir> <out_dir> [--gpu N] [--workers W] [--threads T]
#        [--seeds R] [--prompt-ids SPEC] [--aspect-ids SPEC] [--no-gpu-metrics] [--no-tpips] [--no-aspects]
#
# <images_dir> holds <prompt_idx:03d>_s<seed>.png for every prompt of prompts/geneval_553.txt (or of
# --prompt-ids, a comma list / a-b ranges / a file) and seeds 0..R-1 (R = 100); any other grid is refused.
# The four TPIPS aspects use --aspect-ids (default prompts/geneval_subset100.txt). Two interpreters are
# needed because ImageReward requires transformers < 4.57 and TPIPS transformers 5.x:
#   PY_IMAGE  python of the `image` environment (envs/requirements-image.txt): stages A, H, B
#   PY_TPIPS  python of the `tpips` environment (envs/requirements-tpips.txt): stage T
# Stages (A, H and T run concurrently; T and H share one GPU, ~25 GB together):
#   A  CPU  evaluation.image.extract_feats    W workers x T threads -> <out_dir>/feats/{dino,clip,dreamsim}/
#   H  GPU  evaluation.image.run_metrics      lpips hpsv2 imagereward -> <out_dir>/metrics_gpu.json
#   T  GPU  evaluation.image.tpips_embed      overall, then the four aspects -> <out_dir>/feats/tpips/<factor>/
#   B  CPU  evaluation.image.feature_metrics  dino vendi dreamsim clipscore tpips_* -> <out_dir>/metrics_cpu.json
# then the set's row (evaluation.tables.image_table). Logs go to <out_dir>/logs/. Stages A and T skip
# prompts whose caches are newer than their images.
set -uo pipefail
case "${1:-}" in -h|--help|"") sed -n '2,/^set /{/^#/p;}' "$0"; exit 0;; esac
IMAGES=${1:?images_dir}; OUT=${2:?out_dir}; shift 2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU=0; W=4; T=32; SEEDS=100; IDS=""; ASPECT_IDS=$ROOT/prompts/geneval_subset100.txt; HEAVY=1; TPIPS=1; ASPECTS=1
while [ $# -gt 0 ]; do case $1 in
  --gpu) GPU=$2; shift 2;; --workers) W=$2; shift 2;; --threads) T=$2; shift 2;; --seeds) SEEDS=$2; shift 2;;
  --prompt-ids) IDS=$2; shift 2;; --aspect-ids) ASPECT_IDS=$2; shift 2;;
  --no-gpu-metrics) HEAVY=0; shift;; --no-tpips) TPIPS=0; ASPECTS=0; shift;; --no-aspects) ASPECTS=0; shift;;
  *) echo "unknown argument $1"; exit 1;; esac; done
: "${PY_IMAGE:?set PY_IMAGE to the python of the image environment}"
[ "$TPIPS" = 0 ] || : "${PY_TPIPS:?set PY_TPIPS to the python of the tpips environment (or pass --no-tpips)}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
GRID=(--seeds "$SEEDS"); [ -n "$IDS" ] && GRID+=(--prompt-ids "$IDS")
LOG=$OUT/logs; mkdir -p "$LOG"
log() { echo "[eval] $* $(date -u +%H:%M:%S)"; }

log "A: features on CPU, $W workers x $T threads"
PIDS_A=()
for i in $(seq 0 $((W-1))); do
  OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T CUDA_VISIBLE_DEVICES="" "$PY_IMAGE" -m evaluation.image.extract_feats \
      --images-dir "$IMAGES" --out "$OUT/feats" "${GRID[@]}" --shard "$i" --num-shards "$W" --threads "$T" > "$LOG/feats_$i.log" 2>&1 &
  PIDS_A+=($!)
done
if [ "$HEAVY" = 1 ]; then
  log "H: lpips hpsv2 imagereward on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$PY_IMAGE" -m evaluation.image.run_metrics \
      --images-dir "$IMAGES" "${GRID[@]}" --out "$OUT/metrics_gpu.json" --metrics lpips hpsv2 imagereward \
      --device cuda:0 > "$LOG/gpu_metrics.log" 2>&1 &
  PID_H=$!
fi
RT=0
if [ "$TPIPS" = 1 ]; then
  log "T: TPIPS overall on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY_TPIPS" -m evaluation.image.tpips_embed \
      --images-dir "$IMAGES" --out "$OUT/feats" "${GRID[@]}" --factor overall > "$LOG/tpips_overall.log" 2>&1 || RT=1
  if [ "$ASPECTS" = 1 ]; then
    for asp in subject background camera lighting; do
      log "T: TPIPS $asp on GPU $GPU"
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY_TPIPS" -m evaluation.image.tpips_embed \
          --images-dir "$IMAGES" --out "$OUT/feats" --seeds "$SEEDS" --factor "$asp" --prompt-ids "$ASPECT_IDS" \
          > "$LOG/tpips_$asp.log" 2>&1 || RT=1
    done
  fi
fi
RA=0; for p in "${PIDS_A[@]}"; do wait "$p" || RA=1; done
RH=0; if [ -n "${PID_H:-}" ]; then wait "$PID_H" || RH=1; fi
if [ "$RA$RH$RT" != 000 ]; then log "FAILED (A=$RA H=$RH T=$RT); see $LOG"; exit 5; fi

METRICS=dino,vendi,dreamsim,clipscore
[ "$TPIPS" = 1 ] && METRICS=$METRICS,tpips_overall
[ "$ASPECTS" = 1 ] && METRICS=$METRICS,tpips_subject,tpips_background,tpips_camera,tpips_lighting
log "B: $METRICS from the caches"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY_IMAGE" -m evaluation.image.feature_metrics --feats "$OUT/feats" \
    --images-dir "$IMAGES" "${GRID[@]}" --metrics "$METRICS" --aspect-ids "$ASPECT_IDS" --out "$OUT/metrics_cpu.json" \
    | tee "$LOG/feature_metrics.log" || { log "FAILED (B); see $LOG"; exit 5; }
if [ -z "$IDS" ]; then
  "$PY_IMAGE" -m evaluation.tables.image_table --seeds "$SEEDS" --block "$(basename "$OUT")" "set=$OUT"
fi
log "done: $OUT/metrics_cpu.json$([ "$HEAVY" = 1 ] && echo " $OUT/metrics_gpu.json")"

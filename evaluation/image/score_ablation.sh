#!/usr/bin/env bash
# Score ONE SD3 arm on the grid of the SD3 ablation table and the design-choice tables (supplementary):
# 50 prompts (prompts/geneval_subset50.txt, the first 50 of prompts/geneval_subset100.txt) x 100 seeds, with
# the main-table metric code. The release generates the reference arms; the scorer evaluates any SD3 variant
# from its images/ directory on the same 50 x 100 grid.
#
#   PY_IMAGE=venvs/image/bin/python PY_TPIPS=venvs/tpips/bin/python \
#   bash evaluation/image/score_ablation.sh <arm_dir> [--gpu N] [--threads T] [--seeds 100] [--prompts FILE]
#                                           [--feats-only] [--force]
#
# <arm_dir>/images must hold exactly the prompts of FILE x seeds 0..R-1 (anything else exits 4).
# Stages (A, H and T run concurrently; ~25 GB of GPU memory):
#   A  CPU  evaluation.image.extract_feats     -> <arm_dir>/eval/feats/{dino,clip,dreamsim}/
#   H  GPU  evaluation.image.run_metrics       hpsv2 imagereward -> <arm_dir>/eval/metrics_gpu.json (no LPIPS)
#   T  GPU  evaluation.image.tpips_embed       overall + the four aspects, all on the same prompts
#   B  CPU  evaluation.image.feature_metrics   -> <arm_dir>/eval/metrics_cpu.json
#   C  CPU  evaluation.image.collect_ablation  -> <arm_dir>/metrics_ablate.json, then <arm_dir>/EVAL_DONE
# --feats-only runs stage A alone (no GPU); a later full run reuses the caches.
# Logs: <arm_dir>/eval/*.log. Rows: python -m evaluation.tables.image_table --protocol ablation --block SD3 name=<arm_dir> ...
set -uo pipefail
case "${1:-}" in -h|--help|"") sed -n '2,/^set /{/^#/p;}' "$0"; exit 0;; esac
ARM_DIR=$1; shift
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU=0; T=16; SEEDS=100; PROMPTS=$ROOT/prompts/geneval_subset50.txt; FEATS_ONLY=0; FORCE=0
while [ $# -gt 0 ]; do case $1 in
  --gpu) GPU=$2; shift 2;; --threads) T=$2; shift 2;; --seeds) SEEDS=$2; shift 2;; --prompts) PROMPTS=$2; shift 2;;
  --feats-only) FEATS_ONLY=1; shift;; --force) FORCE=1; shift;; *) echo "unknown argument $1"; exit 1;; esac; done
: "${PY_IMAGE:?set PY_IMAGE to the python of the image environment}"
[ "$FEATS_ONLY" = 1 ] || : "${PY_TPIPS:?set PY_TPIPS to the python of the tpips environment}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
IMAGES=$ARM_DIR/images; EV=$ARM_DIR/eval; NAME=$(basename "$ARM_DIR")
log() { echo "[score $NAME] $* $(date -u +%H:%M:%S)"; }

# ---- preconditions: exactly the NP prompts of $PROMPTS x SEEDS seeds ------------------------------------------
[ -d "$IMAGES" ] || { log "no images dir $IMAGES"; exit 4; }
IDS=$(tr ',' '\n' < "$PROMPTS" | tr -s ' \n' '\n' | sed '/^$/d')
NP=$(echo "$IDS" | wc -l)
WANT=$(for i in $IDS; do for ((s=0; s<SEEDS; s++)); do printf "%03d_s%d.png\n" "$i" "$s"; done; done | sort)
HAVE=$(ls "$IMAGES" | grep -E '^[0-9]+_s[0-9]+\.png$' | sort)
if [ "$WANT" != "$HAVE" ]; then
  log "$IMAGES is not exactly $NP prompts x $SEEDS seeds ($(comm -12 <(echo "$WANT") <(echo "$HAVE") | wc -l) of the grid present, $(echo "$HAVE" | grep -c .) png in total)"; exit 4
fi
if [ "$FORCE" = 0 ] && [ -f "$ARM_DIR/EVAL_DONE" ] && [ -f "$ARM_DIR/metrics_ablate.json" ]; then
  log "already scored ($ARM_DIR/EVAL_DONE exists); --force to redo"; exit 0; fi
mkdir -p "$EV"
GRID=(--prompt-ids "$PROMPTS" --seeds "$SEEDS")

# ---- stage A (CPU features) ------------------------------------------------------------------------------------
log "A: feature caches (dino/clip/dreamsim, $T threads, CPU)"
OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T CUDA_VISIBLE_DEVICES="" "$PY_IMAGE" -m evaluation.image.extract_feats \
    --images-dir "$IMAGES" --out "$EV/feats" "${GRID[@]}" --threads "$T" > "$EV/feats.log" 2>&1 &
PA=$!
if [ "$FEATS_ONLY" = 1 ]; then
  wait $PA; RA=$?; log "A: rc=$RA"; exit $((RA == 0 ? 0 : 5)); fi
# ---- stage H (GPU quality) -------------------------------------------------------------------------------------
log "H: hpsv2 + imagereward on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$PY_IMAGE" -m evaluation.image.run_metrics \
    --images-dir "$IMAGES" "${GRID[@]}" --out "$EV/metrics_gpu.json" --metrics hpsv2 imagereward --device cuda:0 \
    > "$EV/gpu_metrics.log" 2>&1 &
PH=$!
# ---- stage T (GPU TPIPS embeddings) ----------------------------------------------------------------------------
RT=0
for F in overall subject background camera lighting; do
  log "T: TPIPS $F on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY_TPIPS" -m evaluation.image.tpips_embed \
      --images-dir "$IMAGES" --out "$EV/feats" "${GRID[@]}" --factor $F > "$EV/tpips_$F.log" 2>&1 || RT=1
done
wait $PA; RA=$?; wait $PH; RH=$?
log "A: rc=$RA  H: rc=$RH  T: rc=$RT"

# ---- stage B (cache metrics) and C (collect) -------------------------------------------------------------------
log "B: metrics from caches"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY_IMAGE" -m evaluation.image.feature_metrics --feats "$EV/feats" \
    --images-dir "$IMAGES" "${GRID[@]}" --aspect-ids "$PROMPTS" --out "$EV/metrics_cpu.json" > "$EV/metrics.log" 2>&1; RB=$?
log "C: collect -> $ARM_DIR/metrics_ablate.json"
"$PY_IMAGE" -m evaluation.image.collect_ablation --arm-dir "$ARM_DIR" --n-prompts "$NP" --seeds "$SEEDS"; RC=$?
if [ "$RA$RH$RT$RB$RC" = 00000 ]; then
  echo "scored $NP x $SEEDS" > "$ARM_DIR/EVAL_DONE"; log "DONE"; exit 0
fi
log "FAILED rc A=$RA H=$RH T=$RT B=$RB C=$RC (see $EV/*.log)"; exit 5

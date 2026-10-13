#!/usr/bin/env bash
# Video metrics of one set: <set>/videos/<prompt>-<k>.mp4, 300 prompts of prompts/vbench_300.txt x 10 clips.
#
# The metric stack spans three Python environments with incompatible pins (envs/README.md):
#   tpips   DINOv2 + V-JEPA 2 features, TPIPS features, aggregation   (torch 2.13, transformers 5.x)
#   vbench  ViCLIP features, VBench subject / background consistency  (torch 2.7, vbench 0.1.5, transformers 4.33)
#   vipe    ViPE camera poses for the camera-trajectory metric         (torch 2.7, nvidia-vipe 1.2.0, CUDA 12.6 toolkit)
# Stages run in this order by default; each can be run alone, e.g. on a machine that has only one of the
# environments. Every extractor skips prompts whose cache already exists, so a stage can be re-run to resume.
#
#   evaluation/video/evaluate_set.sh <set dir> [tpips] [vbench] [vipe] [metrics]
#
# Runs from any directory; a relative <set dir> is resolved against the current directory.
# Environment variables:
#   PY_TPIPS, PY_VBENCH, PY_VIPE   interpreters of the three environments (default <repo>/venvs/<env>/bin/python)
#   CUDA_VISIBLE_DEVICES           the GPU to use (TPIPS-Embed needs about 24 GB, ViPE about 10 GB)
#   HF_HOME                        Hugging Face cache (DINOv2, V-JEPA 2, TPIPS-Embed, ViPE's UniDepth)
#   VBENCH_CACHE_DIR               VBench weights, ViCLIP included (VBench's default: ~/.cache/vbench)
#   VIPE_CUDA_HOME                 CUDA 12.6 toolkit prepended to PATH / LD_LIBRARY_PATH of ViPE (envs/vipe.md)
#   VIPE_BIN                       ViPE executable (default: `vipe` next to $PY_VIPE)
set -euo pipefail
case "${1:-}" in -h|--help|"") sed -n '2,/^set /{/^#/p;}' "$0"; exit 0;; esac

SET=$1
shift
STAGES=${*:-tpips vbench vipe metrics}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY_TPIPS=${PY_TPIPS:-$ROOT/venvs/tpips/bin/python}
PY_VBENCH=${PY_VBENCH:-$ROOT/venvs/vbench/bin/python}
PY_VIPE=${PY_VIPE:-$ROOT/venvs/vipe/bin/python}
VIDEOS="$SET/videos"
FEATS="$SET/feats"

for stage in $STAGES; do
  case "$stage" in
    tpips)
      # DINO (16 frames, CLS) and VJ (64 frames) -> feats/vdino/dinov2, feats/vjepa2
      "$PY_TPIPS" -m evaluation.video.video_dino --videos-dir "$VIDEOS" --out "$FEATS"
      # TP: overall on all 300 prompts, the four aspects on prompts/vbench_subset100.txt -> feats/tpips_video/<aspect>
      for factor in overall subject background lighting camera; do
        "$PY_TPIPS" -m evaluation.video.tpips_video --videos-dir "$VIDEOS" --out "$FEATS" --factor "$factor"
      done
      ;;
    vbench)
      # VC, VCS (ViCLIP, VBench 'middle' 8-frame sampling) -> feats/viclip
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$PY_VBENCH" -m evaluation.video.viclip_scores --videos-dir "$VIDEOS" --out "$FEATS"
      # SC, BC on VBench's own prompt subsets of the dimensions -> <set>/vbench/<dimension>.json
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$PY_VBENCH" -m evaluation.video.vbench_run --set "$SET"
      ;;
    vipe)
      # Cam, CamEnd on prompts/vbench_subset30.txt -> feats/camtraj (poses under feats/vipe)
      "$PY_VIPE" -m evaluation.video.camtraj_vipe --videos-dir "$VIDEOS" --out "$FEATS"
      ;;
    metrics)
      "$PY_TPIPS" -m evaluation.video.video_metrics --feats "$FEATS" --out "$SET/metrics_video.json"
      ;;
    *)
      echo "unknown stage '$stage' (expected tpips, vbench, vipe or metrics)" >&2
      exit 2
      ;;
  esac
done

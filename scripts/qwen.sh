#!/usr/bin/env bash
# ReDiverse on a Qwen-Image student, end to end on one node: bash scripts/qwen.sh <student> <stage> [<stage> ...]
#
#   preview      gap gate: teacher and student on 6 GenEval prompts x 8 seeds (prompts 0,11,17,27,28,34)
#   probe        activation probe of the teacher and the student (probe_video20 prompts, N = 200), then the report
#   bank         ReDiverse-FT target bank (teacher forward TAU, block BLOCK) for the 100 training captions
#   train        ReDiverse-FT training on all GPUs (torchrun, data parallel)
#   directions   ReDiverse-TF directions (teacher forward TAU, block BLOCK) for GenEval subset100
#   sweep        FT / TF / FT+TF previews (bands 2 and 4) with the trained adapter
#   generate     student, rediverse_ft, rediverse (BAND) and teacher on GenEval subset100 x 16
#
# BLOCK and TAU come from the probe (and qwenimage.model.teacher_target_sigmas); GPUS lists the devices (default 0 1 2 3);
# PY the interpreter of an environment with diffusers >= 0.43 and transformers >= 5.17; ADAPTER the FT adapter used
# by sweep/generate (default: step 200 of the train stage); BAND the TF alpha range of generate (default 2); RAM_GB the
# probe's host staging budget per worker (default 150: the 256 GB default was OOM-killed in a 150 GB job).
set -euo pipefail
cd "$(dirname "$0")/.."
S=${1:?student}; shift
PY=${PY:-python}
read -r -a G <<< "${GPUS:-0 1 2 3}"
OUT=${REDIVERSE_OUT:-outputs}
FAM=$("$PY" -c "from qwenimage import config as C; print(C.family_of('$S'))")
BK=$("$PY" -c "from qwenimage import config as C; print(C.family('$FAM')['backbone'])")
BLOCK=${BLOCK:-}; TAU=${TAU:-}; BAND=${BAND:-2}
ADAPTER=${ADAPTER:-$OUT/$BK/$S/train/${S}_b${BLOCK}/adapters/rediverse_ft_0000200.safetensors}
PREV="--prompt-ids 0,11,17,27,28,34 --num-samples 8"
SUB=prompts/geneval_subset100.txt
n=${#G[@]}
fan() {  # fan <module args...>: one worker per GPU, shard k of n; fails if any shard fails
    local k=0 p rc=0 pids=()
    for g in "${G[@]}"; do "$PY" -m "$@" --shard "$k" --num-shards "$n" --device "cuda:$g" & pids+=($!); k=$((k + 1)); done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$rc" = 0 ] || { echo "fan: a shard of '$*' failed" >&2; return 1; }
}
need() { [ -n "$BLOCK" ] && [ -n "$TAU" ] || { echo "set BLOCK and TAU" >&2; exit 1; }; }
for stage in "$@"; do
    case "$stage" in
        preview)
            "$PY" -m qwenimage.generate --family "$FAM" --arm teacher $PREV --tag _preview --device "cuda:${G[0]}"
            "$PY" -m qwenimage.generate --student "$S" --arm student $PREV --tag _preview --device "cuda:${G[0]}"
            ;;
        probe)
            fan qwenimage.probe --family "$FAM" --arm teacher --n 200 --ram-gb "${RAM_GB:-150}"
            fan qwenimage.probe --student "$S" --arm student --n 200 --ram-gb "${RAM_GB:-150}"
            "$PY" -m rediverse.probe_report --teacher "$OUT/$BK/probe_n200/teacher" --student "$OUT/$BK/probe_n200/$S" \
                --tsv "$OUT/$BK/probe_n200/report_$S.tsv" --plot "$OUT/$BK/probe_n200/report_$S.png"
            ;;
        bank)
            need
            [ -f prompts/train_captions.txt ] || "$PY" prompts/download_train_captions.py
            fan qwenimage.build_targets --student "$S" --tau "$TAU" --blocks "$BLOCK"
            ;;
        train)
            need
            CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${G[*]}") "$PY" -m torch.distributed.run --standalone --nproc_per_node "$n" \
                -m qwenimage.train --student "$S" --block "$BLOCK" --tau "$TAU" --run "${S}_b${BLOCK}"
            ;;
        directions) need; fan qwenimage.estimate_directions --student "$S" --tau "$TAU" --blocks "$BLOCK" --prompt-ids "$SUB" ;;
        sweep)
            need
            "$PY" -m qwenimage.sweep --student "$S" --adapter "$ADAPTER" --block "$BLOCK" --tau "$TAU" \
                --configs ft,tf2,tf4,rd2,rd4 --device "cuda:${G[0]}"
            ;;
        generate)
            need
            fan qwenimage.generate --student "$S" --arm student --prompt-ids "$SUB"
            fan qwenimage.generate --student "$S" --arm rediverse_ft --adapter "$ADAPTER" --prompt-ids "$SUB"
            fan qwenimage.generate --student "$S" --arm rediverse --adapter "$ADAPTER" --block "$BLOCK" --tau "$TAU" \
                --band "$BAND" --prompt-ids "$SUB" --tag "_a$BAND"
            fan qwenimage.generate --family "$FAM" --arm teacher --prompt-ids "$SUB"
            ;;
        *) echo "unknown stage $stage (preview | probe | bank | train | directions | sweep | generate)" >&2; exit 1 ;;
    esac
done

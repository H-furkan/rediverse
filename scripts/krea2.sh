#!/usr/bin/env bash
# Krea 2 Raw / Turbo end to end: bash scripts/krea2.sh <stage> [<stage> ...]
#
#   probe        activation probe of Raw and Turbo on the probe prompts (one worker per GPU), then the per-block report
#   directions   ReDiverse-TF directions at block 26 for GenEval subset100 (one worker per GPU)
#   bank         ReDiverse-FT target bank (Raw, block 26, forward 5) for the 100 training captions (one worker per GPU)
#   train        ReDiverse-FT training on all GPUs (torchrun, data parallel), then conversion to the release files
#   generate     teacher, student, rediverse_ft, rediverse_tf and rediverse on GenEval subset100 x 16 (one worker per GPU)
#
# GPUS lists the devices (default 0 1 2 3); PY the interpreter of an environment with diffusers >= 0.39, transformers 5.x and
# peft; BAND the ReDiverse-TF alpha range used by generate (default 4; the paper uses 2); ADAPTER the adapter used by generate
# (default: the one trained by the train stage).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
read -r -a G <<< "${GPUS:-0 1 2 3}"
OUT=${REDIVERSE_OUT:-outputs}
BAND=${BAND:-4}
ADAPTER=${ADAPTER:-$OUT/krea2/release/krea-2-rediverse.safetensors}
SUB=prompts/geneval_subset100.txt
n=${#G[@]}
fan() {  # fan <module args...>: one worker per GPU, shard k of n
    local k=0
    for g in "${G[@]}"; do "$PY" -m "$@" --shard "$k" --num-shards "$n" --device "cuda:$g" & k=$((k + 1)); done
    wait
}
for stage in "$@"; do
    case "$stage" in
        probe)
            for arm in teacher student; do fan krea2.probe --arm "$arm" --n 200; done
            "$PY" -m rediverse.probe_report --teacher "$OUT/krea2/probe_n200/teacher" --student "$OUT/krea2/probe_n200/student" \
                --collapsed block26 --tsv "$OUT/krea2/probe_n200/report.tsv" --plot "$OUT/krea2/probe_n200/report.png"
            ;;
        directions) fan krea2.estimate_directions --blocks 26 --prompt-ids "$SUB" ;;
        bank)
            [ -f prompts/train_captions.txt ] || "$PY" prompts/download_train_captions.py
            fan krea2.build_targets --blocks 26
            ;;
        train)
            CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${G[*]}") "$PY" -m torch.distributed.run --standalone --nproc_per_node "$n" \
                -m krea2.train --block 26 --run b26_mse
            "$PY" -m krea2.convert_weights --src "$OUT/krea2/train/b26_mse/adapters/rediverse_ft_0000200.safetensors" \
                --out-dir "$OUT/krea2/release"
            ;;
        generate)
            fan krea2.generate --arm student --prompt-ids "$SUB"
            fan krea2.generate --arm rediverse_tf --block 26 --band "$BAND" --prompt-ids "$SUB"
            fan krea2.generate --arm rediverse_ft --adapter "$ADAPTER" --prompt-ids "$SUB"
            fan krea2.generate --arm rediverse --block 26 --band "$BAND" --adapter "$ADAPTER" --prompt-ids "$SUB"
            fan krea2.generate --arm teacher --prompt-ids "$SUB"
            ;;
        *) echo "unknown stage $stage (probe | directions | bank | train | generate)" >&2; exit 1 ;;
    esac
done

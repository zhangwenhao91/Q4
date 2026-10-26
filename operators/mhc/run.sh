#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-smoke}"
case "$MODE" in
    smoke) ARGS=(--tokens 512 --hidden 4096 --rounds 3 --iters 20) ;;
    full) ARGS=() ;;
    *) echo "Usage: bash operators/mhc/run.sh [smoke|full]" >&2; exit 2 ;;
esac
OUT="$ROOT/results/mhc/$(date +%Y%m%d-%H%M%S)-${MODE}-$$"
mkdir -p "$OUT"
echo "Results: $OUT"
npu-smi info > "$OUT/npu-smi.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
{
    date -Is
    uname -a
    git -C "$ROOT" rev-parse HEAD
    for name in TileKernels ops-transformer; do
        if [[ -d "$ROOT/third_party/$name/.git" ]]; then
            echo "$name"
            git -C "$ROOT/third_party/$name" rev-parse HEAD
            git -C "$ROOT/third_party/$name" status --short
        fi
    done
    env | sort | grep -E '^(ASCEND_HOME_PATH|ASCEND_OPP_PATH|ASCEND_CUSTOM_OPP_PATH|ASCEND_RT_VISIBLE_DEVICES)=' || true
} > "$OUT/environment.txt"
python "$ROOT/operators/mhc/bench_mhc_post.py" "${ARGS[@]}" --output "$OUT/mhc_post.csv" 2>&1 | tee "$OUT/run.log"

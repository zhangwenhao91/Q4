#!/usr/bin/env bash
# Requires activated zwh, sourced CANN + q4mhc vendor, CC/CXX and NPU visibility.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-check}"
OUT="results/mhc/fused-projection-$(date +%Y%m%d-%H%M%S)-${MODE}-$$"
mkdir -p "$OUT"
echo "Results: $ROOT/$OUT"
python -m pip freeze > "$OUT/pip-freeze.txt"
{
    git rev-parse HEAD
    for src in TileKernels ops-transformer DeepGEMM-Ascend; do
        echo "$src"
        git -C "$ROOT/third_party/$src" rev-parse HEAD
        git -C "$ROOT/third_party/$src" status --short
    done
    "$CXX" --version
} > "$OUT/revisions.txt"
case "$MODE" in
  check)
    for seed in 42 43 44; do
      for suite in pre pre_sinkhorn; do
        python operators/mhc/extended/bench_forward.py \
          --suite "$suite" --projection deepgemm --pre-mode hf32 \
          --tokens 512 --hidden 4096 --seed "$seed" --check-only \
          --output "$OUT/$suite-seed-$seed" 2>&1 | tee "$OUT/$suite-seed-$seed.log"
      done
    done
    ;;
  bench)
    # Three independent process repetitions. Both baseline and fused paths use HF32.
    for rep in 1 2 3; do
      for suite in pre pre_sinkhorn; do
        BACKENDS=(torch deepgemm)
        if (( rep % 2 == 0 )); then BACKENDS=(deepgemm torch); fi
        for backend in "${BACKENDS[@]}"; do
          python operators/mhc/extended/bench_forward.py \
            --suite "$suite" --projection "$backend" --pre-mode hf32 \
            --tokens 512 8192 --hidden 4096 7168 --warmup 200 --iters 1000 --rounds 9 \
            --output "$OUT/$suite-$backend-rep-$rep" 2>&1 | tee "$OUT/$suite-$backend-rep-$rep.log"
        done
      done
    done
    ;;
  profile)
    for suite in pre pre_sinkhorn; do
      for impl in tilelang ascendc; do
        python operators/mhc/extended/bench_forward.py \
          --suite "$suite" --projection deepgemm --pre-mode hf32 \
          --tokens 8192 --hidden 4096 --profile "$impl" \
          --output "$OUT/$suite-$impl" 2>&1 | tee "$OUT/$suite-$impl.log"
      done
    done
    find "$OUT" -type f -name kernel_details.csv -print -exec cat {} \; > "$OUT/kernel-details.txt"
    ;;
  *) echo "Usage: bash operators/mhc/extended/run_fused.sh [check|bench|profile]" >&2; exit 2 ;;
esac

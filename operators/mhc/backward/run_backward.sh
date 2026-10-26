#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
ACTION="${1:-check-basic}"
OUT="results/mhc/backward-$(date +%Y%m%d-%H%M%S)-${ACTION}-$$"
mkdir -p "$OUT"
echo "Results: $ROOT/$OUT"
python -m pip freeze > "$OUT/pip-freeze.txt"
git rev-parse HEAD > "$OUT/revision.txt"
run_case() {
  local suite="$1" mode="$2" t="$3" d="$4" tag="$5"
  shift 5
  python operators/mhc/backward/bench_backward.py --suite "$suite" --mode "$mode" \
    --tokens "$t" --hidden "$d" --output "$OUT/$tag" "$@" 2>&1 | tee "$OUT/$tag.log"
}
case "$ACTION" in
  check-basic|check-pre)
    SUITES=(post sinkhorn)
    if [[ "$ACTION" == check-pre ]]; then SUITES=(pre pre_sinkhorn); fi
    for seed in 42 43 44; do
      for suite in "${SUITES[@]}"; do
        for mode in backward train; do
          run_case "$suite" "$mode" 512 4096 "$suite-$mode-seed$seed" --seed "$seed" --check-only
        done
      done
    done
    ;;
  bench)
    for rep in 1 2 3; do
      for suite in post sinkhorn pre pre_sinkhorn; do
        HIDDEN=(4096 7168)
        if [[ "$suite" == sinkhorn ]]; then HIDDEN=(4096); fi
        for mode in backward train; do
          for t in 512 8192; do
            for d in "${HIDDEN[@]}"; do
              run_case "$suite" "$mode" "$t" "$d" "$suite-$mode-${t}x${d}-rep$rep" \
                --warmup 100 --iters 100 --rounds 9
            done
          done
        done
      done
    done
    ;;
  profile)
    for suite in post sinkhorn pre pre_sinkhorn; do
      for mode in backward train; do
        for impl in tilelang ascendc; do
          run_case "$suite" "$mode" 8192 4096 "$suite-$mode-$impl" --profile "$impl"
        done
      done
    done
    find "$OUT" -type f -name kernel_details.csv -print -exec cat {} \; > "$OUT/kernel-details.txt"
    ;;
  *) echo 'Usage: run_backward.sh [check-basic|check-pre|bench|profile]' >&2; exit 2 ;;
esac

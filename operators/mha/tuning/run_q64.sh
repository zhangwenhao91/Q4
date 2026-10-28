#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
ACTION="${1:-check}"
OUT="results/mha-tuning/q64-$(date +%Y%m%d-%H%M%S)-$ACTION-$$"
mkdir -p "$OUT"
echo "Results: $ROOT/$OUT"
python -m pip freeze > "$OUT/pip-freeze.txt"
run_case() {
  local tag="$1"
  shift
  echo "CASE: suite=mha_q64 metadata=cached tag=$tag output=$OUT/$tag"
  python operators/spec_v1/bench.py --suite mha_q64 --ac-metadata cached \
    --output "$OUT/$tag" "$@" 2>&1 | tee "$OUT/$tag.log"
}
case "$ACTION" in
  check)
    for seed in 42 43 44; do
      run_case "seed$seed" --seed "$seed" --check-only
    done ;;
  bench)
    for rep in 1 2 3; do
      run_case "rep$rep" --warmup 100 --iters 100 --rounds 9
    done ;;
  profile)
    for impl in tilelang_q128 tilelang_q64 ascendc; do
      run_case "$impl" --profile "$impl"
    done
    find "$OUT" -type f -name kernel_details.csv -print -exec cat {} \; > "$OUT/kernel-details.txt"
    ;;
  *) echo 'Usage: bash operators/mha/tuning/run_q64.sh [check|bench|profile]' >&2; exit 2 ;;
esac

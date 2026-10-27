#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ACTION="${1:-check}"
OUT="results/mha-bf16/$(date +%Y%m%d-%H%M%S)-$ACTION-$$"
mkdir -p "$OUT"
echo "Results: $ROOT/$OUT"
python -m pip freeze > "$OUT/pip-freeze.txt"
run_case() {
  local mode="$1" tag="$2"
  shift 2
  python operators/spec_v1/bench.py --suite mha_bf16 --ac-metadata "$mode" \
    --output "$OUT/$tag" "$@" 2>&1 | tee "$OUT/$tag.log"
}
case "$ACTION" in
  check)
    for seed in 42 43 44; do
      for mode in cached per_call; do
        run_case "$mode" "$mode-seed$seed" --seed "$seed" --check-only
      done
    done ;;
  bench)
    for rep in 1 2 3; do
      MODES=(cached per_call)
      if (( rep % 2 == 0 )); then MODES=(per_call cached); fi
      for mode in "${MODES[@]}"; do
        run_case "$mode" "$mode-rep$rep" --warmup 100 --iters 100 --rounds 9
      done
    done ;;
  profile)
    for mode in cached per_call; do
      for impl in tilelang ascendc; do
        run_case "$mode" "$mode-$impl" --profile "$impl"
      done
    done
    find "$OUT" -type f -name kernel_details.csv -print -exec cat {} \; > "$OUT/kernel-details.txt"
    ;;
  *) echo 'Usage: bash operators/mha/run_bf16.sh [check|bench|profile]' >&2; exit 2 ;;
esac

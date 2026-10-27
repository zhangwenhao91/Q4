#!/usr/bin/env bash
# Activated zwh + CANN + existing q4mhc vendor; no reinstallation of mHC/DeepGEMM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODE="${1:-check-mha}"
OUT="results/spec-v1/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
echo "Results: $ROOT/$OUT"
python -m pip freeze > "$OUT/pip-freeze.txt"
run_case() {
  local suite="$1" tokens="$2" tag="$3"
  shift 3
  python operators/spec_v1/bench.py --suite "$suite" --tokens "$tokens" --output "$OUT/$tag" "$@" \
    2>&1 | tee "$OUT/$tag.log"
}
case "$MODE" in
  check-mha)
    for seed in 42 43 44; do
      run_case mha 4096 "mha-seed$seed" --seed "$seed" --check-only
    done ;;
  check-mhc)
    for seed in 42 43 44; do
      for suite in mhc_post mhc_shifted; do
        run_case "$suite" 512 "$suite-seed$seed" --seed "$seed" --check-only
      done
    done ;;
  bench)
    for rep in 1 2 3; do
      run_case mha 4096 "mha-rep$rep"
      for suite in mhc_post mhc_shifted; do
        for tokens in 512 8192; do
          run_case "$suite" "$tokens" "$suite-N$tokens-rep$rep"
        done
      done
    done ;;
  profile)
    run_case mha 4096 mha-tilelang --profile tilelang
    for tokens in 512 8192; do
      for impl in tilelang ascendc; do
        run_case mhc_post "$tokens" "post-N$tokens-$impl" --profile "$impl"
      done
      for impl in deepgemm_tilelang ascendc_pre_plus_torch_apply; do
        run_case mhc_shifted "$tokens" "shifted-N$tokens-$impl" --profile "$impl"
      done
    done
    find "$OUT" -type f -name kernel_details.csv -print -exec cat {} \; > "$OUT/kernel-details.txt" ;;
  *) echo 'Usage: run.sh [check-mha|check-mhc|bench|profile]' >&2; exit 2 ;;
esac

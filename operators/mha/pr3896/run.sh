#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
: "${ASCEND_HOME_PATH:?Source CANN first}"
source operators/mha/env_recovered.sh
python operators/spec_v1/prepare_mha.py
MODE="${1:-smoke}"
case "$MODE" in smoke|all|tails-smoke|tails-all) ;; *) echo 'Usage: run.sh [smoke|all|tails-smoke|tails-all]'; exit 2;; esac
EXTRA=()
if [[ "$MODE" == tails-* ]]; then EXTRA=(--tails); fi
OUT="results/mha-pr3896/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/runner.log") 2>&1
echo "Results: $OUT"
git rev-parse HEAD > "$OUT/revision.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
npu-smi info > "$OUT/npu-smi.txt"
run_case() {
  local tag="$1"; shift
  python operators/mha/pr3896/bench_pr.py --suite mha_bf16 --output "$OUT/$tag" "${EXTRA[@]}" "$@"
}
run_case smoke --check-only
if [[ "$MODE" == all || "$MODE" == tails-all ]]; then
  for seed in 42 43 44; do
    run_case "check-$seed" --seed "$seed" --check-only
  done
  for rep in 1 2 3; do
    for mode in cached per_call; do
      run_case "bench-$mode-rep$rep" --ac-metadata "$mode" --warmup 100 --iters 100 --rounds 10
    done
  done
fi
python operators/mha/pr3896/summarize.py --input "$OUT" | tee "$OUT/summary.txt"
echo "Completed: $OUT/summary.txt"

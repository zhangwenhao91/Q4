#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${ASCEND_HOME_PATH:?Source CANN first}"
source operators/mha/env_recovered.sh
python operators/spec_v1/prepare_mha.py
MODE="${1:-smoke}"
case "$MODE" in smoke|check|bench|profile|all) ;; *) echo 'Usage: run_plain3584.sh [smoke|check|bench|profile|all]'; exit 2 ;; esac
OUT="$ROOT/results/mha-plain3584/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
echo "Results: $OUT"
exec > >(tee "$OUT/runner.log") 2>&1
trap 'rc=$?; echo "Stopped ($rc). Logs: $OUT/runner.log"; exit "$rc"' ERR
git rev-parse HEAD > "$OUT/revision.txt"
git status --short >> "$OUT/revision.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
npu-smi info > "$OUT/npu-smi.txt"
run_case() {
  local mode="$1" tag="$2"
  shift 2
  python operators/spec_v1/mha_plain3584.py --suite mha_bf16 --mha-seqlen 3584 --ac-metadata "$mode" \
    --output "$OUT/$tag" "$@" 2>&1 | tee "$OUT/$tag.log"
}
if [[ "$MODE" == smoke ]]; then run_case cached smoke --check-only; fi
if [[ "$MODE" == check || "$MODE" == all ]]; then
  for seed in 42 43 44; do
    for metadata in cached per_call; do
      run_case "$metadata" "check-$metadata-seed$seed" --seed "$seed" --check-only
    done
  done
fi
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  for rep in 1 2 3; do
    modes=(cached per_call)
    if (( rep % 2 == 0 )); then modes=(per_call cached); fi
    for metadata in "${modes[@]}"; do
      run_case "$metadata" "bench-$metadata-rep$rep" --warmup 100 --iters 100 --rounds 10
    done
  done
fi
if [[ "$MODE" == profile || "$MODE" == all ]]; then
  for metadata in cached per_call; do
    for impl in tilelang ascendc; do
      run_case "$metadata" "profile-$metadata-$impl" --profile "$impl"
    done
  done
fi
python operators/mha/collect_plain3584.py --input "$OUT" | tee "$OUT/summary.txt"
echo 'Completed. Copy summary.txt to error.txt.'

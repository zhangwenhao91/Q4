#!/usr/bin/env bash
# Benchmark original fused H4096 and PreV2 + Sinkhorn H5120.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-smoke}"
case "$MODE" in smoke|check|bench|profile|all) ;; *) echo 'Usage: run_original.sh [smoke|check|bench|profile|all]'; exit 2 ;; esac
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
export ASCEND_NPU_ARCH=dav-3510
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$ROOT/third_party/torch-extensions-gcc13"
export ASCEND_CUSTOM_OPP_PATH="$ROOT/third_party/q4mhc-opp/vendors/q4mhc_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
test -f "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so" || { echo "Original vendor missing: run bash operators/mhc/restore_vendor.sh"; exit 1; }
OUT="$ROOT/results/mhc-original/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
echo "Results: $OUT"
exec > >(tee "$OUT/runner.log") 2>&1
trap 'rc=$?; echo "Stopped ($rc). Logs: $OUT/runner.log"; exit "$rc"' ERR
git rev-parse HEAD > "$OUT/revision.txt"
git status --short >> "$OUT/revision.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
npu-smi info > "$OUT/npu-smi.txt"

run_one() {
  local suite="$1" tag="$2" tokens="$3"
  shift 3
  python operators/spec_v1/bench.py --suite "$suite" --tokens "$tokens" \
    --output "$OUT/$tag" "$@" 2>&1 | tee "$OUT/$tag.log"
}
run_case() {
  local tag="$1" tokens="$2"
  shift 2
  if [[ "$tag" != *-rep* ]]; then tag="$tag-rep0"; fi
  run_one mhc_original "${tag/-rep/-H5120-rep}" "$tokens" "$@"
  run_one mhc_original_fused "${tag/-rep/-H4096-fused-rep}" "$tokens" "$@"
}
if [[ "$MODE" == smoke ]]; then
  run_case smoke-N128 128 --check-only
fi
if [[ "$MODE" == check || "$MODE" == all ]]; then
  for tokens in 128 512; do
    for seed in 42 43 44; do
      run_case "check-N$tokens-seed$seed" "$tokens" --seed "$seed" --check-only
    done
  done
  for tokens in 1 17 64 127 129 256 513; do
    run_case "tail-N$tokens" "$tokens" --check-only
  done
fi
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  for rep in 1 2 3; do
    for tokens in 64 128 256 512 8192; do
      run_case "bench-original-N$tokens-rep$rep" "$tokens" --warmup 100 --rounds 12 --iters 100
    done
  done
fi
if [[ "$MODE" == profile || "$MODE" == all ]]; then
  for tokens in 64 128 256 512 8192; do
    run_one mhc_original "profile-N$tokens-H5120" "$tokens" --profile ascendc_original_pre_sinkhorn
    run_one mhc_original_fused "profile-N$tokens-H4096-fused" "$tokens" --profile ascendc_original_fused_h4096
    run_one mhc_original_fused "profile-N$tokens-H4096-tilelang" "$tokens" --profile tilelang_current_pre_h4096
  done
fi
python operators/mhc/summarize_baseline.py --input "$OUT" | tee "$OUT/summary.txt"
echo "Completed. Copy $OUT/summary.txt to error.txt; no server push needed."

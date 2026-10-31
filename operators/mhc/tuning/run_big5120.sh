#!/usr/bin/env bash
# Compare upstream big-fuse H5120 with original PreV2 + Sinkhorn H5120.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-smoke}"
case "$MODE" in smoke|check|bench|profile|all) ;; *) echo 'Usage: run_big5120.sh [smoke|check|bench|profile|all]'; exit 2 ;; esac
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
OUT="$ROOT/results/mhc-big5120/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
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
  run_one mhc_original_big5120 "$tag" "$tokens" "$@"
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
  for tokens in 16 32 64 112 144 256 528; do
    run_case "tail-N$tokens" "$tokens" --check-only
  done
fi
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  for rep in 1 2 3; do
    for tokens in 64 128 256 512 8192; do
      run_case "bench-big5120-N$tokens-rep$rep" "$tokens" --warmup 100 --rounds 12 --iters 100
    done
  done
fi
if [[ "$MODE" == profile || "$MODE" == all ]]; then
  for tokens in 64 128 256 512 8192; do
    for impl in tilelang_upstream_big_fuse_h5120 tilelang_upstream_big_fuse_bound_h5120 ascendc_original_pre_sinkhorn; do
      run_case "profile-N$tokens-$impl" "$tokens" --profile "$impl"
    done
  done
fi
python operators/mhc/summarize_baseline.py --input "$OUT" | tee "$OUT/summary.txt"
echo "Completed. Copy $OUT/summary.txt to error.txt; no server push needed."

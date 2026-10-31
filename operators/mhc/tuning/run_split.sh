#!/usr/bin/env bash
# Compare bound two-/three-kernel TL paths and native AC.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-smoke}"
case "$MODE" in smoke|check|bench|profile|all) ;; *) echo 'Usage: run_split.sh [smoke|check|bench|profile|all]'; exit 2 ;; esac
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
export ASCEND_NPU_ARCH=dav-3510
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$ROOT/third_party/torch-extensions-gcc13"
export ASCEND_CUSTOM_OPP_PATH="$ROOT/third_party/q4mhcmatch-opp/vendors/q4mhcmatch_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
test -f "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so"
OUT="$ROOT/results/mhc-split/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
echo "Results: $OUT"
exec > >(tee "$OUT/runner.log") 2>&1
trap 'rc=$?; echo "Stopped ($rc). Logs: $OUT/runner.log"; exit "$rc"' ERR
git rev-parse HEAD > "$OUT/revision.txt"
git status --short >> "$OUT/revision.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
npu-smi info > "$OUT/npu-smi.txt"
cp operators/mhc/tuning/source.json "$OUT/split-source.json"
cp third_party/mhc-matched-source/source.json "$OUT/native-source.json"
run_case() {
  local tag="$1" tokens="$2"
  shift 2
  local weight_args=()
  if [[ -n "${MHC_WEIGHTS:-}" ]]; then weight_args=(--mhc-weights "$MHC_WEIGHTS"); fi
  python operators/spec_v1/bench.py --suite mhc_shifted_split --tokens "$tokens" \
    --output "$OUT/$tag" "${weight_args[@]}" "$@" 2>&1 | tee "$OUT/$tag.log"
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
  for pattern in zero prev_zero prev_basis; do
    run_case "edge-$pattern" 129 --mhc-pattern "$pattern" --check-only
  done
fi
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  for rep in 1 2 3; do
    for tokens in 64 128 256 512 8192; do
      run_case "bench-split-N$tokens-rep$rep" "$tokens" --warmup 100 --rounds 12 --iters 100
    done
  done
fi
if [[ "$MODE" == profile || "$MODE" == all ]]; then
  for tokens in 64 128 256 512 8192; do
    for impl in tilelang_apply_bound tilelang_coeff_bound ascendc_shifted_native; do
      run_case "profile-N$tokens-$impl" "$tokens" --profile "$impl"
    done
  done
fi
python operators/mhc/summarize_baseline.py --input "$OUT" | tee "$OUT/summary.txt"
echo "Completed. Copy $OUT/summary.txt to error.txt; no server push needed."

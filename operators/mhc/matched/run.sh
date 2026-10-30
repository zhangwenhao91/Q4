#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-check}"
case "$MODE" in check|bench|profile|all) ;; *) echo 'Usage: run.sh [check|bench|profile|all]'; exit 2 ;; esac
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
OUT="$ROOT/results/mhc-matched/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
echo "Results: $OUT"
exec > >(tee "$OUT/runner.log") 2>&1
trap 'rc=$?; echo "Stopped ($rc). Logs: $OUT"; exit "$rc"' ERR
git rev-parse HEAD > "$OUT/revision.txt"
git status --short >> "$OUT/revision.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
npu-smi info > "$OUT/npu-smi.txt"
cp "$ROOT/third_party/mhc-matched-source/source.json" "$OUT/native-source.json"
run_case() {
  local tag="$1" suite="$2" tokens="$3"
  shift 3
  local weight_args=()
  if [[ "$suite" == mhc_shifted_native && -n "${MHC_WEIGHTS:-}" ]]; then
    weight_args=(--mhc-weights "$MHC_WEIGHTS")
  fi
  python operators/spec_v1/bench.py --suite "$suite" --tokens "$tokens" \
    --output "$OUT/$tag" "${weight_args[@]}" "$@" 2>&1 | tee "$OUT/$tag.log"
}
if [[ "$MODE" == check || "$MODE" == all ]]; then
  # Start with a known full tile before testing tails.
  for seed in 42 43 44; do
    for suite in mhc_post mhc_shifted_native; do
      run_case "check-$suite-seed$seed" "$suite" 512 --seed "$seed" --check-only
    done
  done
  for tokens in 1 17 127 128 129 513; do
    for suite in mhc_post mhc_shifted_native; do
      run_case "tail-$suite-N$tokens" "$suite" "$tokens" --check-only
    done
  done
  for pattern in zero prev_zero prev_basis; do
    run_case "edge-$pattern" mhc_shifted_native 129 --mhc-pattern "$pattern" --check-only
  done
fi
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  for rep in 1 2 3; do
    for suite in mhc_post mhc_shifted_native; do
      for tokens in 512 8192; do
        run_case "bench-$suite-N$tokens-rep$rep" "$suite" "$tokens" --warmup 100 --rounds 9 --iters 100
      done
    done
  done
fi
if [[ "$MODE" == profile || "$MODE" == all ]]; then
  for tokens in 512 8192; do
    for impl in tilelang ascendc; do
      run_case "profile-post-N$tokens-$impl" mhc_post "$tokens" --profile "$impl"
    done
    for impl in deepgemm_tilelang ascendc_shifted_native; do
      run_case "profile-shifted-N$tokens-$impl" mhc_shifted_native "$tokens" --profile "$impl"
    done
  done
fi
python operators/mhc/summarize_baseline.py --input "$OUT" | tee "$OUT/summary.txt"
echo "Completed. Copy $OUT/summary.txt into error.txt; no server push needed."

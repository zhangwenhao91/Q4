#!/usr/bin/env bash
# Full API comparison: AC, baseline TL, all-three-kernel N pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_NPU_ARCH=dav-3510
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$ROOT/third_party/torch-extensions-gcc13"
export ASCEND_CUSTOM_OPP_PATH="$ROOT/third_party/q4mhcmatch-opp/vendors/q4mhcmatch_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
test -f "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so"
PY="$CONDA_PREFIX/bin/python"
OUT="$ROOT/results/mhc-all-token-bench/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/revision.txt"
touch "$OUT/runner.log"
summarize() {
  "$PY" "$ROOT/operators/mhc/tuning/summarize_token_pipeline.py" \
    --input "$OUT" --candidate tilelang_all_token_h5120s3
}
echo "Results: $OUT"
for n in 128 256 512 2048 8192; do
  for rep in 1 2 3; do
    echo "Running N=$n repeat=$rep (full output saved to log)"
    if "$PY" operators/spec_v1/bench.py --suite mhc_shifted_split \
      --mhc-all-token-pipeline \
      --implementations ascendc_shifted_native tilelang_coeff_bound tilelang_all_token_h5120s3 \
      --tokens "$n" --warmup 100 --rounds 12 --iters 100 \
      --output "$OUT/N$n-rep$rep" > "$OUT/N$n-rep$rep.log" 2>&1; then
      echo "Completed N=$n repeat=$rep" >> "$OUT/runner.log"
    else
      echo "Failed N=$n repeat=$rep" >> "$OUT/runner.log"
      tail -n 40 "$OUT/N$n-rep$rep.log" >> "$OUT/runner.log"
      summarize
      exit 1
    fi
  done
done
echo 'All 15 cases completed.' >> "$OUT/runner.log"
summarize

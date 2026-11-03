#!/usr/bin/env bash
# One full three-kernel call, all N-pipeline candidates, no warmup/check launches.
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
OUT="$ROOT/results/mhc-all-token-simulator/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUT"
REPORT="$OUT/summary.txt"
echo "Results: $OUT" | tee "$REPORT"
git rev-parse HEAD > "$OUT/revision.txt"
echo 'Collecting three kernels. Full console output is saved to console.log.'
if msprof op simulator --soc-version="${MHC_SIM_SOC:-Ascend950PR_9579}" \
  --kernel-name='mhc_pre_apply_flat_pipeline_asc|mhc_projection_token_pipeline|mhc_coeff_token_pipeline' \
  --launch-count=3 --aic-metrics=PipeUtilization --dump=true \
  --timeout=60 --output="$OUT/prof" \
  "$CONDA_PREFIX/bin/python" "$ROOT/operators/mhc/tuning/simulate_all_token.py" \
    --tokens "${MHC_SIM_TOKENS:-2048}" --output "$OUT/source" \
    > "$OUT/console.log" 2>&1; then
  grep -E 'Start profiling on kernel:|ALL TOKEN CHECK:|Profiling results saved|Profiling data parse finished|\[ERROR\]' \
    "$OUT/console.log" | tee -a "$REPORT" || true
else
  tail -n 60 "$OUT/console.log" | tee -a "$REPORT"
  cp "$REPORT" "$ROOT/error.txt"
  exit 1
fi
find "$OUT" -name visualize_data.bin -print | tee -a "$REPORT"
echo 'Inspect same-core N-iteration overlap in all three traces; actual overlap is not yet verified.' | tee -a "$REPORT"
cp "$REPORT" "$ROOT/error.txt"
echo "Compact report: $ROOT/error.txt"

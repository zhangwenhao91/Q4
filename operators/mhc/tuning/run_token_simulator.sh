#!/usr/bin/env bash
# Isolated original/full-hidden token pipeline launches, identical CPU inputs.
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
SOC="${MHC_SIM_SOC:-Ascend950PR_9579}"
TOKENS="${MHC_SIM_TOKENS:-512}"
CORE="${MHC_SIM_CORE:-0}"
OUT="$ROOT/results/mhc-token-simulator/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUT"
REPORT="$OUT/summary.txt"
echo "Results: $OUT" | tee "$REPORT"
git rev-parse HEAD > "$OUT/revision.txt"
echo "N=$TOKENS H=5120; parsed core=$CORE; original versus token stages=3" | tee -a "$REPORT"
for impl in original token; do
  kernel=mhc_pre_apply_mix_fwd_asc_kernel
  if [[ "$impl" == token ]]; then kernel=mhc_pre_apply_flat_pipeline_asc_kernel; fi
  mkdir -p "$OUT/$impl"
  echo "Collecting $impl (full log: $OUT/$impl/console.log)" | tee -a "$REPORT"
  if msprof op simulator --soc-version="$SOC" --kernel-name="$kernel" \
    --launch-count=1 --core-id="$CORE" --aic-metrics=PipeUtilization \
    --dump=true --timeout=30 --output="$OUT/$impl/prof" \
    "$CONDA_PREFIX/bin/python" "$ROOT/operators/mhc/tuning/simulate_apply.py" \
      --impl "$impl" --tokens "$TOKENS" --output "$OUT/$impl/source" \
      > "$OUT/$impl/console.log" 2>&1; then
    grep -E 'Start profiling on kernel:|APPLY CHECK:|Profiling data parse finished|Profiling results saved' \
      "$OUT/$impl/console.log" | tee -a "$REPORT" || true
  else
    tail -n 40 "$OUT/$impl/console.log" | tee -a "$REPORT"
    cp "$REPORT" "$ROOT/error.txt"
    exit 1
  fi
done
find "$OUT" -name visualize_data.bin -print | tee -a "$REPORT"
echo 'Collection finished. Inspect same-core MTE2/Vector/MTE3 intervals in Insight; overlap is not yet verified.' | tee -a "$REPORT"
cp "$REPORT" "$ROOT/error.txt"
echo "Compact report saved to $ROOT/error.txt. Keep full result directories for trace analysis."

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-original}"
case "$MODE" in original|nested|flat|all) ;; *) echo 'Usage: run_simulator.sh [original|nested|flat|all]'; exit 2 ;; esac
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
export ASCEND_NPU_ARCH=dav-3510
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$ROOT/third_party/torch-extensions-gcc13"
SOC="${MHC_SIM_SOC:-Ascend950PR_9579}"
TOKENS="${MHC_SIM_TOKENS:-128}"
CORE="${MHC_SIM_CORE:-0}"
OUT="$ROOT/results/mhc-simulator/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
echo "Results: $OUT"
exec > >(tee "$OUT/runner.log") 2>&1
trap 'rc=$?; echo "Stopped ($rc). Logs: $OUT/runner.log"; exit "$rc"' ERR
git rev-parse HEAD > "$OUT/revision.txt"
msprof op simulator --help > "$OUT/simulator-help.txt"
impls=("$MODE")
if [[ "$MODE" == all ]]; then impls=(original nested flat); fi
for impl in "${impls[@]}"; do
  case "$impl" in
    original) kernel=mhc_pre_apply_mix_fwd_asc_kernel ;;
    nested) kernel=mhc_pre_apply_pipeline_asc_kernel ;;
    flat) kernel=mhc_pre_apply_flat_pipeline_asc_kernel ;;
  esac
  mkdir -p "$OUT/$impl"
  msprof op simulator --soc-version="$SOC" --kernel-name="$kernel" \
    --launch-count=1 --core-id="$CORE" --aic-metrics=PipeUtilization \
    --dump=true --timeout=30 --output="$OUT/$impl" \
    "$CONDA_PREFIX/bin/python" operators/mhc/tuning/simulate_apply.py \
      --impl "$impl" --tokens "$TOKENS" --output "$OUT/$impl/source" \
      2>&1 | tee "$OUT/$impl/console.log"
done
find "$OUT" -type f \( -name 'visualize_data.bin' -o -name '*trace*.json' -o -name '*timeline*.json' -o -name 'correctness.json' \) -print
echo "Completed. Copy runner.log into error.txt; keep all simulator output for Insight analysis."

#!/usr/bin/env bash
# One full three-kernel call, all N-pipeline candidates, no warmup/check launches.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
VARIANT="${1:-candidate}"
COMPARE="${2:-}"
case "${MHC_SIM_STAGE:-all}" in
  all) stages=(apply projection tail) ;;
  projection) stages=(projection) ;;
  *) echo 'MHC_SIM_STAGE must be all or projection' >&2; exit 2 ;;
esac
case "$VARIANT" in
  candidate|baseline|k8-prealloc|fused) ;;
  *) echo 'Usage: run_all_token_simulator.sh [candidate|k8-prealloc|fused|baseline [candidate_run]]' >&2; exit 2 ;;
esac
if [[ "$VARIANT" == fused && "${MHC_SIM_STAGE:-all}" == all ]]; then stages=(projection tail); fi
if [[ -n "$COMPARE" && ( "$VARIANT" != baseline || ! -d "$COMPARE" ) ]]; then
  echo 'Pass an existing candidate run directory with baseline mode.' >&2
  exit 2
fi
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_NPU_ARCH=dav-3510
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$ROOT/third_party/torch-extensions-gcc13"
OUT="$ROOT/results/mhc-all-token-simulator/$(date +%Y%m%d-%H%M%S)-$$"
if [[ "$VARIANT" == baseline ]]; then OUT="$OUT-baseline"; fi
if [[ "$VARIANT" == k8-prealloc || "$VARIANT" == fused ]]; then OUT="$OUT-$VARIANT-N${MHC_SIM_TOKENS:-128}"; fi
mkdir -p "$OUT"
REPORT="$OUT/summary.txt"
echo "Results: $OUT" | tee "$REPORT"
git rev-parse HEAD > "$OUT/revision.txt"
PY="$CONDA_PREFIX/bin/python"
default_tokens=2048
if [[ "$VARIANT" == k8-prealloc || "$VARIANT" == fused ]]; then default_tokens=128; fi
TOKENS="${MHC_SIM_TOKENS:-$default_tokens}"
touch "$OUT/console.log"
diagnose() {
  "$PY" "$ROOT/operators/mhc/tuning/diagnose_all_token.py" --input "$OUT"
}
variants=(baseline candidate)
if [[ "$VARIANT" == baseline ]]; then variants=(baseline); fi
if [[ "$VARIANT" == k8-prealloc || "$VARIANT" == fused ]]; then variants=("$VARIANT"); fi
echo 'Checking precision on NPU without msprof first.'
for variant in "${variants[@]}"; do
  echo "Preflight: $variant" | tee -a "$OUT/console.log"
  if timeout 600 "$PY" "$ROOT/operators/mhc/tuning/simulate_all_token.py" \
    --variant "$variant" --tokens "$TOKENS" --output "$OUT/preflight-$variant" \
    > "$OUT/preflight-$variant.log" 2>&1; then
    tail -n 3 "$OUT/preflight-$variant.log" >> "$OUT/console.log"
  else
    tail -n 100 "$OUT/preflight-$variant.log" >> "$OUT/console.log"
    diagnose
    exit 1
  fi
done
echo 'Preflight passed. Collecting one target kernel per process; parsing core 0 by default.'
for stage in "${stages[@]}"; do
  case "$stage" in
    apply) kernel=mhc_pre_apply_flat_pipeline_asc_kernel ;;
    projection) kernel=mhc_projection_token_pipeline_kernel ;;
    tail) kernel=mhc_coeff_token_pipeline_kernel ;;
  esac
  if [[ "$VARIANT" == baseline ]]; then
    case "$stage" in
      apply) kernel=mhc_pre_apply_mix_fwd_asc_kernel ;;
      projection) kernel=tf32_hc_prenorm_gemm_kernel_kernel ;;
      tail) kernel=mhc_coeff_tail_fwd_asc_kernel ;;
    esac
  fi
  if [[ "$VARIANT" == k8-prealloc && "$stage" == projection ]]; then
    kernel=mhc_projection_split_partials_kernel
  fi
  if [[ "$VARIANT" == fused && "$stage" == tail ]]; then
    kernel=mhc_coeff_apply_small_pipeline_kernel
  fi
  mkdir -p "$OUT/$stage"
  echo "Capture: $stage" | tee -a "$OUT/console.log"
  if "$PY" "$ROOT/operators/mhc/tuning/collect_guarded.py" --log "$OUT/$stage/console.log" -- \
    msprof op simulator --soc-version="${MHC_SIM_SOC:-Ascend950PR_9579}" \
    --kernel-name="$kernel" --core-id="${MHC_SIM_CORE:-0}" \
    --launch-count=1 --aic-metrics=PipeUtilization --dump=true \
    --timeout="${MHC_SIM_TIMEOUT:-30}" --output="$OUT/$stage/prof" \
    "$PY" "$ROOT/operators/mhc/tuning/simulate_all_token.py" \
      --variant "$VARIANT" --tokens "$TOKENS" --output "$OUT/$stage/source"; then
    cat "$OUT/$stage/console.log" >> "$OUT/console.log"
  else
    cat "$OUT/$stage/console.log" >> "$OUT/console.log"
    diagnose
    exit 1
  fi
  if grep -q '\[ERROR\]' "$OUT/$stage/console.log" || \
     ! grep -q 'Profiling data parse finished' "$OUT/$stage/console.log" || \
     ! test -f "$OUT/$stage/source/correctness.json" || \
     [[ "$(find "$OUT/$stage/prof" -name visualize_data.bin | wc -l)" -lt 1 ]]; then
    echo "Capture incomplete: $stage" >> "$OUT/console.log"
    diagnose
    exit 1
  fi
  "$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if len(c)==4 and all(v["passed"] for v in c) else 1)' "$OUT/$stage/source/correctness.json"
done
echo 'Selected captures and CPU checks passed.' >> "$OUT/console.log"
diagnose
if [[ "$VARIANT" == k8-prealloc || "$VARIANT" == fused ]]; then
  find "$OUT" -name visualize_data.bin -print | tee "$OUT/insight-files.txt"
  if [[ "${MHC_SIM_STAGE:-all}" == all ]]; then
    "$PY" "$ROOT/operators/mhc/tuning/summarize_all_token_trace.py" --input "$OUT" \
      --stages "${stages[@]}" --output "$OUT/overlap.txt"
    cat "$OUT/insight-files.txt" "$OUT/overlap.txt" > "$ROOT/error.txt"
  fi
fi
if [[ -n "$COMPARE" ]]; then
  "$PY" "$ROOT/operators/mhc/tuning/summarize_all_token_trace.py" --input "$COMPARE" --baseline "$OUT"
fi

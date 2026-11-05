#!/usr/bin/env bash
# Build pinned unmodified v3 HcPre; validate and compare full API on NPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
case "$MODE" in build|smoke|bench|all) ;; *) echo 'Usage: run.sh [build|smoke|bench|all]'; exit 2 ;; esac
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_NPU_ARCH=dav-3510
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS="${Q4_BUILD_JOBS:-8}"
export TORCH_EXTENSIONS_DIR="$ROOT/third_party/torch-extensions-gcc13"
export ASCEND_CUSTOM_OPP_PATH="$ROOT/third_party/q4vllmv3-opp/vendors/q4vllmv3_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
PY="$CONDA_PREFIX/bin/python"
OUT="$ROOT/results/mhc-vllm-v3/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/revision.txt"
touch "$OUT/runner.log"
echo "Results: $OUT"
failure() {
  "$PY" operators/mhc/vllm_v3/report_failure.py --stage "$1" --log "$2"
}
if [[ "$MODE" == build || "$MODE" == all ]]; then
  echo 'Building isolated pinned upstream HcPre (details in build.log).'
  if ! bash operators/mhc/vllm_v3/build.sh > "$OUT/build.log" 2>&1; then
    failure build "$OUT/build.log"
    exit 1
  fi
  echo 'Build completed.' | tee -a "$OUT/runner.log"
fi
if [[ "$MODE" == build ]]; then
  cp "$OUT/runner.log" "$ROOT/error.txt"
  exit 0
fi
test -f "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so" || {
  echo 'Missing isolated vendor. Run: bash operators/mhc/vllm_v3/run.sh all' | tee "$ROOT/error.txt"; exit 1;
}
summarize() {
  "$PY" operators/mhc/tuning/summarize_token_pipeline.py --input "$OUT" \
    --reference ascendc_vllm_v3 --candidate tilelang_all_token_h5120s3
}
# Every benchmark case also runs CPU checks and pairwise comparisons before timing.
if [[ "$MODE" == smoke || "$MODE" == all ]]; then
  for n in 128 2048; do
    echo "Checking N=$n"
    if ! "$PY" operators/spec_v1/bench.py --suite mhc_vllm_v3 --tokens "$n" \
      --check-only --output "$OUT/check-N$n" > "$OUT/check-N$n.log" 2>&1; then
      failure "precision N=$n" "$OUT/check-N$n.log"
      exit 1
    fi
    echo "Precision passed N=$n" | tee -a "$OUT/runner.log"
  done
fi
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  for n in 128 256 512 2048 8192; do
    for rep in 1 2 3; do
      echo "Timing N=$n repeat=$rep"
      if ! "$PY" operators/spec_v1/bench.py --suite mhc_vllm_v3 --tokens "$n" \
        --warmup 100 --rounds 12 --iters 100 --output "$OUT/N$n-rep$rep" \
        > "$OUT/N$n-rep$rep.log" 2>&1; then
        failure "benchmark N=$n repeat=$rep" "$OUT/N$n-rep$rep.log"
        exit 1
      fi
      echo "Completed N=$n repeat=$rep" >> "$OUT/runner.log"
    done
  done
  echo 'All 15 cases completed.' >> "$OUT/runner.log"
  summarize
else
  { echo "Run: $OUT"; cat "$OUT/runner.log"; } > "$ROOT/error.txt"
  cat "$ROOT/error.txt"
fi

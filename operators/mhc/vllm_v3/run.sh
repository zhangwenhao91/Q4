#!/usr/bin/env bash
# Build pinned unmodified v3 HcPre; validate and compare full API on NPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
case "$MODE" in build|smoke|bench|all|resume|projection) ;; *) echo 'Usage: run.sh [build|smoke|bench|all|resume|projection]'; exit 2 ;; esac
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
if [[ "$MODE" == resume ]]; then
  echo 'Installing previously built upstream HcPre (details in install.log).'
  if ! bash operators/mhc/vllm_v3/build.sh install-existing > "$OUT/install.log" 2>&1; then
    failure install "$OUT/install.log"
    exit 1
  fi
  echo 'Install completed.' | tee -a "$OUT/runner.log"
fi
if [[ "$MODE" == build ]]; then
  cp "$OUT/runner.log" "$ROOT/error.txt"
  exit 0
fi
test -f "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so" || {
  echo 'Missing isolated vendor. Run: bash operators/mhc/vllm_v3/run.sh all' | tee "$ROOT/error.txt"; exit 1;
}
if [[ "$MODE" == projection ]]; then
  exec bash operators/mhc/vllm_v3/profile_projection.sh "$OUT"
fi
stages=(pre post combined)
# Every benchmark case also runs CPU checks and pairwise comparisons before timing.
if [[ "$MODE" == smoke || "$MODE" == all || "$MODE" == resume ]]; then
  for stage in "${stages[@]}"; do
    for n in 128 2048; do
      echo "Checking $stage N=$n"
      log="$OUT/check-$stage-N$n.log"
      if ! "$PY" operators/spec_v1/bench.py --suite mhc_vllm_v3 --mhc-vllm-stage "$stage" --tokens "$n" \
        --check-only --output "$OUT/check-$stage-N$n" > "$log" 2>&1; then
        failure "precision $stage N=$n" "$log"
        exit 1
      fi
      echo "Precision passed $stage N=$n" | tee -a "$OUT/runner.log"
    done
  done
fi
if [[ "$MODE" == bench || "$MODE" == all || "$MODE" == resume ]]; then
  for stage in "${stages[@]}"; do
    mkdir -p "$OUT/$stage"
    for n in 128 256 512 2048 8192; do
      for rep in 1 2 3; do
        echo "Timing $stage N=$n repeat=$rep"
        case_dir="$OUT/$stage/N$n-rep$rep"
        if ! "$PY" operators/spec_v1/bench.py --suite mhc_vllm_v3 --mhc-vllm-stage "$stage" --tokens "$n" \
          --warmup 100 --rounds 12 --iters 100 --output "$case_dir" > "$case_dir.log" 2>&1; then
          failure "benchmark $stage N=$n repeat=$rep" "$case_dir.log"
          exit 1
        fi
        echo "Completed $stage N=$n repeat=$rep" >> "$OUT/runner.log"
      done
    done
    labels=(--reference ascendc_vllm_v3 --candidate tilelang_all_token_h5120s3)
    if [[ "$stage" == post ]]; then
      labels=(--reference ascendc_vllm_post --baseline tilelang_post --candidate '')
    fi
    "$PY" operators/mhc/tuning/summarize_token_pipeline.py --input "$OUT/$stage" \
      --output "$OUT/$stage/summary.txt" "${labels[@]}" > "$OUT/$stage/summary.log"
  done
  {
    for stage in "${stages[@]}"; do
      printf '\n=== %s ===\n' "$stage"
      cat "$OUT/$stage/summary.txt"
    done
  } > "$ROOT/error.txt"
  cat "$ROOT/error.txt"
else
  { echo "Run: $OUT"; cat "$OUT/runner.log"; } > "$ROOT/error.txt"
  cat "$ROOT/error.txt"
fi

#!/usr/bin/env bash
# Full H5120 apply: compare AC, original TL and token-pipelined TL.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-bench}"
case "$MODE" in smoke|bench) ;; *) echo 'Usage: run_token_pipeline.sh [smoke|bench]'; exit 2 ;; esac
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
OUT="$ROOT/results/mhc-token-pipeline/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/runner.log") 2>&1
trap 'rc=$?; echo "Stopped ($rc). Logs: $OUT/runner.log"; exit "$rc"' ERR
echo "Results: $OUT"
git rev-parse HEAD > "$OUT/revision.txt"
git status --short >> "$OUT/revision.txt"
tokens=(128 256 512 8192)
reps=(1 2 3)
extra=(--warmup 100 --rounds 12 --iters 100)
if [[ "$MODE" == smoke ]]; then
  tokens=(128 129)
  reps=(1)
  extra=(--check-only)
fi
for n in "${tokens[@]}"; do
  for rep in "${reps[@]}"; do
    "$CONDA_PREFIX/bin/python" operators/spec_v1/bench.py \
      --suite mhc_shifted_split \
      --mhc-apply-pipeline 5120:3 --mhc-pipeline-flat \
      --implementations ascendc_shifted_native tilelang_coeff_bound tilelang_flat_h5120s3 \
      --tokens "$n" "${extra[@]}" --output "$OUT/N$n-rep$rep" \
      2>&1 | tee "$OUT/N$n-rep$rep.log"
  done
done
echo "Completed. Results and console logs: $OUT"

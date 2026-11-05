#!/usr/bin/env bash
# Benchmark original fused H4096 and PreV2 + Sinkhorn H5120.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-smoke}"
case "$MODE" in smoke|check|bench|profile|all|h4096-ab|h5120-fused-ab) ;; *) echo 'Usage: run_original.sh [smoke|check|bench|profile|all|h4096-ab|h5120-fused-ab]'; exit 2 ;; esac
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
export ASCEND_NPU_ARCH=dav-3510
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$ROOT/third_party/torch-extensions-gcc13"
vendor=q4mhc
if [[ "$MODE" == h5120-fused-ab ]]; then vendor=q4mhc5120; fi
export ASCEND_CUSTOM_OPP_PATH="$ROOT/third_party/$vendor-opp/vendors/${vendor}_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
OUT="$ROOT/results/mhc-original/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
echo "Results: $OUT"
if [[ ! -f "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so" ]]; then
  if [[ "$MODE" != h5120-fused-ab ]]; then
    echo "Original vendor missing: run bash operators/mhc/restore_vendor.sh"
    exit 1
  fi
  echo "Building isolated H=5120 fused vendor (details in $OUT/build.log)"
  if ! bash operators/mhc/restore_vendor.sh h5120-fused > "$OUT/build.log" 2>&1; then
    python operators/mhc/vllm_v3/report_failure.py --stage build --log "$OUT/build.log"
    exit 1
  fi
fi
if [[ "$MODE" == h4096-ab || "$MODE" == h5120-fused-ab ]]; then
  hidden=4096
  reference=ascendc_original_fused_h4096
  if [[ "$MODE" == h5120-fused-ab ]]; then
    hidden=5120
    reference=ascendc_adapted_fused_h5120
  fi
  candidate="tilelang_current_pre_all_token_l2_default_h$hidden"
  printf 'Q4 revision: %s\n' "$(git rev-parse HEAD)" > "$OUT/runner.log"
  for tokens in 128 256 512 2048 8192; do
    for rep in 1 2 3; do
      tag="N$tokens-rep$rep"
      labels=("$reference" "$candidate")
      if [[ "$rep" == 2 ]]; then labels=("$candidate" "$reference"); fi
      echo "Timing H=$hidden $tag"
      if ! python operators/spec_v1/bench.py --suite mhc_original_fused --tokens "$tokens" \
        --implementations "${labels[@]}" --warmup 100 --rounds 12 --iters 100 \
        --output "$OUT/$tag" > "$OUT/$tag.log" 2>&1; then
        python operators/mhc/vllm_v3/report_failure.py --stage "$tag" --log "$OUT/$tag.log"
        exit 1
      fi
      echo "Raw TL projection, CPU and pairwise correctness passed for $tag" >> "$OUT/runner.log"
    done
  done
  python operators/mhc/tuning/summarize_token_pipeline.py --input "$OUT" --output "$ROOT/error.txt" \
    --reference "$reference" --baseline "$candidate" --candidate ''
  exit 0
fi
exec > >(tee "$OUT/runner.log") 2>&1
trap 'rc=$?; echo "Stopped ($rc). Logs: $OUT/runner.log"; exit "$rc"' ERR
git rev-parse HEAD > "$OUT/revision.txt"
git status --short >> "$OUT/revision.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
npu-smi info > "$OUT/npu-smi.txt"

run_one() {
  local suite="$1" tag="$2" tokens="$3"
  shift 3
  python operators/spec_v1/bench.py --suite "$suite" --tokens "$tokens" \
    --output "$OUT/$tag" "$@" 2>&1 | tee "$OUT/$tag.log"
}
run_case() {
  local tag="$1" tokens="$2"
  shift 2
  if [[ "$tag" != *-rep* ]]; then tag="$tag-rep0"; fi
  run_one mhc_original "${tag/-rep/-H5120-rep}" "$tokens" "$@"
  run_one mhc_original_fused "${tag/-rep/-H4096-fused-rep}" "$tokens" "$@"
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
fi
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  for rep in 1 2 3; do
    for tokens in 64 128 256 512 8192; do
      run_case "bench-original-N$tokens-rep$rep" "$tokens" --warmup 100 --rounds 16 --iters 100
    done
  done
fi
if [[ "$MODE" == profile || "$MODE" == all ]]; then
  for tokens in 64 128 256 512 8192; do
    run_one mhc_original "profile-N$tokens-H5120" "$tokens" --profile ascendc_original_pre_sinkhorn
    run_one mhc_original_fused "profile-N$tokens-H4096-fused" "$tokens" --profile ascendc_original_fused_h4096
    run_one mhc_original_fused "profile-N$tokens-H4096-tilelang" "$tokens" --profile tilelang_current_pre_h4096
    run_one mhc_original_fused "profile-N$tokens-H4096-upstream" "$tokens" --profile tilelang_upstream_big_fuse_h4096
    run_one mhc_original_fused "profile-N$tokens-H4096-upstream-bound" "$tokens" --profile tilelang_upstream_big_fuse_bound_h4096
  done
fi
python operators/mhc/summarize_baseline.py --input "$OUT" | tee "$OUT/summary.txt"
echo "Completed. Copy $OUT/summary.txt to error.txt; no server push needed."

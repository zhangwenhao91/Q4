#!/usr/bin/env bash
# Compare bound two-/three-kernel TL paths and native AC.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODE="${1:-smoke}"
case "$MODE" in smoke|check|bench|profile|all) ;; *) echo 'Usage: run_tail_scan.sh [smoke|check|bench|profile|all]'; exit 2 ;; esac
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
OUT="$ROOT/results/mhc-tail-scan/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
echo "Results: $OUT"
exec > >(tee "$OUT/runner.log") 2>&1
trap 'rc=$?; echo "Stopped ($rc). Logs: $OUT/runner.log"; exit "$rc"' ERR
git rev-parse HEAD > "$OUT/revision.txt"
git status --short >> "$OUT/revision.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
npu-smi info > "$OUT/npu-smi.txt"
cp operators/mhc/tuning/source.json "$OUT/split-source.json"
cp third_party/mhc-matched-source/source.json "$OUT/native-source.json"
printf '%s\n' '{"apply_hidden_tile":1280,"token_block":16}' > "$OUT/tail-scan.json"
run_case() {
  local tag="$1" tokens="$2" splits="$3"
  shift 3
  python operators/spec_v1/bench.py --suite mhc_shifted_split --tokens "$tokens" \
    --mhc-tail-config "$splits" --output "$OUT/$tag" "$@" 2>&1 | tee "$OUT/$tag.log"
}
if [[ "$MODE" == smoke ]]; then
  for splits in 14:1 14:2 28:1 28:2 56:1 56:2; do run_case "smoke-K$splits" 128 "$splits" --check-only; done
fi
if [[ "$MODE" == check || "$MODE" == all ]]; then
  for splits in 14:1 14:2 28:1 28:2 56:1 56:2; do
    for tokens in 64 128 129 256 512; do
      for seed in 42 43 44; do
        run_case "check-N$tokens-K$splits-seed$seed" "$tokens" "$splits" --seed "$seed" --check-only
      done
    done
  done
fi
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  for rep in 1 2 3; do
    for tokens in 64 128 256 512; do
      for splits in 14:1 14:2 28:1 28:2 56:1 56:2; do
        run_case "bench-N$tokens-K$splits-rep$rep" "$tokens" "$splits" --warmup 100 --rounds 16 --iters 100
      done
    done
  done
fi
if [[ "$MODE" == profile || "$MODE" == all ]]; then
  for tokens in 64 128 256 512; do
    for splits in 14:1 14:2 28:1 28:2 56:1 56:2; do
      run_case "profile-N$tokens-K$splits" "$tokens" "$splits" --profile "tilelang_tail_c${splits/:/s}"
    done
  done
fi
python operators/mhc/summarize_baseline.py --input "$OUT" | tee "$OUT/summary.txt"
echo "Completed. Copy summary.txt to error.txt."

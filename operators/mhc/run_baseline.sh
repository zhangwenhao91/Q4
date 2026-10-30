#!/usr/bin/env bash
# Run from an activated NPU environment; does not install or rebuild dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
case "$MODE" in
  check|bench|profile|all|preflight) ;;
  *) echo 'Usage: bash operators/mhc/run_baseline.sh [preflight|check|bench|profile|all]' >&2; exit 2 ;;
esac
OUT="$ROOT/results/mhc-baseline/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
echo "Results: $OUT"
trap 'rc=$?; echo "Stopped (exit $rc). Logs: $OUT" >&2; exit "$rc"' ERR
{
  git rev-parse HEAD
  git status --short
  printf 'Device selection: %s\nVendor: %s\n' "${ASCEND_RT_VISIBLE_DEVICES:-unset}" "${ASCEND_CUSTOM_OPP_PATH:-unset}"
  command -v python
  npu-smi info
  python -m pip freeze
} > "$OUT/environment.txt" 2>&1
python - <<'PY' 2>&1 | tee "$OUT/preflight.log"
import importlib
import os
from pathlib import Path
import torch
import torch_npu
import tilelang
import deep_gemm
from tile_kernels.mhc import (mhc_post_fwd, mhc_pre_apply_mix_fwd,
    mhc_reduce_partials_and_rmsnorm_fwd, mhc_pre_split_mixes_fwd, mhc_sinkhorn_fwd)
importlib.import_module('cann_ops_transformer.ops.mhc.mhc_post.mhc_post')
vendor = Path(os.environ.get('ASCEND_CUSTOM_OPP_PATH', '/__q4_missing_vendor__'))
assert (vendor / 'op_api/lib/libcust_opapi.so').is_file(), f'Missing mHC vendor library: {vendor}'
assert torch.npu.is_available(), 'NPU unavailable'
torch.npu.set_device(0)
assert callable(deep_gemm.tf32_hc_prenorm_gemm)
print('Device:', torch.npu.get_device_name(0))
print('TileLang:', tilelang.__version__)
print('HC kernel:', Path(deep_gemm.__file__).parent / 'tilelang_ops/hc_prenorm.py')
print('DeepGEMM cores:', deep_gemm.get_num_sms())
print('Preflight passed; device compilation and numerical checks follow in check mode.')
PY
if [[ "$MODE" == preflight ]]; then exit 0; fi
run_case() {
  local tag="$1" suite="$2" tokens="$3"
  shift 3
  python operators/spec_v1/bench.py --suite "$suite" --tokens "$tokens" \
    --output "$OUT/$tag" "$@" 2>&1 | tee "$OUT/$tag.log"
}
if [[ "$MODE" == check || "$MODE" == all ]]; then
  for suite in mhc_post mhc_shifted; do
    for seed in 42 43 44; do
      run_case "check-$suite-seed$seed" "$suite" 512 --seed "$seed" --check-only
    done
  done
fi
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  for rep in 1 2 3; do
    for suite in mhc_post mhc_shifted; do
      for tokens in 512 8192; do
        run_case "bench-$suite-N$tokens-rep$rep" "$suite" "$tokens" \
          --warmup 100 --rounds 9 --iters 100
      done
    done
  done
fi
if [[ "$MODE" == profile || "$MODE" == all ]]; then
  for tokens in 512 8192; do
    for impl in tilelang ascendc; do
      run_case "profile-post-N$tokens-$impl" mhc_post "$tokens" --profile "$impl"
    done
    for impl in deepgemm_tilelang ascendc_pre_plus_torch_apply; do
      run_case "profile-shifted-N$tokens-$impl" mhc_shifted "$tokens" --profile "$impl"
    done
  done
  find "$OUT" -type f -name kernel_details.csv -print > "$OUT/kernel-files.txt"
fi
tar -czf "${OUT}.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "Completed. Archive: ${OUT}.tar.gz"

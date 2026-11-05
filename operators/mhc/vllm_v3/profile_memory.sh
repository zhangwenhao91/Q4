#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
PY="$CONDA_PREFIX/bin/python"
for metric in Memory L2Cache; do
  for impl in tilelang_coeff_bound tilelang_all_token_h5120s3; do
    name="$metric-$impl"
    echo "N=8192 $name"
    if ! "$PY" operators/spec_v1/bench.py --suite mhc_vllm_v3 --mhc-vllm-stage pre \
      --tokens 8192 --output "$OUT/$name" --implementations "$impl" \
      --profile "$impl" --profile-aic-metric "$metric" > "$OUT/$name.log" 2>&1; then
      "$PY" operators/mhc/vllm_v3/report_failure.py --stage "$name" --log "$OUT/$name.log"
      exit 1
    fi
  done
done
"$PY" operators/mhc/vllm_v3/summarize_memory.py --input "$OUT"

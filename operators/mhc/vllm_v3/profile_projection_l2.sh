#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
PY="$CONDA_PREFIX/bin/python"
BASE=tilelang_all_token_h5120s3
DEFAULT=tilelang_all_token_l2_default
common=(--suite mhc_vllm_v3 --mhc-vllm-stage pre --mhc-projection-l2-ab --tokens 8192)
bench() {
  local name="$1"
  shift
  echo "N=8192 $name"
  if ! "$PY" operators/spec_v1/bench.py "${common[@]}" --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1; then
    "$PY" operators/mhc/vllm_v3/report_failure.py --stage "$name" --log "$OUT/$name.log"
    exit 1
  fi
}
for rep in 1 2 3; do
  labels=("$BASE" "$DEFAULT")
  if [[ "$rep" == 2 ]]; then labels=("$DEFAULT" "$BASE"); fi
  bench "bench-$rep" --implementations "${labels[@]}" --warmup 100 --rounds 12 --iters 100
done
for impl in "$BASE" "$DEFAULT"; do
  bench "latency-$impl" --implementations "$impl" --profile "$impl"
  bench "L2Cache-$impl" --implementations "$impl" --profile "$impl" --profile-aic-metric L2Cache
done
if ! "$PY" operators/mhc/vllm_v3/summarize_projection_l2.py "$OUT" > "$OUT/summary.log" 2>&1; then
  "$PY" operators/mhc/vllm_v3/report_failure.py --stage summary --log "$OUT/summary.log"
  exit 1
fi
cat "$OUT/projection-l2-summary.txt"

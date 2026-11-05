#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
PY="$CONDA_PREFIX/bin/python"
BASE=tilelang_coeff_bound
CANDIDATE=tilelang_all_token_h5120s3
common=(--suite mhc_vllm_v3 --mhc-vllm-stage pre --tokens 8192)
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
  bench "bench-$rep" --implementations ascendc_vllm_v3 "$BASE" "$CANDIDATE" \
    --warmup 100 --rounds 12 --iters 100
done
for impl in "$BASE" "$CANDIDATE"; do
  bench "profile-$impl" --implementations "$impl" --profile "$impl"
done
summary=("$PY" operators/mhc/vllm_v3/summarize_projection.py --input "$OUT")
"${summary[@]}"
sim_runs=()
for variant in baseline candidate; do
  echo "N=8192 simulation projection $variant"
  log="$OUT/sim-$variant.log"
  if ! MHC_SIM_TOKENS=8192 MHC_SIM_CORE=0 MHC_SIM_STAGE=projection MHC_SIM_TIMEOUT=60 \
    bash operators/mhc/tuning/run_all_token_simulator.sh "$variant" > "$log" 2>&1; then
    "$PY" operators/mhc/vllm_v3/report_failure.py --stage "simulation $variant" --log "$log"
    cp error.txt "$OUT/sim-$variant-failure.txt"
    cat "$OUT/device-summary.txt" "$OUT/sim-$variant-failure.txt" > error.txt
    exit 1
  fi
  sim_runs+=("$(sed -n 's/^Results: //p' "$log" | head -n 1)")
done
"${summary[@]}" --baseline "${sim_runs[0]}" --candidate "${sim_runs[1]}"

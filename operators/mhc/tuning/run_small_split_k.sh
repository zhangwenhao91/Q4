#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_NPU_ARCH=dav-3510
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
PY="$CONDA_PREFIX/bin/python"
base=tilelang_all_token_l2_default
mode="${1:-scan}"
case "$mode" in
  scan) shapes=(128 256 512) ;;
  batch|alloc|workspace|refactor|timeline|fused-apply|fused-pack128|fused-projection-coeff|fused-projection-coeff-k8) shapes=(128 256) ;;
  *) echo 'Usage: run_small_split_k.sh [scan|batch|alloc|refactor|timeline|fused-apply|fused-pack128|fused-projection-coeff|fused-projection-coeff-k8|workspace [run-directory]]'; exit 2 ;;
esac
resume="${2:-}"
results="$ROOT/results/mhc-small-split-k"
if [[ -n "$resume" ]]; then
  [[ "$mode" == workspace && $# == 2 ]] || { echo 'Only workspace supports resume'; exit 2; }
  OUT="$(realpath -e -- "$resume")"
  [[ "$OUT" == "$results/"* && -f "$OUT/revision.txt" ]] || { echo 'Invalid results directory'; exit 2; }
  # The first workspace result predates the mode marker.
  [[ "$(cat "$OUT/mode.txt" 2>/dev/null || true)" == workspace ||
     -f "$OUT/N128-profile-batch-workspace/metadata.json" ]] || { echo 'Not a workspace run'; exit 2; }
else
  OUT="$results/$(date +%Y%m%d-%H%M%S)-$$"
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/revision.txt"
fi
printf '%s\n' "$mode" > "$OUT/mode.txt"
echo "Results: $OUT"

run_case() {
  local tag="$1" n="$2"
  shift 2
  if [[ -n "$resume" ]]; then
    if [[ -f "$OUT/$tag/summary.json" ]] ||
       [[ "$tag" == *-profile-* && -n "$(find "$OUT/$tag" -type f -name kernel_details.csv -print -quit 2>/dev/null)" ]]; then
      echo "N=$n $tag already complete"
      return
    fi
    local saved="$OUT/$tag.failed-$(date +%Y%m%d-%H%M%S)-$$"
    if [[ -e "$OUT/$tag" ]]; then mv -- "$OUT/$tag" "$saved"; fi
    if [[ -e "$OUT/$tag.log" ]]; then mv -- "$OUT/$tag.log" "$saved.log"; fi
  fi
  echo "N=$n $tag"
  if ! "$PY" operators/spec_v1/bench.py --suite mhc_vllm_v3 --mhc-vllm-stage pre \
    --mhc-projection-l2-ab --tokens "$n" --output "$OUT/$tag" "$@" > "$OUT/$tag.log" 2>&1; then
    "$PY" operators/mhc/vllm_v3/report_failure.py --stage "$tag" --log "$OUT/$tag.log"
    exit 1
  fi
}

if [[ "$mode" == timeline ]]; then
  label=tilelang_deferred_k8_batch_prealloc
  for n in "${shapes[@]}"; do
    run_case "N$n-profile-batch-prealloc" "$n" --implementations "$label" --profile "$label"
  done
  status=0
  "$PY" operators/mhc/vllm_v3/summarize_small_split_timeline.py "$OUT" --prealloc-only \
    --output "$OUT/timeline-summary.txt" || status=$?
  {
    cat "$OUT/timeline-summary.txt"
    echo 'Profiler timeline and CSV files:'
    find "$OUT" -type f \( -name 'trace_view.json' -o -name 'kernel_details.csv' \) -print
  } | tee "$ROOT/error.txt"
  exit "$status"
fi

if [[ "$mode" == scan || "$mode" == batch ]]; then
  for n in "${shapes[@]}"; do
    run_case "N$n-profile-base" "$n" --implementations "$base" --profile "$base"
  done
fi
for n in "${shapes[@]}"; do
  if [[ "$mode" == fused-projection-coeff-k8 ]]; then
    labels=(tilelang_deferred_k8_batch_prealloc tilelang_projection_coeff_k8_prealloc)
  elif [[ "$mode" == fused-projection-coeff ]]; then
    labels=(tilelang_deferred_k8_batch_prealloc tilelang_projection_coeff_prealloc
            tilelang_projection_coeff_m32_prealloc)
  elif [[ "$mode" == fused-pack128 ]]; then
    labels=(tilelang_fused_small_prealloc tilelang_fused_small_prealloc_pack128)
  elif [[ "$mode" == fused-apply ]]; then
    labels=(tilelang_fused_small_prealloc tilelang_fused_small_prealloc_u2 tilelang_fused_small_prealloc_u4)
  elif [[ "$mode" == refactor ]]; then
    labels=(tilelang_deferred_k8_batch_prealloc_reference tilelang_deferred_k8_batch_prealloc)
  elif [[ "$mode" == workspace ]]; then
    labels=(tilelang_deferred_k8_batch tilelang_deferred_k8_batch_workspace
            tilelang_deferred_k8_batch_prealloc)
  elif [[ "$mode" == alloc ]]; then
    labels=(tilelang_deferred_k8 tilelang_deferred_k8_prealloc
            tilelang_deferred_k8_batch tilelang_deferred_k8_batch_prealloc)
  elif [[ "$mode" == batch ]]; then
    labels=("$base" tilelang_deferred_k8 tilelang_deferred_k8_batch)
  else
    labels=("$base" tilelang_deferred_k4 tilelang_deferred_k8)
    if [[ "$n" == 128 ]]; then labels+=(tilelang_deferred_k16); fi
  fi
  for rep in 1 2 3; do
    run_case "N$n-rep$rep" "$n" --implementations "${labels[@]}" \
      --warmup 100 --rounds 24 --iters 100
  done
  if [[ "$mode" == refactor ]]; then continue; fi
  if [[ "$mode" == fused-apply || "$mode" == fused-pack128 || "$mode" == fused-projection-coeff* ]]; then
    for label in "${labels[@]}"; do
      run_case "N$n-profile-$label" "$n" --implementations "$label" --profile "$label"
    done
    continue
  fi
  if [[ "$mode" == workspace ]]; then
    for tag in batch batch-workspace batch-prealloc; do
      label="tilelang_deferred_k8_${tag//-/_}"
      run_case "N$n-profile-$tag" "$n" --implementations "$label" --profile "$label"
    done
    continue
  fi
  run_case "N$n-profile-k8" "$n" --implementations tilelang_deferred_k8 --profile tilelang_deferred_k8
  if [[ "$mode" == batch || "$mode" == alloc ]]; then
    run_case "N$n-profile-batch" "$n" --implementations tilelang_deferred_k8_batch --profile tilelang_deferred_k8_batch
  fi
  if [[ "$mode" == alloc ]]; then
    run_case "N$n-profile-k8-prealloc" "$n" --implementations tilelang_deferred_k8_prealloc --profile tilelang_deferred_k8_prealloc
    run_case "N$n-profile-batch-prealloc" "$n" --implementations tilelang_deferred_k8_batch_prealloc --profile tilelang_deferred_k8_batch_prealloc
  fi
done
"$PY" operators/mhc/vllm_v3/summarize_small_split_k.py "$OUT" "$mode" | tee "$ROOT/error.txt"

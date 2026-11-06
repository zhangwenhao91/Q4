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
OUT="$ROOT/results/mhc-small-split-k/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/revision.txt"
echo "Results: $OUT"
base=tilelang_all_token_l2_default
mode="${1:-scan}"
case "$mode" in
  scan) shapes=(128 256 512) ;;
  batch|alloc|workspace) shapes=(128 256) ;;
  *) echo 'Usage: run_small_split_k.sh [scan|batch|alloc|workspace]'; exit 2 ;;
esac

run_case() {
  local tag="$1" n="$2"
  shift 2
  echo "N=$n $tag"
  if ! "$PY" operators/spec_v1/bench.py --suite mhc_vllm_v3 --mhc-vllm-stage pre \
    --mhc-projection-l2-ab --tokens "$n" --output "$OUT/$tag" "$@" > "$OUT/$tag.log" 2>&1; then
    "$PY" operators/mhc/vllm_v3/report_failure.py --stage "$tag" --log "$OUT/$tag.log"
    exit 1
  fi
}

if [[ "$mode" == scan || "$mode" == batch ]]; then
  for n in "${shapes[@]}"; do
    run_case "N$n-profile-base" "$n" --implementations "$base" --profile "$base"
  done
fi
for n in "${shapes[@]}"; do
  if [[ "$mode" == workspace ]]; then
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

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
ACTION="${1:-probe}"
case "$ACTION" in
  probe)
    OUT="results/mha-tuning/timeline-probe-$(date +%Y%m%d-%H%M%S)-$$"
    mkdir -p "$(dirname "$OUT")"
    python operators/mha/tuning/timeline_probe.py --output "$OUT" 2>&1 | tee "$OUT.log"
    ;;
  load|pipeline)
    OUT="results/mha-tuning/timeline-$ACTION-$(date +%Y%m%d-%H%M%S)-$$"
    mkdir -p "$OUT"
    PYTHON="$(command -v python)"
    MSPROF="$(command -v msprof)"
    "$MSPROF" op --version > "$OUT/tool-version.txt" 2>&1
    "$MSPROF" op --help > "$OUT/tool-help.txt" 2>&1
    if [[ "$ACTION" == load ]]; then
      METRICS="Default,Occupancy"
    else
      METRICS="Default,PipeTimeline"
    fi
    for impl in tilelang_s2buf tilelang_kv2 ascendc; do
      mkdir -p "$OUT/$impl"
      if [[ "$impl" == ascendc ]]; then KERNEL="FlashAttn"; else KERNEL="main_kernel"; fi
      # Validate once outside the collector. Capture app repeats validation/warmup outside MSTX.
      "$PYTHON" operators/mha/tuning/timeline_target.py --implementation "$impl" \
        --mode validate --output "$OUT/$impl/validate" 2>&1 | tee "$OUT/$impl/validate.log"
      command=("$MSPROF" op "--output=$ROOT/$OUT/$impl/op" "--aic-metrics=$METRICS"
        "--kernel-name=$KERNEL" --launch-count=1 --warm-up=50 --replay-mode=kernel
        --mstx=true --mstx-include=Q4_TIMELINE
        "$PYTHON" "$ROOT/operators/mha/tuning/timeline_target.py"
        --implementation "$impl" --mode capture --output "$ROOT/$OUT/$impl/app")
      printf '%q ' "${command[@]}" > "$OUT/$impl/command.txt"
      printf '\n' >> "$OUT/$impl/command.txt"
      echo "COLLECT: implementation=$impl metrics=$METRICS output=$OUT/$impl"
      if "${command[@]}" 2>&1 | tee "$OUT/$impl/collect.log"; then
        if "$PYTHON" operators/mha/tuning/timeline_verify.py --input "$OUT/$impl"; then
          echo "Collector and result checks completed: $impl"
        else
          status=$?
          "$PYTHON" operators/mha/tuning/timeline_inventory.py --input "$OUT"
          echo "Collector returned zero but result validation failed for $impl. Stopping." >&2
          exit "$status"
        fi
      else
        status=$?
        "$PYTHON" operators/mha/tuning/timeline_inventory.py --input "$OUT"
        echo "Collection failed for $impl; no metric downgrade. Inspect $OUT/$impl/collect.log" >&2
        exit "$status"
      fi
    done
    "$PYTHON" operators/mha/tuning/timeline_inventory.py --input "$OUT"
    echo "Retain full collection directory: $ROOT/$OUT"
    ;;
  *) echo 'Usage: bash operators/mha/tuning/run_timeline.sh [probe|load|pipeline]' >&2; exit 2 ;;
esac

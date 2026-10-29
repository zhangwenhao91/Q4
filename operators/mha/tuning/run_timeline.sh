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
  *) echo 'Usage: bash operators/mha/tuning/run_timeline.sh probe' >&2; exit 2 ;;
esac

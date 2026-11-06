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
OUT="$ROOT/results/mhc-shifted-single/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUT"
printf 'Q4 revision: %s\n' "$(git rev-parse HEAD)" > "$OUT/runner.log"
echo "Results: $OUT"

baseline=tilelang_all_token_l2_default
case "${1:-original}" in
  original)
    candidate=tilelang_shifted_single_h5120
    kernel=mhc_shifted_single_kernel
    ;;
  reuse)
    candidate=tilelang_shifted_single_reuse_h5120
    kernel=mhc_shifted_single_reuse_kernel
    ;;
  *) echo 'Usage: run_shifted_single.sh [original|reuse]'; exit 2 ;;
esac
printf 'Candidate: %s\n' "$candidate" >> "$OUT/runner.log"
run_case() {
  local tag="$1" n="$2"
  shift 2
  echo "H=5120 N=$n $tag"
  if ! "$PY" operators/spec_v1/bench.py --suite mhc_vllm_v3 --mhc-vllm-stage pre \
    --mhc-projection-l2-ab --tokens "$n" --output "$OUT/$tag" "$@" > "$OUT/$tag.log" 2>&1; then
    "$PY" operators/mhc/vllm_v3/report_failure.py --stage "$tag" --log "$OUT/$tag.log"
    exit 1
  fi
}

run_case check 128 --implementations "$baseline" "$candidate" --check-only
run_case profile 128 --implementations "$candidate" --profile "$candidate"
if ! "$PY" - "$OUT/profile" "$kernel" > "$OUT/launch-check.log" 2>&1 <<'PYTHON'
import csv
from pathlib import Path
import sys

paths = list(Path(sys.argv[1]).rglob('kernel_details.csv'))
if len(paths) != 1:
    raise RuntimeError(f'Expected one kernel_details.csv; found {paths}')
with paths[0].open(encoding='utf-8-sig', newline='') as stream:
    rows = csv.DictReader(stream)
    field = next((name for name in ('Name', 'Op Name', 'Kernel Name') if name in (rows.fieldnames or ())), None)
    if field is None:
        raise RuntimeError(f'Unknown kernel name column: {rows.fieldnames}')
    names = [row[field] for row in rows if row.get(field)]
if len(names) != 30 or not all(sys.argv[2] in name for name in names):
    raise RuntimeError(f'Expected 30 single-kernel launches, observed {len(names)}: {sorted(set(names))}')
print(f'Profiler verified 30 launches for 30 Pre calls at N=128: {names[0]}')
PYTHON
then
  "$PY" operators/mhc/vllm_v3/report_failure.py --stage launch-check --log "$OUT/launch-check.log"
  exit 1
fi
cat "$OUT/launch-check.log" | tee -a "$OUT/runner.log"

for n in 128 256 512 2048 8192; do
  for rep in 1 2 3; do
    labels=("$baseline" "$candidate")
    if [[ "$rep" == 2 ]]; then labels=("$candidate" "$baseline"); fi
    run_case "N$n-rep$rep" "$n" --implementations "${labels[@]}" --warmup 100 --rounds 12 --iters 100
  done
done
"$PY" operators/mhc/tuning/summarize_token_pipeline.py --input "$OUT" --output "$OUT/summary.txt" \
  --reference "$baseline" --baseline "$candidate" --candidate ''
sed -e 's/ratios > 1 mean TL is faster/ratios > 1 mean single kernel is faster/' \
    -e 's/case | AC us | TL us | AC\/TL/case | three-kernel TL us | single-kernel TL us | three\/single/' \
    -e 's/AC reference:/Three-kernel reference:/' "$OUT/summary.txt" | tee "$ROOT/error.txt"

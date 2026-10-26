#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
SOURCE="$ROOT/third_party/DeepGEMM-Ascend"
RECORD="$ROOT/third_party/hc-aiv-layout-fix"
python "$ROOT/operators/mhc/extended/patch_hc_aiv_layout.py" --source "$SOURCE" --record "$RECORD"
# Retain/re-apply the previously verified host-toolkit fix as well.
python "$ROOT/operators/mhc/extended/patch_deepjit_host.py" --source "$SOURCE" --record "$ROOT/third_party/deepjit-host-fix"
cd "$SOURCE"
python -m pip install --force-reinstall --no-deps --no-build-isolation .
cd "$ROOT"
python - <<'PY'
import hashlib
from pathlib import Path
import deep_gemm
src = Path('third_party/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py')
dst = Path(deep_gemm.__file__).parent/'tilelang_ops/hc_prenorm.py'
assert src.read_bytes() == dst.read_bytes(), 'Installed HC module differs from patched source'
print('Installed HC SHA256:', hashlib.sha256(dst.read_bytes()).hexdigest())
print('Explicit AIV patch installed; NPU compilation and correctness still require run_fused.sh check.')
PY

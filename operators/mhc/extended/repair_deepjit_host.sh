#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
SOURCE="$ROOT/third_party/DeepGEMM-Ascend"
RECORD="$ROOT/third_party/deepjit-host-fix"
python "$ROOT/operators/mhc/extended/patch_deepjit_host.py" --source "$SOURCE" --record "$RECORD"
cd "$SOURCE"
python -m pip install --force-reinstall --no-deps --no-build-isolation .
cd "$ROOT"
python - <<'PY'
import importlib.metadata
import torch
import torch_npu
import deep_gemm
# Exercise the lazy JIT initializer: import/API existence alone missed this failure.
torch.npu.set_device(0)
print('distribution:', importlib.metadata.version('deep_gemm'))
print('module:', deep_gemm.__file__)
print('AI cores:', deep_gemm.get_num_sms())
print('DeepGEMM JIT initialization passed')
PY
python -m pip freeze > "$RECORD/pip-freeze.txt"
echo "Patch provenance: $RECORD"
echo 'This source now contains an intentional patch; do not re-run the original clean-source installer.'

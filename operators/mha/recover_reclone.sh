#!/usr/bin/env bash
# Restore only missing Q4-local MHA assets; keep the installed Python environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?Activate zwh first}"
[[ "$(basename "$CONDA_PREFIX")" == zwh ]] || { echo 'Activate the existing zwh environment.' >&2; exit 1; }
: "${ASCEND_HOME_PATH:?Source the CANN set_env.sh first}"
test -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
test -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
command -v msprof >/dev/null
python -c 'import torch, torch_npu, tilelang; print("Existing imports:", torch.__version__, torch_npu.__version__, tilelang.__version__)'
mkdir -p third_party results/mha-recovery
LOG="$ROOT/results/mha-recovery/restore-$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(tee "$LOG") 2>&1
echo "Restore log: $LOG"
python operators/spec_v1/prepare_mha.py
SRC="$ROOT/third_party/ops-transformer"
REV=ba5162411c919e15fbabb28298e8b420c11ad905
if [[ ! -e "$SRC" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://gitcode.com/cann/ops-transformer.git "$SRC"
fi
git -C "$SRC" rev-parse --git-dir >/dev/null
if ! git -C "$SRC" cat-file -e "$REV^{commit}"; then
  git -C "$SRC" fetch origin "$REV"
fi
# Existing helper pins a detached worktree, verifies source, seeds checked archives,
# strips incompatible Conda flags, and builds only FlashAttn/FlashAttnMetadata.
bash operators/mha/build_ascendc.sh build
bash operators/mha/build_ascendc.sh install
test -d "$ROOT/third_party/q4fa-opp/vendors/q4fa_transformer/op_api/lib"
echo 'MHA source and native package restored. No Python package reinstall performed.'
echo 'Next: source operators/mha/env_recovered.sh'
echo 'Then: bash operators/mha/tuning/run_timeline.sh pipeline'

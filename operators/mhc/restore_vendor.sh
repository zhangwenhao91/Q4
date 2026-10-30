#!/usr/bin/env bash
# Restore the native Post/Pre/Sinkhorn package without reinstalling Python packages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
SRC="$ROOT/third_party/ops-transformer"
TARGET="$ROOT/third_party/ops-transformer-q4mhc"
INSTALL="$ROOT/third_party/q4mhc-opp"
REV=ba5162411c919e15fbabb28298e8b420c11ad905
mkdir -p "$ROOT/results/mhc-recovery" "$ROOT/third_party"
LOG="$ROOT/results/mhc-recovery/vendor-$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(tee "$LOG") 2>&1
echo "Recovery log: $LOG"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
"$CXX" --version
if [[ ! -e "$SRC" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://gitcode.com/cann/ops-transformer.git "$SRC"
fi
git -C "$SRC" cat-file -e "$REV^{commit}" 2>/dev/null || git -C "$SRC" fetch origin "$REV"
if [[ ! -e "$TARGET" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git -C "$SRC" worktree add --detach "$TARGET" "$REV"
fi
[[ "$(git -C "$TARGET" rev-parse HEAD)" == "$REV" ]] || { echo 'Unexpected mHC source revision'; exit 1; }
[[ -z "$(git -C "$TARGET" status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked source modified: $TARGET; stopping without overwriting it."; exit 1;
}
for flag_var in CFLAGS CXXFLAGS CPPFLAGS LDFLAGS ASFLAGS; do
  unset "$flag_var"
done
python - <<'PY'
from pathlib import Path
import sys
root = Path.cwd()
sys.path.insert(0, str(root / 'operators/mha'))
from clean_build_flags import clean_cache
from seed_dependency_cache import seed
target = root / 'third_party/ops-transformer-q4mhc'
clean_cache(target / 'build/CMakeCache.txt')
seed(root / 'third_party/ops-transformer/third_party', target / 'third_party')
PY
cd "$TARGET"
bash -o pipefail build.sh --incremental --pkg --soc=ascend950 --vendor_name=q4mhc \
  --ops=mhc_post,mhc_pre,mhc_sinkhorn,mhc_pre_sinkhorn "-j${Q4_BUILD_JOBS:-8}"
PACKAGE="$TARGET/build_out/cann-ops-transformer-q4mhc_linux-x86_64.run"
test -f "$PACKAGE"
mkdir -p "$INSTALL"
bash "$PACKAGE" --install-path="$INSTALL"
test -f "$INSTALL/vendors/q4mhc_transformer/op_api/lib/libcust_opapi.so"
echo 'mHC vendor restored. In your current shell run:'
echo "export ASCEND_CUSTOM_OPP_PATH='$INSTALL/vendors/q4mhc_transformer'"
echo 'export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"'
echo 'bash operators/mhc/run_baseline.sh preflight'

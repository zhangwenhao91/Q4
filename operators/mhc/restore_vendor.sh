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
VENDOR=q4mhc
OPS=mhc_post,mhc_pre,mhc_sinkhorn,mhc_pre_sinkhorn
if [[ "${1:-}" == h5120-fused ]]; then
  TARGET="$ROOT/third_party/ops-transformer-q4mhc5120"
  INSTALL="$ROOT/third_party/q4mhc5120-opp"
  VENDOR=q4mhc5120
  OPS=mhc_pre_sinkhorn
elif [[ $# != 0 ]]; then
  echo 'Usage: restore_vendor.sh [h5120-fused]'
  exit 2
fi
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
  # At REV only PNG assets use LFS. Keep their pointers without requiring git-lfs.
  git -C "$SRC" -c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.required=false \
    worktree add --detach "$TARGET" "$REV"
fi
[[ "$(git -C "$TARGET" rev-parse HEAD)" == "$REV" ]] || { echo 'Unexpected mHC source revision'; exit 1; }
status="$(git -C "$TARGET" status --porcelain --untracked-files=no)"
if [[ "$VENDOR" == q4mhc5120 ]]; then
  patch="$ROOT/operators/mhc/tuning/fused_h5120.patch"
  if [[ -z "$status" ]]; then
    git -C "$TARGET" apply "$patch"
    status="$(git -C "$TARGET" status --porcelain --untracked-files=no)"
  fi
  [[ "$status" == ' M mhc/mhc_pre_sinkhorn/op_host/op_tiling/mhc_pre_sinkhorn_tiling_arch35.cpp' ]]
  git -C "$TARGET" apply --reverse --check "$patch"
else
  [[ -z "$status" ]] || {
  echo "Tracked source modified: $TARGET; stopping without overwriting it."; exit 1;
  }
fi
for flag_var in CFLAGS CXXFLAGS CPPFLAGS LDFLAGS ASFLAGS; do
  unset "$flag_var"
done
python - "$TARGET" <<'PY'
from pathlib import Path
import sys
root = Path.cwd()
sys.path.insert(0, str(root / 'operators/mha'))
from clean_build_flags import clean_cache
from seed_dependency_cache import seed
target = Path(sys.argv[1])
clean_cache(target / 'build/CMakeCache.txt')
seed(root / 'third_party/ops-transformer/third_party', target / 'third_party')
PY
cd "$TARGET"
bash -o pipefail build.sh --incremental --pkg --soc=ascend950 --vendor_name="$VENDOR" \
  --ops="$OPS" "-j${Q4_BUILD_JOBS:-8}"
PACKAGE="$TARGET/build_out/cann-ops-transformer-${VENDOR}_linux-x86_64.run"
test -f "$PACKAGE"
mkdir -p "$INSTALL"
bash "$PACKAGE" --install-path="$INSTALL"
test -f "$INSTALL/vendors/${VENDOR}_transformer/op_api/lib/libcust_opapi.so"
echo 'mHC vendor restored. In your current shell run:'
echo "export ASCEND_CUSTOM_OPP_PATH='$INSTALL/vendors/${VENDOR}_transformer'"
echo 'export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"'
echo 'bash operators/mhc/run_baseline.sh preflight'

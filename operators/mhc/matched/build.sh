#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
SRC="$ROOT/third_party/ops-transformer"
TARGET="$ROOT/third_party/ops-transformer-q4mhcmatch"
INSTALL="$ROOT/third_party/q4mhcmatch-opp"
REV=ba5162411c919e15fbabb28298e8b420c11ad905
mkdir -p results/mhc-matched-build
LOG="$ROOT/results/mhc-matched-build/$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(tee "$LOG") 2>&1
echo "Build log: $LOG"
git -C "$SRC" cat-file -e "$REV^{commit}"
if [[ ! -e "$TARGET" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git -C "$SRC" worktree add --detach "$TARGET" "$REV"
fi
[[ "$(git -C "$TARGET" rev-parse HEAD)" == "$REV" ]]
[[ -z "$(git -C "$TARGET" status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked source changed; stopping without overwriting it.'; exit 1;
}
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
for flag_var in CFLAGS CXXFLAGS CPPFLAGS LDFLAGS ASFLAGS; do unset "$flag_var"; done
python operators/mhc/matched/prepare_native.py --source "$TARGET" \
  --record "$ROOT/third_party/mhc-matched-source"
python - <<'PY'
from pathlib import Path
import sys
root = Path.cwd()
sys.path.insert(0, str(root / 'operators/mha'))
from clean_build_flags import clean_cache
from seed_dependency_cache import seed
target = root / 'third_party/ops-transformer-q4mhcmatch'
clean_cache(target / 'build/CMakeCache.txt')
seed(root / 'third_party/ops-transformer/third_party', target / 'third_party')
PY
cd "$TARGET"
bash -o pipefail build.sh --incremental --pkg --soc=ascend950 --vendor_name=q4mhcmatch \
  --ops=mhc_post,mhc_sinkhorn,mhc_shifted_pre "-j${Q4_BUILD_JOBS:-8}"
PACKAGE="$TARGET/build_out/cann-ops-transformer-q4mhcmatch_linux-x86_64.run"
test -f "$PACKAGE"
mkdir -p "$INSTALL"
bash "$PACKAGE" --install-path="$INSTALL"
test -f "$INSTALL/vendors/q4mhcmatch_transformer/op_api/lib/libcust_opapi.so"
echo 'Native matched package built and installed; device correctness is not yet verified.'
echo 'Next: bash operators/mhc/matched/run.sh check'

#!/usr/bin/env bash
# Build/install stock native AscendC attention in an isolated checkout/vendor.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/third_party/ops-transformer"
TARGET="$ROOT/third_party/ops-transformer-q4fa"
INSTALL="$ROOT/third_party/q4fa-opp"
REV=ba5162411c919e15fbabb28298e8b420c11ad905
ACTION="${1:-build}"
: "${CONDA_PREFIX:?Activate zwh}"
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
"$CXX" --version
case "$ACTION" in
  build)
    git -C "$SRC" cat-file -e "$REV^{commit}"
    if [[ ! -d "$TARGET" ]]; then
      GIT_LFS_SKIP_SMUDGE=1 git -C "$SRC" worktree add --detach "$TARGET" "$REV"
    fi
    [[ "$(git -C "$TARGET" rev-parse HEAD)" == "$REV" ]] || {
      echo "Wrong source revision at $TARGET; preserve it and resolve before building." >&2; exit 1;
    }
    [[ -z "$(git -C "$TARGET" status --porcelain --untracked-files=no)" ]] || {
      echo "Tracked source modified at $TARGET; preserve it before building." >&2; exit 1;
    }
    # Host GCC and ARM AICPU share global CMake flags. Conda's x86 ISA flags
    # must not leak into the ARM compiler; keep sanitation local to this script.
    for flag_var in CFLAGS CXXFLAGS CPPFLAGS LDFLAGS ASFLAGS; do
      if [[ -n "${!flag_var:-}" ]]; then
        echo "Ignoring inherited $flag_var for this mixed-architecture build."
      fi
      unset "$flag_var"
    done
    python "$ROOT/operators/mha/clean_build_flags.py"
    python "$ROOT/operators/mha/seed_dependency_cache.py"
    cd "$TARGET"
    bash -o pipefail build.sh --incremental --pkg --soc=ascend950 --vendor_name=q4fa \
      --ops=flash_attn,flash_attn_metadata "-j${Q4_BUILD_JOBS:-8}"
    [[ -f "$TARGET/build_out/cann-ops-transformer-q4fa_linux-x86_64.run" ]] || {
      echo "Build did not produce the expected q4fa package; inspect the full log." >&2; exit 1;
    }
    ;;
  install)
    PACKAGE="$TARGET/build_out/cann-ops-transformer-q4fa_linux-x86_64.run"
    [[ -f "$PACKAGE" ]] || { echo "Missing package: $PACKAGE" >&2; exit 1; }
    mkdir -p "$INSTALL"
    bash "$PACKAGE" --install-path="$INSTALL"
    find "$INSTALL" -name set_env.bash -print
    ;;
  *) echo 'Usage: bash operators/mha/build_ascendc.sh [build|install]' >&2; exit 2 ;;
esac

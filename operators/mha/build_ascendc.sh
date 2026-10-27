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
    # Reuse the exact-version archive from the previously successful mHC build.
    # A fresh worktree has no ignored dependency cache; corporate TLS may block re-download.
    JSON_SRC="$SRC/third_party/pkg/json-3.12.0.tar.gz"
    JSON_DST="$TARGET/third_party/pkg/json-3.12.0.tar.gz"
    if [[ -f "$JSON_SRC" ]]; then
      tar -tzf "$JSON_SRC" > /dev/null
      mkdir -p "$TARGET/third_party/pkg"
      if ! cmp -s "$JSON_SRC" "$JSON_DST"; then
        cp -p -- "$JSON_SRC" "$JSON_DST"
      fi
      echo "Reused JSON 3.12.0 archive from previous build:"
      sha256sum "$JSON_DST"
    else
      echo "No prior JSON cache at $JSON_SRC; upstream may need a network download." >&2
    fi
    cd "$TARGET"
    bash build.sh --pkg --soc=ascend950 --vendor_name=q4fa \
      --ops=flash_attn,flash_attn_metadata "-j${Q4_BUILD_JOBS:-8}"
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

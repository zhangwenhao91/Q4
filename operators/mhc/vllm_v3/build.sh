#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
SRC="$ROOT/third_party/vllm-ascend-q4-v3"
REV=e838a32888113cd20e1b5bad2053d1b4a06bf935
INSTALL="$ROOT/third_party/q4vllmv3-opp"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS="${Q4_BUILD_JOBS:-8}"
if [[ ! -d "$SRC" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm-ascend.git "$SRC"
  git -C "$SRC" fetch --depth 1 origin "$REV"
  git -C "$SRC" checkout --detach "$REV"
fi
"$CONDA_PREFIX/bin/python" operators/mhc/vllm_v3/prepare.py
cd "$SRC/csrc"
# The upstream build entry accepts individual operators and a private vendor.
# No kernel, tiling, adapter or upstream binding source is patched.
bash build.sh --pkg --soc=ascend950 --ops=hc_pre --vendor_name=q4vllmv3 "-j${MAX_JOBS}"
mapfile -t packages < <(find build build_out -maxdepth 2 -type f -name '*q4vllmv3*.run' 2>/dev/null)
if [[ "${#packages[@]}" -ne 1 ]]; then
  printf 'Expected one q4vllmv3 installer, found %s\n' "${#packages[@]}"
  printf '%s\n' "${packages[@]}"
  exit 1
fi
mkdir -p "$INSTALL"
bash "${packages[0]}" --install-path="$INSTALL"
test -f "$INSTALL/vendors/q4vllmv3_transformer/op_api/lib/libcust_opapi.so"
"$CONDA_PREFIX/bin/python" "$ROOT/operators/mhc/vllm_v3/prepare.py" --installer "${packages[0]}"
echo 'Pinned HcPre installed to isolated q4vllmv3 vendor. Device validation follows in run.sh.'

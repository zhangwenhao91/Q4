#!/usr/bin/env bash
# Source this file in the NPU server shell after recovery succeeds.
q4_mha_set_environment() {
  local root prefix
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 1
  prefix="${CONDA_PREFIX:-}"
  if [[ -z "$prefix" || ! -x "$prefix/bin/x86_64-conda-linux-gnu-g++" ]]; then
    echo 'Activate zwh with its installed compiler first.' >&2; return 1
  fi
  if [[ ! -d "$root/third_party/q4fa-opp/vendors/q4fa_transformer/op_api/lib" ]]; then
    echo 'Missing q4fa native library directory; complete recovery first.' >&2; return 1
  fi
  export CC="$prefix/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$prefix/bin/x86_64-conda-linux-gnu-g++"
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
  export ASCEND_NPU_ARCH=dav-3510
  export ASCEND_CUSTOM_OPP_PATH="$root/third_party/q4fa-opp/vendors/q4fa_transformer"
  export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
  export TORCH_EXTENSIONS_DIR="$root/third_party/torch-extensions-gcc13"
  mkdir -p "$TORCH_EXTENSIONS_DIR" || return 1
  echo "MHA environment ready; physical devices=$ASCEND_RT_VISIBLE_DEVICES, logical target=0"
}
q4_mha_set_environment

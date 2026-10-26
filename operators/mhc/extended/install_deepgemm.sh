#!/usr/bin/env bash
# Run from any directory, inside the already configured zwh environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REV=3a35326271b069b8b5bf1a754435acd6cc91575b
TARGET="$ROOT/third_party/DeepGEMM-Ascend"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
"$CXX" --version
command -v bisheng
mkdir -p "$ROOT/third_party/toolchain-check"
cat > "$ROOT/third_party/toolchain-check/format.cpp" <<'CPP'
#include <format>
#include <iostream>
int main() { std::cout << std::format("C++{} format OK\n", 20); }
CPP
"$CXX" -std=c++20 "$ROOT/third_party/toolchain-check/format.cpp" -o "$ROOT/third_party/toolchain-check/format-check"
"$ROOT/third_party/toolchain-check/format-check"
if [[ ! -d "$TARGET" ]]; then
    git clone https://github.com/deepseek-ai/DeepGEMM-Ascend.git "$TARGET"
fi
if [[ -n "$(git -C "$TARGET" status --porcelain)" ]]; then
    echo "Source has local changes: $TARGET. Preserve them before proceeding." >&2
    exit 1
fi
git -C "$TARGET" cat-file -e "$REV^{commit}" 2>/dev/null || git -C "$TARGET" fetch origin "$REV"
git -C "$TARGET" checkout --detach "$REV"
git -C "$TARGET" submodule update --init --recursive
python -m pip install 'tree-sitter>=0.25' tree-sitter-cpp setuptools wheel ninja
cd "$TARGET"
# Preserve the tested torch/torch_npu/tilelang environment. Build the native _C module too.
python -m pip install --no-deps --no-build-isolation .
cd "$ROOT"
python - <<'PY'
import importlib.metadata
import deep_gemm
print('distribution:', importlib.metadata.version('deep_gemm'))
print('module:', deep_gemm.__file__)
assert callable(deep_gemm.tf32_hc_prenorm_gemm)
print('HC fused projection API available')
PY
python -m pip freeze > "$ROOT/third_party/deepgemm-install-freeze.txt"
git -C "$TARGET" rev-parse HEAD > "$ROOT/third_party/deepgemm-revision.txt"
git -C "$TARGET" submodule status --recursive >> "$ROOT/third_party/deepgemm-revision.txt"

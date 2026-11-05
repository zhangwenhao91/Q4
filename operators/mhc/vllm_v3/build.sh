#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?Activate zwh first}"
: "${ASCEND_HOME_PATH:?Source CANN first}"
SRC="$ROOT/third_party/vllm-ascend-q4-v3-archive"
REV=e838a32888113cd20e1b5bad2053d1b4a06bf935
INSTALL="$ROOT/third_party/q4vllmv3-opp"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS="${Q4_BUILD_JOBS:-8}"
if ! "$CONDA_PREFIX/bin/python" -c 'import regex' >/dev/null 2>&1; then
  WHEEL="$ROOT/operators/mhc/vllm_v3/dependency_archives/regex-2024.11.6-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
  "$CONDA_PREFIX/bin/python" - "$WHEEL" <<'PY'
import hashlib
from pathlib import Path
import platform
import sys
wheel = Path(sys.argv[1])
if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('Missing regex; bundled wheel requires CPython 3.12/Linux x86_64')
if hashlib.sha256(wheel.read_bytes()).hexdigest() != '70b7fa6606c2881c1db9479b0eaa11ed5dfa11c8d60a474ff0e095099f39d98e':
    raise RuntimeError('regex wheel checksum mismatch')
PY
  "$CONDA_PREFIX/bin/python" -m pip install --no-index --no-deps "$WHEEL"
  "$CONDA_PREFIX/bin/python" -c 'import regex; print("Build dependency regex:", regex.__version__)'
fi
if ! "$CONDA_PREFIX/bin/python" -c 'from decorator import decorate' >/dev/null 2>&1; then
  WHEEL="$ROOT/operators/mhc/vllm_v3/dependency_archives/decorator-5.2.1-py3-none-any.whl"
  "$CONDA_PREFIX/bin/python" - "$WHEEL" <<'PY'
import hashlib
from pathlib import Path
import sys
wheel = Path(sys.argv[1])
if hashlib.sha256(wheel.read_bytes()).hexdigest() != 'd316bb415a2d9e2d2b3abcc4084c6502fc09240e292cd76a76afc106a1c8e04a':
    raise RuntimeError('decorator wheel checksum mismatch')
PY
  "$CONDA_PREFIX/bin/python" -m pip install --no-index --no-deps "$WHEEL"
fi
PYTHONPATH="$ASCEND_HOME_PATH/python/site-packages${PYTHONPATH:+:$PYTHONPATH}" \
  "$CONDA_PREFIX/bin/python" -c 'from tbe.tikcpp import compile_op; print("CANN TBE compiler import passed")'
ARCHIVE="$ROOT/third_party/vllm-ascend-v3-e838a328.tar.gz"
BUNDLED="$ROOT/operators/mhc/vllm_v3/dependency_archives/vllm-ascend-e838a328.tar.gz"
if [[ -f "$BUNDLED" ]]; then
  mkdir -p "$ROOT/third_party"
  "$CONDA_PREFIX/bin/python" - "$BUNDLED" "$ARCHIVE" <<'PY'
import hashlib
from pathlib import Path
import shutil
import sys
source, target = map(Path, sys.argv[1:])
expected = '98cef93d6983f9cb5b193e200a1e7b6b18ee11daef17eb35e2acfa432b8e82cc'
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Bundled upstream source checksum mismatch')
shutil.copyfile(source, target)
print('Using verified source archive from Q4; no external source download.')
PY
elif [[ ! -f "$ARCHIVE" ]]; then
  mkdir -p "$ROOT/third_party"
  curl --fail --location --retry 3 \
    "https://codeload.github.com/vllm-project/vllm-ascend/tar.gz/$REV" \
    --output "$ARCHIVE.part"
  mv "$ARCHIVE.part" "$ARCHIVE"
fi
"$CONDA_PREFIX/bin/python" operators/mhc/vllm_v3/prepare.py --unpack
# Conda's x86 flags must not leak into the upstream AArch64 tiling-sink build.
# Keep explicit host CC/CXX; the upstream ExternalProject selects hcc itself.
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS ASMFLAGS CMAKE_ARGS
"$CONDA_PREFIX/bin/python" - "$ROOT" "$SRC" <<'PY'
from pathlib import Path
import sys
root, source = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / 'operators/mha'))
from clean_build_flags import clean_cache
clean_cache(source / 'csrc/build/CMakeCache.txt')
for cache in (source / 'csrc/build/common').glob('tiling_sink_task-prefix/src/tiling_sink_task-build/CMakeCache.txt'):
    clean_cache(cache)
print('Host compiler paths preserved; inherited global x86 flags cleared before cross compilation.')
PY
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

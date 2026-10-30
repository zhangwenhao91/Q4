#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="$ROOT/third_party/TileKernels"
REV=c1cf914a82e6901c679445f872051ce44ac2cc63
mkdir -p "$ROOT/third_party"
if [[ ! -e "$TARGET" ]]; then
  git clone https://github.com/deepseek-ai/TileKernels.git "$TARGET"
fi
if [[ -n "$(git -C "$TARGET" status --porcelain)" ]]; then
  echo "Local source changes found: $TARGET; stopping without overwriting them." >&2
  exit 1
fi
git -C "$TARGET" cat-file -e "$REV^{commit}" 2>/dev/null || git -C "$TARGET" fetch origin "$REV"
git -C "$TARGET" checkout --detach "$REV"
python -m pip install --no-deps --no-build-isolation -e "$TARGET"
python - <<'PY'
import tile_kernels
from tile_kernels.mhc import mhc_post_fwd, mhc_pre_apply_mix_fwd
print('TileKernels restored:', tile_kernels.__file__)
PY

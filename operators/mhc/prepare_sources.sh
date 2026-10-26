#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/third_party"
checkout_source() {
    local name="$1" url="$2" revision="$3"
    local target="$ROOT/third_party/$name"
    if [[ ! -d "$target" ]]; then
        git clone "$url" "$target"
    fi
    if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
        echo "Source has local changes: $target; preserve them before proceeding." >&2
        exit 1
    fi
    git -C "$target" cat-file -e "$revision^{commit}" 2>/dev/null || git -C "$target" fetch origin "$revision"
    git -C "$target" checkout --detach "$revision"
    git -C "$target" rev-parse HEAD
}
checkout_source TileKernels https://github.com/deepseek-ai/TileKernels.git c1cf914a82e6901c679445f872051ce44ac2cc63
checkout_source ops-transformer https://gitcode.com/cann/ops-transformer.git ba5162411c919e15fbabb28298e8b420c11ad905

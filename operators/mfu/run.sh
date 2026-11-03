#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python operators/mfu/run.py --operator "${1:-mha}" --scale "${2:-both}"

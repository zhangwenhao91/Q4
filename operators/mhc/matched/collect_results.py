"""Summarize the latest existing check/bench/profile runs; never execute NPU work."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=root / 'results/mhc-matched')
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error('Missing result root: ' + str(args.root))
    runs = []
    for path in args.root.iterdir():
        match = re.fullmatch(r'\d{8}-\d{6}-(check|bench|profile|all)-\d+', path.name)
        if path.is_dir() and match:
            runs.append((path, match.group(1)))
    seen = set()
    for mode in ('check', 'bench', 'profile'):
        candidates = sorted(p for p, m in runs if m in (mode, 'all'))
        if not candidates:
            print('MISSING RUN:', mode, flush=True)
            continue
        selected = candidates[-1]
        print('SELECTED', mode, selected, flush=True)
        if selected in seen:
            continue
        seen.add(selected)
        identities = set()
        for meta_file in selected.glob('*/metadata.json'):
            meta = json.loads(meta_file.read_text())
            native = meta.get('native_adaptation') or {}
            identities.add((meta.get('q4_revision'), native.get('library_sha256'),
                            meta.get('hc_source_sha256'), meta.get('weight_file')))
        for identity in sorted(identities, key=str):
            print('IDENTITY q4/native_library/hc_source/weights:', identity, flush=True)
        # Each run remains separate, including partial runs; no fallback to older successes.
        subprocess.run([sys.executable, str(root / 'operators/mhc/summarize_baseline.py'),
                        '--input', str(selected)], check=True)
    print('Offline collection complete. No kernels executed.', flush=True)


if __name__ == '__main__':
    main()

"""Offline generated-code evidence; no imports of torch/TileLang or NPU execution."""
import argparse
import hashlib
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/mhc-apply-flat'))
    a = p.parse_args()
    runs = sorted(x for x in a.root.glob('*') if x.is_dir() and re.search(r'-(all|profile)-\d+$',x.name))
    if not runs:
        p.error('No all/profile result directories found')
    run = runs[-1]
    print('SELECTED',run)
    for tokens in (128,512):
        folder = run / f'profile-N{tokens}-K1280:3'
        for name in ('apply_pipeline_kernel.cpp','apply_flat_pipeline_kernel.cpp'):
            path = folder / name
            if not path.exists():
                print('MISSING',path)
                continue
            data = path.read_bytes()
            lines = data.decode('utf-8').splitlines()
            print('SOURCE',path,'SHA256',hashlib.sha256(data).hexdigest(),'LINES',len(lines))
            # Keep source order and context; these excerpts are not a full schedule proof.
            match = re.compile(r'for\s*\(|while\s*\(|[Ss]et[Ff]lag|[Ww]ait[Ff]lag|[Ss]ync|[Bb]arrier|[Ee]vent|[Dd]ata[Cc]opy|[Ss]imd|[Vv][Ff]|[Pp]ipe|[Bb]uffer')
            selected = set()
            for i,line in enumerate(lines):
                if match.search(line):
                    selected.update(range(max(0,i-2),min(len(lines),i+4)))
            previous = -2
            for i in sorted(selected):
                if i != previous+1:
                    print('...')
                print(f'{i+1}: {lines[i]}')
                previous = i
    print('Offline excerpts complete. No kernels executed; missing regions may require full source review.')


if __name__ == '__main__':
    main()

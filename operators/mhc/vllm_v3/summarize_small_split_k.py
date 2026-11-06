"""Summarize the small-N split-K A/B and device profiles."""
import json
from pathlib import Path
from statistics import median
import sys

from summarize_projection import profile_lines


def report(root):
    base = 'tilelang_all_token_l2_default'
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'N | implementation | Cube cores | Pre event us (3 processes) | median us | paired baseline/TL']
    for n in (128, 256, 512):
        runs = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        metadata = json.loads((runs[0] / 'metadata.json').read_text())['variants']
        timings = [json.loads((run / 'summary.json').read_text()) for run in runs]
        for label in metadata:
            times = [run[label]['event_us'] for run in timings]
            ratio = median(run[base]['event_us'] / value for run, value in zip(timings, times))
            cores = metadata[label]['projection_config']['num_aic_cores']
            lines.append(f'{n} | {label} | {cores} | {", ".join(f"{v:.3f}" for v in times)} | '
                         f'{median(times):.3f} | {ratio:.4f}')
        for tag, kernel in (('base', 'mhc_projection_token_pipeline'), ('k8', 'mhc_projection_split_partials')):
            lines.extend(profile_lines(root / f'N{n}-profile-{tag}', kernel))
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    root = Path(sys.argv[1])
    summary = report(root)
    (root / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary, end='')

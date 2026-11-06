"""Summarize the small-N split-K A/B and device profiles."""
import json
from pathlib import Path
from statistics import median
import sys

from summarize_projection import profile_lines
from summarize_small_split_timeline import timeline_lines


def allocation_report(root, workspace=False):
    labels = ('tilelang_deferred_k8', 'tilelang_deferred_k8_batch')
    pairs = [(name, f'{name}_prealloc') for name in labels]
    tags = ('k8', 'k8-prealloc', 'batch', 'batch-prealloc')
    if workspace:
        batch = labels[1]
        pairs = [(batch, f'{batch}_workspace'), (batch, f'{batch}_prealloc'),
                 (f'{batch}_workspace', f'{batch}_prealloc')]
        tags = ('batch', 'batch-workspace', 'batch-prealloc')
    hashes = ('standalone_apply_source_sha256', 'projection_device_source_sha256', 'tail_device_source_sha256')
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'CPU and pairwise checks passed. Workspace-only returns fresh outputs; full preallocation aliases.',
             'N | reference | candidate | reference event us | candidate event us | paired reference/candidate | '
             'reference wall us | candidate wall us | source match']
    for n in (128, 256):
        cases = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        timings = [json.loads((case / 'summary.json').read_text()) for case in cases]
        metas = [json.loads((case / 'metadata.json').read_text())['variants'] for case in cases]
        for name, reused in pairs:
            fresh, reuse = ([run[label]['event_us'] for run in timings] for label in (name, reused))
            ratios = [a / b for a, b in zip(fresh, reuse)]
            wall = [median(run[label]['wall_us'] for run in timings) for label in (name, reused)]
            same = all(meta[name][key] == meta[reused][key] for meta in metas for key in hashes)
            lines.append(f'{n} | {name} | {reused} | {median(fresh):.3f} | {median(reuse):.3f} | {median(ratios):.4f} | '
                         f'{wall[0]:.3f} | {wall[1]:.3f} | {same}')
            lines.append(f'  Pair ratios: {", ".join(f"{ratio:.4f}" for ratio in ratios)}')
        if not workspace:
            batch_ratios = [run[labels[0] + '_prealloc']['event_us'] / run[labels[1] + '_prealloc']['event_us']
                            for run in timings]
            lines.append(f'{n} | preallocated sequential/batch K8 event ratio: {median(batch_ratios):.4f}')
    timeline, errors = timeline_lines(root, tags)
    return '\n'.join([*lines, '', *timeline, *errors]) + '\n'


def report(root, batch=False):
    base = 'tilelang_all_token_l2_default'
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'N | implementation | Cube cores | Pre event us (3 processes) | median us | paired baseline/TL']
    if batch:
        lines[-1] += ' | paired sequential K8/TL'
    for n in ((128, 256) if batch else (128, 256, 512)):
        runs = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        metadata = json.loads((runs[0] / 'metadata.json').read_text())['variants']
        timings = [json.loads((run / 'summary.json').read_text()) for run in runs]
        for label in metadata:
            times = [run[label]['event_us'] for run in timings]
            ratio = median(run[base]['event_us'] / value for run, value in zip(timings, times))
            cores = metadata[label]['projection_config']['num_aic_cores']
            line = (f'{n} | {label} | {cores} | {", ".join(f"{v:.3f}" for v in times)} | '
                    f'{median(times):.3f} | {ratio:.4f}')
            if batch:
                k8_ratio = median(run['tilelang_deferred_k8']['event_us'] / value
                                  for run, value in zip(timings, times))
                line += f' | {k8_ratio:.4f}'
            lines.append(line)
        profiles = [('base', 'mhc_projection_token_pipeline'), ('k8', 'mhc_projection_split_partials')]
        if batch:
            profiles.append(('batch', 'mhc_projection_split_partials'))
        for tag, kernel in profiles:
            lines.extend(profile_lines(root / f'N{n}-profile-{tag}', kernel))
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    root = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else 'scan'
    summary = (allocation_report(root, workspace=mode == 'workspace') if mode in ('alloc', 'workspace')
               else report(root, batch=mode == 'batch'))
    (root / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary, end='')

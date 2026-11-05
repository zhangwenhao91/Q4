"""Compare the all-N mHC Pre projection cache policies on NPU."""
import json
from pathlib import Path
from statistics import median
import sys

from summarize_memory import profile_lines as counter_lines
from summarize_projection import profile_lines as latency_lines


LABELS = ('tilelang_all_token_h5120s3', 'tilelang_all_token_l2_default')


def summarize(folder):
    runs = [folder / f'bench-{i}' for i in (1, 2, 3)]
    reports = [json.loads((run / 'summary.json').read_text()) for run in runs]
    lines = [f'Run: {folder}', f'Q4 revision: {(folder / "revision.txt").read_text().strip()}',
             'N=8192 H=5120 all-N pipeline, full Pre; CPU and pairwise correctness passed in each timed process.',
             'Residual GM-to-UB policy: h5120s3=notalloc_keep; l2_default=no cache annotation.',
             'Full Pre event us | three process medians | median across processes']
    for label in LABELS:
        times = [report[label]['event_us'] for report in reports]
        lines.append(f'{label} | {", ".join(f"{t:.3f}" for t in times)} | {median(times):.3f}')
    for run in runs:
        meta = json.loads((run / 'metadata.json').read_text())
        baseline, candidate = (meta['variants'][label] for label in LABELS)
        hashes = ('standalone_apply_source_sha256', 'tail_device_source_sha256',
                  'projection_device_source_sha256')
        matches = [baseline[key] == candidate[key] for key in hashes]
        lines.append(f'{run.name} generated source equality: apply={matches[0]}, '
                     f'tail={matches[1]}, projection={matches[2]}')
        if matches != [True, True, False]:
            lines.append('Review generated sources before attributing performance to the cache policy.')
    lines.append('Kernel latency uses a separate default profiler capture. L2 counters are raw exporter fields.')
    for label in LABELS:
        lines.extend(latency_lines(folder / f'latency-{label}', 'mhc_projection_token_pipeline'))
        lines.extend(counter_lines(folder / f'L2Cache-{label}', 'L2Cache', stages=('projection',)))
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    report = summarize(folder)
    (folder / 'projection-l2-summary.txt').write_text(report, encoding='utf-8')
    Path('error.txt').write_text(report, encoding='utf-8')

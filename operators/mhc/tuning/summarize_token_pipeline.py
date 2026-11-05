"""Summarize existing token-pipeline runs without executing NPU work."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, help='run directory; default: latest bench run')
    p.add_argument('--output', type=Path, default=root / 'error.txt')
    p.add_argument('--candidate', default='tilelang_flat_h5120s3')
    p.add_argument('--reference', default='ascendc_shifted_native')
    a = p.parse_args()
    runs = list((root / 'results/mhc-token-pipeline').glob('*-bench-*'))
    run = a.input if a.input else max(runs, key=lambda x: x.stat().st_mtime, default=None)
    if run is None:
        p.error('No bench run found. Pass --input to select a smoke or other run.')
    lines = [f'Run: {run}', 'Event time: median across rounds (us); ratios > 1 mean TL is faster.',
             'case | AC us | TL original us | TL N-pipeline us | AC/TL original | AC/TL pipeline | original/pipeline']
    labels = (a.reference, 'tilelang_coeff_bound', a.candidate)
    lines.insert(1, f'AC reference: {a.reference}')
    lines.insert(1, f'Candidate: {a.candidate}')
    count = 0
    for case in sorted(run.glob('N*-rep*')):
        if not case.is_dir():
            continue
        summary_path = case / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if not all(label in summary for label in labels):
            rows = []
            for path in case.glob('*.csv'):
                with path.open(newline='') as f:
                    rows.extend(r for r in csv.DictReader(f) if 'event_us' in r and 'implementation' in r)
            summary = {label: {'event_us': statistics.median(float(r['event_us']) for r in rows
                        if r['implementation'] == label)} for label in labels
                       if any(r['implementation'] == label for r in rows)}
        if all(label in summary for label in labels):
            ac, old, new = (summary[label]['event_us'] for label in labels)
            status = '' if summary_path.exists() else ' [CSV recovered; run may be incomplete]'
            lines.append(f'{case.name} | {ac:.3f} | {old:.3f} | {new:.3f} | {ac/old:.4f} | {ac/new:.4f} | {old/new:.4f}{status}')
            count += 1
        else:
            lines.append(f'{case.name}: timing data incomplete')
    lines.append(f'Cases with timing: {count}')
    log = run / 'runner.log'
    if log.exists():
        tail = log.read_text(errors='replace').splitlines()[-50:]
        lines.extend(['Runner log tail (last 20 lines):', *tail[-20:]])
    result = '\n'.join(lines) + '\n'
    a.output.write_text(result, encoding='utf-8')
    print(result, end='')
    print(f'Saved: {a.output}')


if __name__ == '__main__':
    main()

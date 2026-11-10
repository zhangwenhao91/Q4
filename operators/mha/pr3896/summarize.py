"""Summarize measured AC/TL and old-TL/PR ratios without an NPU."""
import argparse
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--input', required=True, type=Path)
a = p.parse_args()
checks = list(a.input.glob('*/*/correctness.json'))
rows = [r for f in checks for r in json.loads(f.read_text())]
if not rows or not all(r['passed'] for r in rows):
    raise SystemExit('Missing or failed accuracy checks; no speedup reported')
print('Accuracy:', len(rows), 'checks passed')
for length in (4096, 3584):
    for mode in ('cached', 'per_call'):
        runs = [json.loads(f.read_text()) for f in sorted(a.input.glob(f'bench-{mode}-rep*/{length}/summary.json'))]
        if not runs:
            continue
        if len(runs) != 3:
            raise SystemExit('Expected three process summaries')
        labels = ('tilelang_q128', 'pr_static', 'pr_dynamic', 'ascendc')
        med = {label: statistics.median(r[label]['event_us'] for r in runs) for label in labels}
        print('S=', length, 'metadata=', mode, 'median_event_us=', json.dumps(med))
        for label in labels[:-1]:
            paired = [r['ascendc']['event_us'] / r[label]['event_us'] for r in runs]
            print(label, 'AC/TL=', med['ascendc']/med[label], 'paired_ratios=', paired,
                  'old_TL/current=', med['tilelang_q128']/med[label])

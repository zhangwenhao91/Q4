"""Bounded diagnostic report from existing simulation logs; no NPU work."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    a = p.parse_args()
    runs = list((root / 'results/mhc-all-token-simulator').glob('*'))
    run = a.input or max((v for v in runs if v.is_dir()), key=lambda v: v.stat().st_mtime, default=None)
    if run is None:
        p.error('No existing all-token simulation run found')
    before, tail = deque(maxlen=20), deque(maxlen=25)
    context, events = [], []
    counts = Counter()
    following = 0
    found = False
    event_pattern = re.compile(r'Start profiling|ALL TOKEN CHECK|Traceback|Error:|Exception|Check failed|Timeout|timeout|Child process|task failed|parse finished|results saved|Model stopped')
    error_pattern = re.compile(r'\[ERROR\]|Traceback|Error:|Exception|Check failed|Segmentation|core dumped', re.I)
    with (run / 'console.log').open(errors='replace') as f:
        for number, raw in enumerate(f, 1):
            line = raw.rstrip()[:1200]
            item = f'{number}: {line}'
            if error_pattern.search(line):
                key = re.sub(r'\[\d[^]]*\]', '[id]', line)
                counts[key] += 1
                if not found:
                    context.extend(before)
                    following = 30
                    found = True
            if following:
                context.append(item)
                following -= 1
            if event_pattern.search(line):
                events.append(item)
            before.append(item)
            tail.append(item)
    report = [f'Run: {run}', '=== Key events (last 30) ===', *events[-30:],
              '=== First error context ===', *context,
              '=== Repeated errors (top 8) ===',
              *(f'{count} occurrences: {line}' for line, count in counts.most_common(8)),
              '=== Last 25 log lines ===', *tail, '=== Output files ===']
    for name in ('correctness.json', 'metadata.json', 'visualize_data.bin'):
        paths = sorted(run.rglob(name))
        report.extend(str(v.relative_to(run)) for v in paths[:8])
        if name == 'correctness.json':
            for path in paths[:2]:
                report.append(json.dumps(json.loads(path.read_text()), ensure_ascii=False))
        if not paths:
            report.append(f'{name}: MISSING')
    text = '\n'.join(report) + '\n'
    (run / 'diagnostic.txt').write_text(text, encoding='utf-8')
    (root / 'error.txt').write_text(text, encoding='utf-8')
    print(text, end='')
    print(f'Saved: {root / "error.txt"}')


if __name__ == '__main__':
    main()

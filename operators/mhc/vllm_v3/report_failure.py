"""Extract first build failure context from existing logs, without rebuilding."""
import argparse
from collections import deque
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path)
    p.add_argument('--stage', default='build')
    a = p.parse_args()
    log = a.log
    if log is None:
        logs = list((root / 'results/mhc-vllm-v3').glob('*/build.log'))
        if not logs:
            p.error('No build.log found')
        log = max(logs, key=lambda v: v.stat().st_mtime)
    error = re.compile(r'FAILED:|\bfatal error:|\berror:|CMake Error|Traceback \(most recent call last\)|ModuleNotFoundError|HTTP response code said error|SSL certificate|Error in cmake code|No such file or directory|Permission denied', re.I)
    progress = re.compile(r'\[download \d+% complete\]')
    before, tail = deque(maxlen=8), deque(maxlen=20)
    contexts, seen, remaining, groups = [], set(), 0, 0
    with log.open(encoding='utf-8', errors='replace') as f:
        for number, raw in enumerate(f, 1):
            line = raw.rstrip()
            if progress.search(line):
                continue
            item = f'{number}: {line[:1800]}'
            if error.search(line) and remaining == 0 and groups < 4:
                groups += 1
                for entry in before:
                    if entry not in seen:
                        contexts.append(entry)
                        seen.add(entry)
                remaining = 60 if 'FAILED:' in line or 'CMake Error' in line else 20
            if remaining:
                if item not in seen:
                    contexts.append(item)
                    seen.add(item)
                remaining -= 1
            before.append(item)
            tail.append(item)
    report = '\n'.join([f'Failed stage: {a.stage}', f'Log: {log.resolve()}',
        '=== First error contexts (up to 4) ===', *contexts,
        '=== Final non-progress lines ===', *tail]) + '\n'
    (root / 'error.txt').write_text(report, encoding='utf-8')
    (log.parent / 'failure-report.txt').write_text(report, encoding='utf-8')
    print(report, end='')
    print(f'Saved: {root / "error.txt"}')


if __name__ == '__main__':
    main()

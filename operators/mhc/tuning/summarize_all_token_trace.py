"""Report instruction overlap from existing per-core simulator traces."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

from collect_simulator import intersection, merge


STAGES = ('apply', 'projection', 'tail')
PIPES = ('MTE1', 'MTE2', 'VECTOR', 'CUBE', 'FIX', 'MTE3')
PAIRS = (('MTE2', 'VECTOR'), ('VECTOR', 'MTE3'), ('MTE1', 'CUBE'),
         ('MTE2', 'CUBE'), ('CUBE', 'FIX'))


def pipe_for(label, name):
    if label.startswith('RVEC') and name.startswith('RV_'):
        return 'VECTOR'
    if re.match(r'^(?:CUBE|M)(?:_|$)', label) and re.match(r'^(?:MAD|MMA)(?:_|$)', name):
        return 'CUBE'
    for pipe in ('MTE1', 'MTE2', 'MTE3', 'FIX'):
        if re.match(rf'^{pipe}(?:_|$)', label) and name.startswith(('MOV_SRC_TO_DST', 'FIXPIPE')):
            return pipe
    return None


def read_core(file):
    data = json.loads(file.read_text(encoding='utf-8'))
    events = data.get('traceEvents') if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError(f'Unsupported trace schema: {file}')
    labels = {}
    for event in events:
        if event.get('ph') == 'M' and event.get('name') == 'thread_name':
            labels[event.get('pid'), event.get('tid')] = str((event.get('args') or {}).get('name', ''))
    intervals, names = defaultdict(list), defaultdict(Counter)
    for event in events:
        if event.get('ph') != 'X':
            continue
        try:
            start, duration = float(event['ts']), float(event['dur'])
        except (KeyError, TypeError, ValueError):
            continue
        end = start + duration
        if not (math.isfinite(start) and math.isfinite(end) and end > start):
            continue
        key = (event.get('pid'), event.get('tid'))
        label = labels.get(key) or f'pid={key[0]},tid={key[1]}'
        name = str(event.get('name', ''))
        names[label][name] += 1
        pipe = pipe_for(label, name)
        if pipe:
            intervals[pipe].append((start, end))
    return intervals, names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, help='specific run directory')
    p.add_argument('--output', type=Path, default=Path('error.txt'))
    a = p.parse_args()
    runs = sorted(Path('results/mhc-all-token-simulator').glob('*/'))
    run = a.input or next((r for r in reversed(runs) if all(
        any((r / s / 'prof').rglob('core*/trace.json')) for s in STAGES)), None)
    if run is None:
        p.error('No run with per-core traces for all three stages')
    lines = [f'Run: {run.resolve()}', 'Times use the raw trace unit. Busy is the union of recognized instruction intervals.',
             'NA means no recognized intervals on one or both pipes. Positive overlap only establishes concurrent trace intervals within one core file.',
             'stage\tcore\tpipe\tinstructions\tbusy']
    overlaps = ['stage\tcore\tpair\toverlap\t% of first pipe busy\t% of second pipe busy']
    lanes = ['stage\tcore\tlane\ttop instruction names (count)']
    for stage in STAGES:
        files = sorted((run / stage / 'prof').rglob('core*/trace.json'))
        if not files:
            lines.append(f'{stage}\tMISSING per-core trace.json')
        for file in files:
            core = file.parent.name
            intervals, names = read_core(file)
            merged = {pipe: merge(intervals[pipe]) for pipe in PIPES}
            busy = {pipe: sum(end - start for start, end in spans) for pipe, spans in merged.items()}
            for pipe in PIPES:
                if intervals[pipe]:
                    lines.append(f'{stage}\t{core}\t{pipe}\t{len(intervals[pipe])}\t{busy[pipe]:.6g}')
            for first, second in PAIRS:
                prefix = f'{stage}\t{core}\t{first}/{second}'
                if not busy[first] or not busy[second]:
                    overlaps.append(f'{prefix}\tNA\tNA\tNA')
                    continue
                value = intersection(merged[first], merged[second])
                overlaps.append(f'{prefix}\t{value:.6g}\t{100*value/busy[first]:.2f}\t{100*value/busy[second]:.2f}')
            for lane, counts in sorted(names.items(), key=lambda item: -item[1].total())[:12]:
                samples = ', '.join(f'{name} ({count})' for name, count in counts.most_common(4))
                lanes.append(f'{stage}\t{core}\t{lane}\t{samples}')
    report = '\n'.join([*lines, '', *overlaps, '', *lanes]) + '\n'
    a.output.write_text(report, encoding='utf-8')
    print(f'Saved {a.output.resolve()}')


if __name__ == '__main__':
    main()

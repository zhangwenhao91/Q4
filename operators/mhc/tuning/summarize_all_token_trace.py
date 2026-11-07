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
    if re.match(r'^(?:CUBE|M)(?:_|$)', label) and re.match(r'^(?:MMAD|MAD|MMA)(?:_|$)', name):
        return 'CUBE'
    if re.match(r'^MTE1(?:_|$)', label) and name == 'LOAD_2Dv2':
        return 'MTE1'
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
        elif re.match(r'^(?:WAIT_FLAG|WAIT_EVENT)(?:_|$)', name):
            intervals['WAIT'].append((start, end))
    return intervals, names


def core_files(run, stage):
    files = {}
    for file in sorted((run / stage / 'prof').rglob('core*/trace.json')):
        core = file.parent.name
        if core in files:
            raise ValueError(f'Multiple traces for {stage}/{core} in {run}')
        files[core] = file
    return files


def compare(baseline, candidate):
    lines = [f'Baseline: {baseline.resolve()}', f'Candidate: {candidate.resolve()}',
             'Time columns use raw trace units. Percentages divide by the second pipe busy time.',
             'stage\tcore\tpair\tbaseline busy first/second\tcandidate busy first/second\tbaseline overlap\tcandidate overlap\tbaseline %\tcandidate %\tdelta pp']
    for stage in STAGES:
        old_meta = json.loads((baseline / stage / 'source/metadata.json').read_text())
        new_meta = json.loads((candidate / stage / 'source/metadata.json').read_text())
        if (old_meta['variant'] != 'baseline' or new_meta['variant'] != 'candidate'
                or old_meta['tokens'] != new_meta['tokens']
                or old_meta['projection_config'] != new_meta['projection_config']):
            raise ValueError(f'{stage}: expected matching N/config and baseline/candidate variants')
        old_files, new_files = core_files(baseline, stage), core_files(candidate, stage)
        if not old_files or old_files.keys() != new_files.keys():
            raise ValueError(f'{stage}: core sets differ: {list(old_files)} vs {list(new_files)}')
        for core in old_files:
            tables = []
            for file in (old_files[core], new_files[core]):
                intervals, _ = read_core(file)
                tables.append({pipe: merge(intervals[pipe]) for pipe in PIPES})
            for first, second in PAIRS:
                results = []
                for table in tables:
                    a, b = table[first], table[second]
                    busy_a = sum(end - start for start, end in a)
                    busy_b = sum(end - start for start, end in b)
                    overlap = intersection(a, b) if busy_a and busy_b else None
                    results.append((busy_a, busy_b, overlap, None if overlap is None else 100 * overlap / busy_b))
                old, new = results
                if old[2] is None and new[2] is None:
                    continue
                delta = None if old[3] is None or new[3] is None else new[3] - old[3]
                def fmt(value):
                    return 'NA' if value is None else f'{value:.6g}'
                lines.append('\t'.join([stage, core, f'{first}/{second}',
                    f'{fmt(old[0])}/{fmt(old[1])}', f'{fmt(new[0])}/{fmt(new[1])}',
                    *(fmt(value) for value in (old[2], new[2], old[3], new[3], delta))]))
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, help='specific run directory')
    p.add_argument('--baseline', type=Path, help='compare with a baseline run directory')
    p.add_argument('--output', type=Path, default=Path('error.txt'))
    p.add_argument('--stages', nargs='+', choices=STAGES, default=STAGES)
    a = p.parse_args()
    runs = sorted(r for r in Path('results/mhc-all-token-simulator').glob('*/')
                  if not r.name.endswith('-baseline'))
    run = a.input or next((r for r in reversed(runs) if all(
        any((r / s / 'prof').rglob('core*/trace.json')) for s in STAGES)), None)
    if run is None:
        p.error('No run with per-core traces for all three stages')
    if a.baseline:
        a.output.write_text(compare(a.baseline, run), encoding='utf-8')
        print(f'Saved {a.output.resolve()}')
        return
    lines = [f'Run: {run.resolve()}', 'Times use the raw trace unit. Busy is the union of recognized instruction intervals.',
             'NA means no recognized intervals on one or both pipes. Positive overlap only establishes concurrent trace intervals within one core file.',
             'stage\tcore\tpipe\tinstructions\tbusy']
    overlaps = ['stage\tcore\tpair\toverlap\t% of first pipe busy\t% of second pipe busy']
    lanes = ['stage\tcore\tlane\ttop instruction names (count)']
    for stage in a.stages:
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

"""Attribute fused-tail simulator activity to instruction families."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

from summarize_all_token_trace import core_files, merge, pipe_for


FAMILIES = (
    ('divide / reciprocal / sqrt / exp', r'DIV|RECIP|VREC|RSQRT|SQRT|EXP'),
    ('group reduction / lane rearrange', r'VCG(?:ADD|MAX|MIN)|VCADD|VCMAX|VCMIN|VSEL[R]?|GATHER|SHUFFLE|TRANSPOSE|VPERM'),
    ('conversion / packing', r'VCVT|CONVERT|PACK|UNPK|INTLV'),
    ('fused multiply-add', r'VMULA|VMADD|VFM(A|ADD)|FMA'),
    ('vector loads / stores', r'VLD|VST|VSS|LOAD|STORE'),
)


def inspect(file):
    data = json.loads(file.read_text(encoding='utf-8-sig'))
    events = data['traceEvents'] if isinstance(data, dict) else data
    names = {(e.get('pid'), e.get('tid')): str((e.get('args') or {}).get('name', ''))
             for e in events if e.get('ph') == 'M' and e.get('name') == 'thread_name'}
    grouped = defaultdict(list)
    vector = []
    for e in events:
        if e.get('ph') != 'X':
            continue
        name = str(e.get('name', ''))
        lane = names.get((e.get('pid'), e.get('tid')), '')
        if pipe_for(lane, name) != 'VECTOR':
            continue
        try:
            start = float(e['ts'])
            end = start + float(e['dur'])
        except (KeyError, ValueError, TypeError):
            continue
        if not (math.isfinite(start) and math.isfinite(end) and end > start):
            continue
        vector.append((start, end))
        grouped[name].append((start, end))
    if not vector:
        return ['No recognized RVEC instruction intervals. Lanes: ' + ', '.join(sorted(set(names.values()))[:12])]
    busy = sum(end - start for start, end in merge(vector))
    span = max(end for _, end in vector) - min(start for start, _ in vector)
    classes = defaultdict(list)
    for name, intervals in grouped.items():
        family = next((label for label, pattern in FAMILIES if re.search(pattern, name, re.I)), 'other vector')
        classes[family].extend(intervals)

    def duration(intervals):
        return sum(end - start for start, end in merge(intervals))

    lines = [f'Vector span={span:.6g}, busy union={busy:.6g}, busy/span={busy/span:.2%}',
             'family | instruction records | busy union | % of Vector busy']
    for family, intervals in sorted(classes.items(), key=lambda item: -duration(item[1])):
        length = duration(intervals)
        lines.append(f'{family} | {len(intervals)} | {length:.6g} | {100*length/busy:.2f}')
    lines.append('top mnemonic | instruction records | busy union')
    for name, intervals in sorted(grouped.items(), key=lambda item: -duration(item[1]))[:12]:
        lines.append(f'{name} | {len(intervals)} | {duration(intervals):.6g}')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='fused simulator run directory; defaults to the latest with a tail trace')
    parser.add_argument('--output', type=Path, default=Path('error.txt'))
    args = parser.parse_args()
    run = args.input
    if run is None:
        runs = Path('results/mhc-all-token-simulator').glob('*-fused-N*')
        run = next((path for path in sorted(runs, reverse=True) if core_files(path, 'tail')), None)
    if run is None:
        parser.error('No fused tail trace found; run run_all_token_simulator.sh fused first.')
    files = core_files(run, 'tail')
    if not files:
        parser.error(f'No tail core*/trace.json under {run}')
    lines = [f'Run: {run.resolve()}',
             'Times are raw simulator units. Families classify instructions, not complete source phases.',
             'Families can overlap in time; their shares need not add to 100%.']
    for core, file in files.items():
        lines.extend(['', core, *inspect(file)])
    report = '\n'.join(lines) + '\n'
    args.output.write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()

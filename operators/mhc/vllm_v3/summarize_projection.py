"""Report device kernel latency and simulated Cube gaps for N=8192."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tuning'))
from summarize_all_token_trace import core_files, intersection, merge, read_core


PROJECTIONS = {
    'tilelang_coeff_bound': 'tf32_hc_prenorm_gemm_kernel',
    'tilelang_all_token_h5120s3': 'mhc_projection_token_pipeline',
}


def profile_lines(folder, projection=None):
    files = list(folder.rglob('kernel_details.csv'))
    if len(files) != 1:
        return [f'{folder.name}: expected one kernel_details.csv, found {len(files)}']
    with files[0].open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        name = next((c for c in ('Name', 'Op Name', 'Kernel Name') if c in columns), None)
        duration = next((c for c in columns if c.replace(' ', '').lower()
                         in ('duration(us)', 'duration(μs)', 'duration(µs)')), None)
        if not name or not duration:
            return [f'{folder.name}: unrecognized CSV columns {columns}']
        timings = defaultdict(list)
        for row in reader:
            try:
                timings[row[name]].append(float(row[duration]))
            except (ValueError, TypeError, KeyError):
                continue
    lines = [f'{folder.name}: kernel | calls | median us']
    for label, values in sorted(timings.items()):
        lines.append(f'{label} | {len(values)} | {median(values):.4f}')
    if projection is None:
        return lines
    matched = [values for name, values in timings.items() if projection in name]
    if len(matched) != 1 or len(matched[0]) != 30:
        lines.append('Review projection attribution: expected one projection kernel name and 30 recorded calls.')
    return lines


def simulation_lines(run, variant):
    meta = json.loads((run / 'projection/source/metadata.json').read_text())
    if meta['tokens'] != 8192 or meta['variant'] != variant:
        raise ValueError(f'Unexpected simulation shape or variant: {run}')
    files = core_files(run, 'projection')
    lines = [f'{variant} simulator: {run}', f'Projection config: {meta["projection_config"]}']
    for core, file in files.items():
        if '.cubecore' not in core:
            continue
        intervals, names = read_core(file)
        cube, mte1, waits = (merge(intervals[pipe]) for pipe in ('CUBE', 'MTE1', 'WAIT'))
        if not cube:
            lines.append(f'{core}: no recognized Cube intervals; lanes={list(names)[:20]}')
            continue
        gaps = [(end, start) for (_, end), (start, _) in zip(cube, cube[1:]) if start > end]
        gap_time = sum(end - start for start, end in gaps)
        origin = cube[0][0]
        lines.extend([f'{core}: Cube busy={sum(end-start for start,end in cube):.6g}, '
                      f'interior gaps={gap_time:.6g}, gaps with MTE1={intersection(gaps,mte1):.6g}, '
                      f'gaps with observed wait events={intersection(gaps,waits):.6g}',
                      'Largest gaps | start relative to first Cube instruction | gap | MTE1 overlap | observed wait overlap'])
        for start, end in sorted(gaps, key=lambda span: span[1]-span[0], reverse=True)[:5]:
            span = [(start, end)]
            lines.append(f'{start-origin:.6g} | {end-start:.6g} | {intersection(span,mte1):.6g} | {intersection(span,waits):.6g}')
    if not any('.cubecore' in core for core in files):
        lines.append('No Cube core trace found.')
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--candidate', type=Path)
    a = p.parse_args()
    if bool(a.baseline) != bool(a.candidate):
        p.error('Provide both simulation directories')
    lines = [f'N=8192 device results: {a.input}', 'Full Pre event us by process and median:']
    runs = [json.loads((a.input / f'bench-{i}/summary.json').read_text()) for i in (1, 2, 3)]
    for label in ('ascendc_vllm_v3', *PROJECTIONS):
        times = [run[label]['event_us'] for run in runs]
        lines.append(f'{label}: {", ".join(f"{t:.3f}" for t in times)} | median {median(times):.3f}')
    for impl, projection in PROJECTIONS.items():
        lines.extend(profile_lines(a.input / f'profile-{impl}', projection))
    device = '\n'.join(lines) + '\n'
    (a.input / 'device-summary.txt').write_text(device, encoding='utf-8')
    if a.baseline:
        lines.append('Simulator uses raw trace units. Gaps are between Cube instructions on one subcore. Concurrent MTE1 or wait events do not identify the blocking dependency.')
        lines.extend(simulation_lines(a.baseline, 'baseline'))
        lines.extend(simulation_lines(a.candidate, 'candidate'))
    report = '\n'.join(lines) + '\n'
    (a.input / 'projection-summary.txt').write_text(report, encoding='utf-8')
    Path('error.txt').write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()

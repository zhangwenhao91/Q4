"""Summarize N=8192 per-kernel memory counters in their exported units."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import re
from statistics import median


def classify(name):
    if 'mhc_pre_apply' in name:
        return 'apply'
    if 'tf32_hc_prenorm_gemm' in name or 'mhc_projection_token_pipeline' in name:
        return 'projection'
    if 'mhc_coeff_' in name:
        return 'tail'
    return None


def selected(metric, column):
    key = column.lower().replace(' ', '_')
    if metric == 'Memory':
        return bool(re.search(r'(main_mem|l2|gm_to_ub|ub_to_gm).*(bw|bytes)', key))
    return bool(re.search(r'(?:cache|l2).*(?:hit|miss)|(?:hit|miss).*(?:cache|l2)', key))


def profile_lines(folder, metric, stages=('apply', 'projection', 'tail')):
    files = sorted(path for path in folder.rglob('*.csv')
                   if path.name == 'kernel_details.csv' or path.name.startswith(('op_summary', 'l2_cache')))
    # Prefer the operator-level text export. Raw op_summary can be split across files.
    files.sort(key=lambda path: (path.name != 'kernel_details.csv', path))
    examined = []
    for path in files:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            name = next((c for c in ('Name', 'Op Name', 'Kernel Name', 'OpName', 'Task Name') if c in columns), None)
            fields = [c for c in columns if selected(metric, c)]
            examined.append(f'{path.name}: name={name}, matching columns={fields}')
            if not name or not fields:
                continue
            groups = defaultdict(lambda: defaultdict(list))
            for row in reader:
                kernel = row.get(name, '') or ''
                stage = classify(kernel)
                if stage not in stages:
                    continue
                for field in fields:
                    try:
                        value = float(row.get(field, ''))
                    except (ValueError, TypeError):
                        continue
                    if math.isfinite(value) and value >= 0:
                        groups[stage, kernel][field].append(value)
        if not groups:
            continue
        lines = [f'{folder.name} | source={path.relative_to(folder)}',
                 'stage | kernel | counter | median | numeric rows']
        for (stage, kernel), metrics in sorted(groups.items()):
            for field, values in sorted(metrics.items()):
                lines.append(f'{stage} | {kernel} | {field} | {median(values):.6g} | {len(values)}')
        missing = set(stages) - {stage for stage, _ in groups}
        if missing:
            lines.append('No numeric counters attributed to ' + ', '.join(sorted(missing)))
        return lines
    return [f'{folder.name}: no usable per-kernel {metric} fields', *examined[:8]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    a = p.parse_args()
    residual_bytes = 8192 * 4 * 5120 * 2
    lines = [f'Run: {a.input}',
             f'Logical residual bytes per full traversal: {residual_bytes} ({residual_bytes / 2**20:.0f} MiB).',
             'Apply and projection each consume residual; logical tensor size does not measure physical memory traffic.',
             'Counters preserve exporter fields and units. Missing fields do not imply zero traffic or zero hit rate.']
    for metric in ('Memory', 'L2Cache'):
        for impl in ('tilelang_coeff_bound', 'tilelang_all_token_h5120s3'):
            lines.extend(['', *profile_lines(a.input / f'{metric}-{impl}', metric)])
    report = '\n'.join(lines) + '\n'
    (a.input / 'memory-summary.txt').write_text(report, encoding='utf-8')
    Path('error.txt').write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()

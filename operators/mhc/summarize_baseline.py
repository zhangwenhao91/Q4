"""Print a compact offline baseline summary for copying into error.txt (no NPU needed)."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    if not root.is_dir():
        parser.error(f'Result directory missing: {root}')
    print('BASELINE:', root)
    checks = list(root.glob('*/correctness.json'))
    checked = failed = 0
    for file in sorted(checks):
        records = json.loads(file.read_text())
        bad = [r for r in records if not r.get('passed', False)]
        checked += len(records)
        failed += len(bad)
        print('CHECK', file.parent.name, f'{len(records)-len(bad)}/{len(records)} passed')
        for record in bad:
            print('FAILED:', json.dumps(record))
    print('CHECK TOTAL:', checked, 'failed:', failed)
    raw_checks = []
    for file in sorted(root.glob('*/metadata.json')):
        data = json.loads(file.read_text())
        raw_checks.extend(data.get('projection_checks') or [])
    if raw_checks:
        print('RAW PROJECTION CHECKS:', len(raw_checks), 'failed=',
              sum(not r.get('passed', False) for r in raw_checks))
    groups = {}
    for file in sorted(root.glob('bench-*/summary.json')):
        data = json.loads(file.read_text())
        case = file.parent.name.rsplit('-rep', 1)[0]
        print('API PROCESS', file.parent.name, json.dumps(data, sort_keys=True))
        for label, values in data.items():
            if isinstance(values, dict) and 'event_us' in values:
                groups.setdefault((case, label), []).append(values)
    for (case, label), runs in sorted(groups.items()):
        print('API MEDIAN', case, label, 'processes=', len(runs),
              'event_us=', round(statistics.median(r['event_us'] for r in runs), 6),
              'wall_us=', round(statistics.median(r['wall_us'] for r in runs), 6))
    profiles = sorted(p for p in root.glob('profile-*') if p.is_dir())
    for folder in profiles:
        files = sorted(folder.rglob('kernel_details.csv'))
        print('PROFILE', folder.name, 'kernel CSV files=', len(files))
        if len(files) != 1:
            print('REVIEW: expected one CSV per profile; do not combine duplicates.')
        for file in files:
            print('CSV', file.relative_to(root))
            with file.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames or []
                print('COLUMNS', json.dumps(columns))
                name = next((c for c in ['Name', 'Op Name', 'Kernel Name'] if c in columns), None)
                duration = next((c for c in columns if c.replace(' ', '').lower()
                                 in ('duration(us)', 'duration(μs)', 'duration(µs)')), None)
                if name is None or duration is None:
                    print('REVIEW: unrecognized CSV schema; report columns before analysis.')
                    continue
                kernels = {}
                invalid = total = 0
                for row in reader:
                    total += 1
                    try:
                        value = float(row[duration])
                    except (ValueError, TypeError, KeyError):
                        invalid += 1
                        continue
                    kernels.setdefault(row[name], []).append(value)
                print('ROWS', total, 'invalid durations=', invalid, 'expected API calls=30')
                for label, values in sorted(kernels.items()):
                    print('KERNEL', label, 'count=', len(values),
                          'mean_us=', round(statistics.mean(values), 6),
                          'median_us=', round(statistics.median(values), 6),
                          'sum_us=', round(sum(values), 6))
                print('DEVICE summed_duration_us=', round(sum(map(sum, kernels.values())), 6))
                print('NOTE: summed durations are device work, not elapsed latency when kernels overlap.')
    print('FILES: correctness=', len(checks), 'API summaries=', len(list(root.glob('bench-*/summary.json'))),
          'profile directories=', len(profiles))
    matched = (root / 'native-source.json').exists()
    fusion = (root / 'fusion-source.json').exists()
    bound = (root / 'bound-source.json').exists()
    split = (root / 'split-source.json').exists()
    mode_match = re.search(r'-(smoke|check|bench|profile|all)-\d+$', root.name)
    mode = mode_match.group(1) if mode_match else 'all'
    original = any(json.loads(p.read_text()).get('arguments', {}).get('suite') in
                   ('mhc_original', 'mhc_original_fused') for p in root.glob('*/metadata.json'))
    if original:
        expected = dict(smoke=(2, 0, 0), check=(26, 0, 0), bench=(30, 30, 0),
                        profile=(10, 0, 10), all=(66, 30, 10))[mode]
    elif split:
        expected = dict(smoke=(1, 0, 0), check=(16, 0, 0), bench=(15, 15, 0),
                        profile=(15, 0, 15), all=(46, 15, 15))[mode]
    elif bound:
        expected = dict(smoke=(1, 0, 0), check=(12, 0, 0), bench=(9, 9, 0),
                        profile=(9, 0, 9), all=(30, 9, 9))[mode]
    elif fusion:
        expected = dict(smoke=(1, 0, 0), check=(12, 0, 0), bench=(9, 9, 0),
                        profile=(8, 0, 8), all=(29, 9, 8))[mode]
    else:
        expected = dict(smoke=(1, 0, 0), check=(21 if matched else 6, 0, 0), bench=(12, 12, 0),
                        profile=(8, 0, 8), all=(41 if matched else 26, 12, 8))[mode]
    print(f'Expected for {mode} mode: {expected[0]} correctness files, '
          f'{expected[1]} API summaries, {expected[2]} profile directories.')


if __name__ == '__main__':
    main()

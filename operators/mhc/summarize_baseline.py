"""Print a compact offline baseline summary for copying into error.txt (no NPU needed)."""
import argparse
import csv
import json
from pathlib import Path
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
    expected_checks = 41 if (root / 'native-source.json').exists() else 26
    print(f'Expected for all mode: {expected_checks} correctness files, 12 API summaries, 8 profile directories.')


if __name__ == '__main__':
    main()

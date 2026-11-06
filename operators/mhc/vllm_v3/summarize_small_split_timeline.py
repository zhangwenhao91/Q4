"""Read device timestamps already captured by run_small_split_k.sh batch."""
import argparse
import csv
from decimal import Decimal
from pathlib import Path
from statistics import median


def read_timeline(folder, projection):
    paths = sorted(folder.rglob('kernel_details.csv'))
    if len(paths) != 1:
        raise ValueError(f'expected one kernel_details.csv, found {len(paths)}')
    with paths[0].open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream)
        headers = {''.join(key.lower().split()).replace('μ', 'u').replace('µ', 'u'): key
                   for key in (rows.fieldnames or [])}

        def column(*keys):
            return next((headers[key] for key in keys if key in headers), None)

        name = column('name', 'opname', 'kernelname')
        start = column('starttime(us)', 'taskstarttime(us)')
        duration = column('duration(us)', 'taskduration(us)')
        stream_id = column('streamid')
        step_id = column('stepid')
        if not all((name, start, duration)):
            raise ValueError(f'unsupported CSV headers: {rows.fieldnames}')
        stages = ('mhc_pre_apply_flat_pipeline_asc', projection, 'mhc_coeff_token_pipeline')
        events = []
        extras = 0
        for row in rows:
            stage = next((i for i, key in enumerate(stages) if key in row[name]), None)
            if stage is None:
                extras += 1
                continue
            # Preserve fractional microseconds at epoch-scale timestamps.
            ts = Decimal(row[start].strip())
            dt = Decimal(row[duration].strip())
            if not ts.is_finite() or not dt.is_finite() or dt <= 0:
                raise ValueError(f'invalid timestamp or duration at CSV line {rows.line_num}')
            events.append((ts, dt, stage, row.get(stream_id), row.get(step_id)))

    events.sort(key=lambda event: event[0])
    if len(events) != 90 or [event[2] for event in events] != [0, 1, 2] * 30:
        raise ValueError(f'expected 30 ordered Apply/projection/tail triples, found {len(events)} stage records')
    if stream_id and len({event[3] for event in events}) != 1:
        raise ValueError('matched kernels use multiple streams')

    calls = []
    next_gaps = []
    periods = []
    for i in range(30):
        apply, project, tail = events[i * 3 : i * 3 + 3]
        gap_ap = project[0] - apply[0] - apply[1]
        gap_pt = tail[0] - project[0] - project[1]
        span = tail[0] + tail[1] - apply[0]
        calls.append((apply[1], project[1], tail[1], gap_ap, gap_pt, span))
        # The benchmark synchronizes between profiler steps of ten calls.
        if (i + 1) % 10:
            following = events[(i + 1) * 3]
            if step_id and apply[4] and following[4] and apply[4] != following[4]:
                raise ValueError('CSV step boundaries disagree with the ten-call schedule')
            period = following[0] - apply[0]
            next_gaps.append(period - span)
            periods.append(period)
    if min([*next_gaps, *(gap for call in calls for gap in call[3:5])]) < Decimal('-0.002'):
        raise ValueError('device intervals overlap; cannot interpret them as a serial Pre timeline')
    return [*(median(column) for column in zip(*calls)), median(next_gaps), median(periods)], extras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    lines = [f'Run: {args.run}', 'Per-invocation device timeline medians (us), 30 complete calls per profile.',
             'Next gap and period use 27 adjacent calls within profiler steps. Profiles are separate captures.',
             'N | variant | Apply | projection | tail | Apply->projection gap | projection->tail gap | '
             'first start to last end | next-call gap | start-to-start period']
    errors = []
    for n in (128, 256):
        for tag in ('base', 'k8', 'batch'):
            folder = args.run / f'N{n}-profile-{tag}'
            kernel = 'mhc_projection_token_pipeline' if tag == 'base' else 'mhc_projection_split_partials'
            try:
                values, extras = read_timeline(folder, kernel)
                lines.append(f'{n} | {tag} | ' + ' | '.join(f'{value:.3f}' for value in values))
                if extras:
                    lines.append(f'{folder.name}: {extras} other device records; gaps may include other work.')
            except (ValueError, ArithmeticError, OSError) as exc:
                errors.append(f'{folder.name}: {exc}')
    result = '\n'.join([*lines, *errors]) + '\n'
    if args.output:
        args.output.write_text(result, encoding='utf-8')
    print(result, end='')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())

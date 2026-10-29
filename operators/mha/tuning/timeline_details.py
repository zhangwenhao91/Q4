"""Read existing profiler CSVs; report block labels without physical-core assumptions."""
import argparse
import csv
import io
import math
from pathlib import Path
import statistics


def describe(rows, column):
    values=[]
    for row in rows:
        try:
            value=float(row.get(column,''))
        except (ValueError,TypeError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return f'{column}: n={len(values)} min={min(values):.6f} median={statistics.median(values):.6f} max={max(values):.6f}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    a=p.parse_args()
    if not a.input.is_dir():p.error('Collection directory not found')
    lines=['Raw block/sub-block labels are preserved; they do not establish physical core mapping.',
           'Pipe times can overlap; do not add them or infer idle time by subtraction.']
    paths=sorted(path for path in a.input.rglob('*.csv') if path.name in (
        'PipeUtilization.csv','L2Cache.csv','ResourceConflictRatio.csv'))
    if not paths:p.error('No requested profiler CSVs found')
    for path in paths:
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            headers=reader.fieldnames or []
            rows=list(reader)
        if not rows or not {'block_id','sub_block_id'}.issubset(headers):
            raise RuntimeError(f'Unexpected or empty CSV: {path}')
        lines.append('\nFILE '+str(path.relative_to(a.input)))
        labels=sorted(set(row['block_id'] for row in rows),key=lambda x:(len(x),x))
        lines.append(f'rows={len(rows)} distinct block labels={len(labels)} labels={labels}')
        columns=[h for h in headers if h and any(word in h for word in ('time(us)','hit_rate','wait_ratio','cflt_ratio'))]
        for group in sorted(set(row['sub_block_id'] for row in rows)):
            group_rows=[row for row in rows if row['sub_block_id']==group]
            lines.append('GROUP '+group)
            for column in columns:
                result=describe(group_rows,column)
                if result:lines.append(result)
        # Full block distribution for timing. Cache/wait distributions summarized above.
        if path.name=='PipeUtilization.csv':
            selected=['block_id','sub_block_id']+columns
            buffer=io.StringIO()
            writer=csv.DictWriter(buffer,fieldnames=selected,extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
            lines.append(buffer.getvalue())
    print('\n'.join(lines))


if __name__=='__main__':main()

"""Offline compact summary for original MHA3584 BF16 comparison."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from collections import defaultdict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path)
    a=p.parse_args()
    if a.input:
        root=a.input
    else:
        runs=sorted(Path('results/mha-plain3584').glob('*-all-*'))
        if not runs: p.error('No all-mode run; specify --input')
        root=runs[-1]
    print('SELECTED',root)
    checks=[];groups=defaultdict(list)
    for f in sorted(root.glob('*/correctness.json')): checks.extend(json.loads(f.read_text()))
    print('CHECKS',len(checks),'FAILED',sum(not x['passed'] for x in checks))
    for f in sorted(root.glob('*/metadata.json')):
        m=json.loads(f.read_text());print('IDENTITY',f.parent.name,m.get('q4_revision'),m.get('contract'),
            'source',m.get('source',{}).get('revision'))
        break
    for f in sorted(root.glob('bench-*/summary.json')):
        d=json.loads(f.read_text());print('API PROCESS',f.parent.name,json.dumps(d))
        groups[f.parent.name.rsplit('-rep',1)[0]].append(d)
    for mode,runs in groups.items():
        tl=statistics.median(d['tilelang']['event_us'] for d in runs)
        ac=statistics.median(d['ascendc']['event_us'] for d in runs)
        print('API MEDIAN',mode,'processes',len(runs),'TL_us',tl,'AC_us',ac,'AC/TL',ac/tl)
    for folder in sorted(x for x in root.glob('profile-*') if x.is_dir()):
        files=list(folder.rglob('kernel_details.csv'));print('PROFILE',folder.name,'CSV',len(files))
        for file in files:
            durations=defaultdict(list)
            with file.open(encoding='utf-8-sig',newline='') as stream:
                rows=csv.DictReader(stream)
                for row in rows:
                    name=row.get('Name') or row.get('Op Name') or 'UNKNOWN'
                    value=row.get('Duration(us)')
                    if value is None: raise ValueError('Unknown Duration schema: '+str(rows.fieldnames))
                    durations[name].append(float(value))
            for name,values in durations.items():
                print('KERNEL',name,'count',len(values),'mean_us',statistics.mean(values))
    print('Expected all: 16 correctness files, 6 API summaries, 4 profile directories; 48 output checks.')


if __name__=='__main__': main()

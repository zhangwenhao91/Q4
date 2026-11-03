"""Inspect simulator trace schema and per-lane busy intervals offline."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def merge(intervals):
    result = []
    for start,end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end,result[-1][1]))
        else:
            result.append((start,end))
    return result


def intersection(a,b):
    i=j=0
    total=0.
    while i<len(a) and j<len(b):
        total += max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1] < b[j][1]: i+=1
        else: j+=1
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/mhc-simulator'))
    a=p.parse_args()
    runs=sorted(x for x in a.root.glob('*') if x.is_dir() and '-all-' in x.name)
    if not runs: p.error('No all-mode simulator directory')
    run=runs[-1]
    print('SELECTED',run)
    print('Times below retain raw trace units; no conversion or pipeline attribution assumed.')
    for impl in ('original','nested','flat'):
        for file in sorted((run/impl).glob('OPPROF_*/simulator/core*/trace.json')):
            data=json.loads(file.read_text(encoding='utf-8'))
            events=data.get('traceEvents',[]) if isinstance(data,dict) else data
            if not isinstance(events,list):
                print('UNSUPPORTED',file,type(events).__name__);continue
            print('TRACE',file)
            print('SCHEMA',list(data)[:12] if isinstance(data,dict) else 'list')
            print('PHASES',dict(Counter(str(e.get('ph')) for e in events)))
            labels={}
            for e in events:
                if e.get('ph')=='M':
                    labels[str((e.get('pid'),e.get('tid'),e.get('name')))]=e.get('args')
            print('LANE_METADATA',json.dumps(labels,ensure_ascii=False))
            lanes=defaultdict(list);names=defaultdict(Counter);samples=[]
            for e in events:
                if e.get('ph')!='X' or 'ts' not in e or 'dur' not in e: continue
                try: start=float(e['ts']);duration=float(e['dur'])
                except (ValueError,TypeError): continue
                if duration<0: continue
                key=str((e.get('pid'),e.get('tid')))
                lanes[key].append((start,start+duration));names[key][str(e.get('name'))]+=1
                if len(samples)<5: samples.append(e)
            print('EVENT_SAMPLES',json.dumps(samples,ensure_ascii=False))
            merged={key:merge(value) for key,value in lanes.items()}
            for key,intervals in merged.items():
                print('LANE',key,'events',len(lanes[key]),'union_busy',sum(b-a for a,b in intervals),
                      'span',intervals[-1][1]-intervals[0][0] if intervals else 0,
                      'names',json.dumps(names[key].most_common(16),ensure_ascii=False))
            keys=list(merged)
            for i,left in enumerate(keys):
                for right in keys[i+1:]:
                    value=intersection(merged[left],merged[right])
                    if value>0: print('LANE_OVERLAP',left,right,value)
    print('Offline inspection complete. Lane overlaps need schema validation; nested parent spans are not execution proof.')


if __name__=='__main__': main()

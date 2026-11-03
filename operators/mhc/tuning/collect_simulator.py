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
    p.add_argument('--compact',action='store_true',help='Bounded lane summary for terminal copy')
    a=p.parse_args()
    runs=sorted(x for x in a.root.glob('*') if x.is_dir() and '-all-' in x.name)
    if not runs: p.error('No all-mode simulator directory')
    run=runs[-1]
    print('SELECTED',run)
    print('Times below retain raw trace units; no conversion or pipeline attribution assumed.')
    for impl in ('original','nested','flat'):
        files=sorted((run/impl).glob('OPPROF_*/simulator/core*/trace.json'))
        if a.compact:
            print('IMPL',impl,'trace_files',len(files),'showing_first',min(2,len(files)))
            files=files[:2]
        for file in files:
            data=json.loads(file.read_text(encoding='utf-8'))
            events=data.get('traceEvents',[]) if isinstance(data,dict) else data
            if not isinstance(events,list):
                print('UNSUPPORTED',file,type(events).__name__);continue
            if a.compact:
                compact_trace(file,events)
                continue
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


def compact_trace(file,events):
    def short(value,limit=90):
        return str(value).replace('\n',' ')[:limit]
    labels={};lanes=defaultdict(list);names=defaultdict(Counter)
    for e in events:
        key=(e.get('pid'),e.get('tid'))
        if e.get('ph')=='M' and e.get('name') in ('thread_name','process_name'):
            labels[key]=short((e.get('args') or {}).get('name',''))
        if e.get('ph')!='X': continue
        try: start=float(e['ts']);duration=float(e['dur'])
        except (KeyError,ValueError,TypeError): continue
        if duration<0: continue
        lanes[key].append((start,start+duration));names[key][short(e.get('name'),50)]+=1
    merged={k:merge(v) for k,v in lanes.items()}
    keys=sorted(merged,key=lambda k:sum(b-a for a,b in merged[k]),reverse=True)[:8]
    print('CORE',file.parent.name,'events',len(events),'lanes',len(merged),'shown',len(keys),
          'phases',short(dict(Counter(str(e.get('ph')) for e in events))))
    for idx,key in enumerate(keys):
        spans=merged[key];busy=sum(b-a for a,b in spans)
        label=labels.get(key) or labels.get((key[0],None),'')
        print(' L',idx,short(key,30),label,'count',len(lanes[key]),'busy',round(busy,6),
              'span',round(spans[-1][1]-spans[0][0],6),'names',short(names[key].most_common(3),130))
    overlaps=[]
    for i,left in enumerate(keys):
        for j in range(i+1,len(keys)):
            value=intersection(merged[left],merged[keys[j]])
            if value>0: overlaps.append((value,i,j))
    print(' OVERLAP_TOP8',[(i,j,round(v,6)) for v,i,j in sorted(overlaps,reverse=True)[:8]])
    print(' Units=raw_trace; lane IDs are local to this CORE; parent spans may overlap child instructions.')


if __name__=='__main__': main()

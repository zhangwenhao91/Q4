"""Offline useful-matmul FLOP throughput and peak-normalized operator MFU proxy."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def matrix_flops(suite,contract):
    if suite=='mha_stats_split':
        q=contract.get('Q',[])
        if not isinstance(q,list) or len(q)!=2 or q[0] not in (3584,4096):
            raise ValueError('Expected MHA sequence length 3584 or 4096')
        seq_len=q[0]
        expected={'Q':[seq_len,128],'K':[seq_len,128],'V':[seq_len,128],
                  'batch':1,'heads':1,'input_dtype':'bfloat16','output_dtype':'bfloat16',
                  'causal':False,'dropout':0}
        if any(contract.get(k)!=v for k,v in expected.items()):
            raise ValueError('MHA contract differs from the validated BF16 dense core')
        return 4*seq_len*seq_len*128,'bf16'
    if suite=='mhc_shifted_split':
        if any(contract.get(k)!=v for k,v in {'hidden':5120,'hc_mult':4,'flattened':20480,
                                              'channels':24,'hc_shifted':True,'repeat':20}.items()):
            raise ValueError('mHC contract differs from shifted H5120 three-kernel path')
        n=contract.get('N')
        if not isinstance(n,int) or n<=0:raise ValueError('Invalid token count')
        return 2*n*20480*24,'hf32'
    raise ValueError('Unsupported suite')


def throughput(flops,time_us):
    if not math.isfinite(time_us) or time_us<=0:raise ValueError('Invalid latency')
    return flops/time_us/1e6


def peak_for(config,mode,properties):
    if mode not in config:return None
    entry=config[mode]
    value=entry.get('tflops')
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:
        raise ValueError('Peak TFLOPS must be finite and positive')
    if not isinstance(entry.get('source'),str) or not entry['source'].strip():
        raise ValueError('Peak needs an explicit source')
    if entry.get('scope')!='visible_device_dense':raise ValueError('Peak must be dense, for the visible device')
    for key in ('name','cube_core_num'):
        if entry.get(key)!=properties.get(key) or entry.get(key) is None:
            raise ValueError('Peak device identity/core count mismatch: '+key)
    return entry


def compare_sizes(results):
    index={(r['suite'],r['tokens'],r['implementation']):r for r in results}
    comparisons=[]
    for row in results:
        suite,n,label=row['suite'],row['tokens'],row['implementation']
        if (suite=='mha_stats_split' and n!=4096) or (suite=='mhc_shifted_split' and n not in (64,128,256,512,8192)):
            continue
        scaled=index.get((suite,n*7//8,label))
        if scaled is None:continue
        comparisons.append(dict(suite=suite,implementation=label,full_tokens=n,scaled_tokens=n*7//8,
            useful_flops_ratio=scaled['useful_matrix_flops']/row['useful_matrix_flops'],
            latency_ratio=scaled['event_us']/row['event_us'],
            throughput_ratio=scaled['effective_tflops']/row['effective_tflops'],
            note='Different workloads; throughput ratio measures shape efficiency, not same-work speedup.'))
    return comparisons


def attention_comparisons(results):
    """Compare AC/TL only within the same shape and same process replica."""
    index={(r['tokens'],r['implementation']):r for r in results if r['suite']=='mha_stats_split'}
    comparisons=[]
    for (n,label),tl in index.items():
        ac=index.get((n,'ascendc'))
        if label=='ascendc' or ac is None:continue
        if ac['useful_matrix_flops']!=tl['useful_matrix_flops']:
            raise ValueError('Cannot compare attention paths with different useful work')
        ac_reps={r['replica']:r for r in ac['replicas']}
        ratios=[dict(replica=r['replica'],ac_over_tl=ac_reps[r['replica']]['event_us']/r['event_us'])
                for r in tl['replicas']]
        comparisons.append(dict(tokens=n,implementation=label,
            ac_over_tl=ac['event_us']/tl['event_us'],ac_over_tl_wall=ac['wall_us']/tl['wall_us'],
            tl_mfu_percent=tl['matrix_mfu_percent'],ac_mfu_percent=ac['matrix_mfu_percent'],
            paired_replicas=ratios,tl_wins=sum(r['ac_over_tl']>1 for r in ratios),
            note='AC/TL > 1 means TL faster. Ratio of aggregate medians; paired replicas reported separately.'))
    return comparisons


def analyze(folder,peak_config=None,operator='all'):
    folder=Path(folder)
    plan=json.loads((folder/'plan.json').read_text(encoding='utf-8'))
    config=peak_config or {}
    groups={}
    identities=set()
    selected=[case for case in plan['cases'] if operator=='all' or
              case['suite']==('mha_stats_split' if operator=='mha' else 'mhc_shifted_split')]
    for case in selected:
        path=folder/case['directory']
        meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        checks=json.loads((path/'correctness.json').read_text(encoding='utf-8'))
        if not checks or not all(c.get('passed') is True for c in checks):
            raise ValueError('Missing/failed precision checks: '+str(path))
        arguments=meta['arguments']
        if case['suite']=='mha_stats_split':
            if meta['contract']['Q'][0]!=case['tokens'] or arguments.get('mha_seqlen',4096)!=case['tokens']:
                raise ValueError('MHA plan/shape/sequence argument mismatch')
        if arguments['suite']!=case['suite'] or arguments['tokens']!=case['tokens']:
            raise ValueError('Plan and measured contract disagree')
        if meta['implementation_labels']!=case['labels'] or arguments['implementations']!=case['labels']:
            raise ValueError('Implementation selection mismatch')
        if arguments['rounds']!=case['rounds'] or arguments['warmup']!=100 or arguments['iters']!=100:
            raise ValueError('Timing settings mismatch')
        if meta.get('hf32') is not True:raise ValueError('Expected HF32-enabled configuration')
        if case['suite']=='mhc_shifted_split':
            if case['labels']!=['tilelang_coeff_bound']:raise ValueError('Expected bound three-kernel label only')
            if meta['contract']['N']!=case['tokens']:raise ValueError('mHC N mismatch')
            if meta['split_bound_launch']['expected_kernels_per_call']!=3:
                raise ValueError('Expected three-kernel bound mHC path')
            projection=meta.get('projection_checks')
            if not projection or not all(c.get('passed') is True for c in projection):
                raise ValueError('Missing/failed raw projection checks')
        flops,mode=matrix_flops(case['suite'],meta['contract'])
        properties=meta['device_properties']
        identities.add((properties['name'],properties['cube_core_num'],meta['q4_revision']))
        peak=peak_for(config,mode,properties)
        with (path/'timings.csv').open(newline='',encoding='utf-8') as f:
            timings=list(csv.DictReader(f))
        if len(timings)!=case['rounds']*len(case['labels']):raise ValueError('Incomplete timing CSV')
        summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
        for label in case['labels']:
            rows=[r for r in timings if r['implementation']==label]
            if sorted(int(r['round']) for r in rows)!=list(range(case['rounds'])):
                raise ValueError('Missing/duplicate timing rounds')
            for r in rows:
                for field in ('event_us','wall_us'):throughput(flops,float(r[field]))
            for field in ('event_us','wall_us'):
                value=statistics.median(float(r[field]) for r in rows)
                if not math.isclose(value,summary[label][field],rel_tol=1e-9):raise ValueError('Summary mismatch')
            key=(case['suite'],case['tokens'],label)
            group=groups.setdefault(key,dict(suite=case['suite'],tokens=case['tokens'],implementation=label,
                useful_matrix_flops=flops,compute_mode=mode,peak=peak,replicas=[]))
            group['replicas'].append(dict(replica=case['replica'],directory=case['directory'],
                event_us=summary[label]['event_us'],wall_us=summary[label]['wall_us']))
    if len(identities)!=1:raise ValueError('Mixed devices/core counts or Q4 revisions')
    results=[]
    for group in groups.values():
        replicas=group['replicas']
        if sorted(r['replica'] for r in replicas)!=[1,2,3]:raise ValueError('Expected exactly three replicas')
        event=statistics.median(r['event_us'] for r in replicas)
        wall=statistics.median(r['wall_us'] for r in replicas)
        rates=[throughput(group['useful_matrix_flops'],r['event_us']) for r in replicas]
        effective=throughput(group['useful_matrix_flops'],event)
        group.update(event_us=event,wall_us=wall,effective_tflops=effective,
            wall_effective_tflops=throughput(group['useful_matrix_flops'],wall),
            replica_effective_tflops_min=min(rates),replica_effective_tflops_max=max(rates),
            matrix_mfu_percent=100*effective/group['peak']['tflops'] if group['peak'] else None)
        results.append(group)
    return dict(device_revision=list(identities)[0],results=results,size_comparisons=compare_sizes(results),
        mha_comparisons=attention_comparisons(results),
        definition='Useful matmul FLOPs / full API event latency / dense peak for the actual compute mode. Operator MFU proxy, not full-model MFU or hardware issue utilization.',
        omitted_flops='MHA softmax/scaling and mHC square-sum/RMS/apply/coefficient/Sinkhorn scalar-vector operations. Their execution time is INCLUDED. No padding/recompute FLOPs credited.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--peak-json',type=Path)
    p.add_argument('--operator',choices=('all','mha','mhc'),default='all')
    a=p.parse_args()
    peaks=json.loads(a.peak_json.read_text(encoding='utf-8')) if a.peak_json else None
    result=analyze(a.input,peaks,operator=a.operator)
    lines=['# Operator MFU proxy',result['definition'],result['omitted_flops'],
           '\n| Suite | N | Implementation | Event us | Effective TFLOPS | Matrix MFU % |',
           '|---|---:|---|---:|---:|---:|']
    for r in result['results']:
        mfu=f"{r['matrix_mfu_percent']:.4f}" if r['matrix_mfu_percent'] is not None else 'pending peak'
        lines.append(f"| {r['suite']} | {r['tokens']} | {r['implementation']} | {r['event_us']:.6f} | {r['effective_tflops']:.6f} | {mfu} |")
    if result['mha_comparisons']:
        lines.extend(['\n## MHA AC/TL speedup at matching shapes',
            'AC/TL > 1 means TL faster. Wins count paired independent processes, not individual calls.',
            '| S | TL | AC/TL event | AC/TL wall | TL paired wins |',
            '|---:|---|---:|---:|---:|'])
        for r in result['mha_comparisons']:
            lines.append(f"| {r['tokens']} | {r['implementation']} | {r['ac_over_tl']:.6f} | {r['ac_over_tl_wall']:.6f} | {r['tl_wins']}/{len(r['paired_replicas'])} |")
    if result['size_comparisons']:
        lines.extend(['\n## Seven-eighths / full shape efficiency',
            'Different workloads. A lower latency alone is not a same-work speedup.',
            '| Suite | Implementation | Full -> scaled | FLOPs ratio | Latency ratio | TFLOPS ratio |',
            '|---|---|---|---:|---:|---:|'])
        for r in result['size_comparisons']:
            lines.append(f"| {r['suite']} | {r['implementation']} | {r['full_tokens']} -> {r['scaled_tokens']} | {r['useful_flops_ratio']:.6f} | {r['latency_ratio']:.6f} | {r['throughput_ratio']:.6f} |")
    text='\n'.join(lines)+'\n'
    (a.input/'mfu.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (a.input/'mfu.md').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()

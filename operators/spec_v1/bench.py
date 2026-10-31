"""Q4 specification v1: forward correctness, API timing and device profiling."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def implementation_order(labels,round_index):
    order=list(labels)
    if len(order)>2:
        shift=round_index%len(order);order=order[shift:]+order[:shift]
        if (round_index//len(order))%2: order.reverse()
    elif round_index%2:
        order.reverse()
    return order


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['mha','mha_bf16','mha_q64','mha_fma','mha_nostore','mha_rowfused','mha_rowbatch','mha_stages','mha_finalize','mha_qresident','mha_buffers','mha_firstpv','mha_pbuffer','mha_l2','mha_stats_split','mha_manual','mhc_post','mhc_shifted','mhc_shifted_native','mhc_shifted_fusion','mhc_shifted_bound','mhc_shifted_split','mhc_original','mhc_original_fused','mhc_original_big5120'],required=True)
    p.add_argument('--tokens',type=int,default=512,help='mHC only; MHA is fixed at 4096')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--mhc-pattern',choices=['random','zero','prev_zero','prev_basis'],default='random',
                   help='native/fusion shifted mHC correctness inputs; no device work is precomputed')
    p.add_argument('--mhc-weights',type=Path,help='native/fusion shifted mHC: CPU torch file containing FP32 fn/scale/base')
    p.add_argument('--ac-metadata',choices=['cached','per_call'],default='cached',
                   help='mha_bf16: reuse fixed-shape metadata or include native metadata generation each call')
    p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--iters',type=int,default=100)
    p.add_argument('--rounds',type=int,default=9)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--profile',help='exact implementation label printed by check')
    p.add_argument('--profile-pipe-metrics',action='store_true',help='explicit Level1 PipeUtilization counters; requires --profile')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.profile_pipe_metrics and (not a.profile or a.check_only):
        p.error('--profile-pipe-metrics requires --profile and cannot be combined with --check-only')
    if a.tokens<=0 or a.iters<=0 or a.rounds<=0 or a.warmup<0:
        p.error('tokens/iters/rounds > 0; warmup >= 0')
    if a.suite=='mhc_shifted' and a.tokens%128:
        p.error('current tested mHC paths require N divisible by 128')
    if a.suite not in ('mhc_shifted_native','mhc_shifted_fusion','mhc_shifted_bound','mhc_shifted_split') and (a.mhc_weights or a.mhc_pattern!='random'):
        p.error('--mhc-weights/--mhc-pattern require a native/fusion/bound shifted suite')
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    import torch_npu
    from cases import mha_case,post_case,shifted_case,shifted_native_case,shifted_fusion_case,shifted_bound_case,shifted_split_case,ROOT
    torch.manual_seed(a.seed);torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32=True
    with torch.no_grad():
        if a.suite in ('mhc_original','mhc_original_fused','mhc_original_big5120'):
            from mhc_original import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_stats_split':
            from mha_stats_split import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_l2':
            from mha_l2 import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_pbuffer':
            from mha_pbuffer import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_firstpv':
            from mha_firstpv import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_buffers':
            from mha_buffers import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_qresident':
            from mha_qresident import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_finalize':
            from mha_finalize import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_stages':
            from mha_stages import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_rowbatch':
            from mha_rowbatch import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_rowfused':
            from mha_rowfused import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_nostore':
            from mha_nostore import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_manual':
            from mha_manual import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_fma':
            from mha_fma import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_q64':
            from mha_q64 import make_case
            functions,refs,names,meta=make_case(a)
        elif a.suite=='mha_bf16':
            from mha_bf16 import make_case
            functions,refs,names,meta=make_case(a)
        else:
            functions,refs,names,meta={'mha':mha_case,'mhc_post':post_case,'mhc_shifted':shifted_case,
                                     'mhc_shifted_native':shifted_native_case,
                                     'mhc_shifted_fusion':shifted_fusion_case,
                                     'mhc_shifted_bound':shifted_bound_case,
                                     'mhc_shifted_split':shifted_split_case}[a.suite](a)
    meta.update(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                torch=torch.__version__,torch_npu=torch_npu.__version__,device=torch.npu.get_device_name(0),
                q4_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                hf32=True,seed=a.seed,implementation_labels=list(functions),
                input_data='same synthetic CPU-generated inputs for every implementation',
                precision_gate='initial smoke acceptance, not a model-level precision guarantee')
    meta['visible_devices']=os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    meta['custom_opp']=os.environ.get('ASCEND_CUSTOM_OPP_PATH')
    for dist in ['tilelang','tile-kernels','deep-gemm','cann-ops-transformer']:
        try: meta[dist]=importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError: pass
    meta['source_revisions']={}
    for name in ['TileKernels','ops-transformer','DeepGEMM-Ascend']:
        folder=ROOT/'third_party'/name
        if (folder/'.git').exists():
            meta['source_revisions'][name]=dict(
                revision=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip(),
                status=subprocess.check_output(['git','-C',str(folder),'status','--short'],text=True).strip())
    if a.suite in ('mhc_shifted','mhc_shifted_native','mhc_shifted_fusion','mhc_shifted_bound','mhc_shifted_split'):
        import deep_gemm
        file=Path(deep_gemm.__file__).parent/'tilelang_ops'/'hc_prenorm.py'
        meta['hc_source_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    records=[];values={}
    def validate(label,outputs,expected):
        if len(outputs)!=len(expected): raise AssertionError('wrong output count')
        for name,y,r in zip(names,outputs,expected):
            y=y.detach().cpu();r=r.detach().cpu()
            if y.shape!=r.shape or y.dtype!=r.dtype:
                raise AssertionError(f'{label}/{name}: {y.shape}/{y.dtype} != {r.shape}/{r.dtype}')
            delta=y.double()-r.double();err=delta.norm().item()/max(r.double().norm().item(),1e-30)
            finite=bool(torch.isfinite(y).all() and torch.isfinite(r).all())
            rtol,atol=(.02,.02) if r.dtype==torch.bfloat16 else (1e-3,1e-4)
            if a.suite in ('mha','mha_bf16','mha_q64','mha_fma','mha_manual','mha_nostore','mha_rowfused','mha_rowbatch','mha_stages','mha_finalize','mha_qresident','mha_buffers','mha_firstpv','mha_pbuffer','mha_l2','mha_stats_split'):
                passed=finite and err<=.01 and delta.abs().max().item()<=.02
                gate={'relative_l2_max':.01,'max_abs_max':.02}
            else:
                passed=finite and torch.allclose(y,r,rtol=rtol,atol=atol)
                gate={'rtol':rtol,'atol':atol}
            record=dict(implementation=label,output=name,shape=list(y.shape),dtype=str(y.dtype),
                        finite=finite,max_abs=delta.abs().max().item(),rmse=delta.square().mean().sqrt().item(),
                        relative_l2=err,gate=gate,passed=passed)
            records.append(record);print(record,flush=True)
    with torch.no_grad():
        for label,fn in functions.items():
            outputs=tuple(fn());torch.npu.synchronize()
            # CPU copies avoid retaining large NPU outputs during timing.
            values[label]=tuple(y.detach().cpu() for y in outputs)
            validate(label,values[label],refs)
        labels=list(functions)
        if len(labels)==2: validate('pairwise',values[labels[0]],values[labels[1]])
        elif len(labels)>2:
            from itertools import combinations
            for left,right in combinations(labels,2):
                validate(left+'_vs_'+right,values[left],values[right])
    (a.output/'correctness.json').write_text(json.dumps(records,indent=2))
    if not all(r['passed'] for r in records): raise AssertionError('Correctness failed; timing skipped')
    print('Correctness passed. Implementations:',list(functions),flush=True)
    if not meta['comparison_available']:
        (a.output/'comparison_status.json').write_text(json.dumps(dict(status='single_side_only',
            reason=meta['ascendc_status'],speedup=None),indent=2))
        print(meta['ascendc_status'],flush=True)
    del values,refs,outputs
    if a.check_only: return
    with torch.no_grad():
        for fn in functions.values():
            for _ in range(a.warmup): fn()
            torch.npu.synchronize()
        if a.profile:
            if a.profile not in functions: p.error(f'profile implementation must be one of {list(functions)}')
            fn=functions[a.profile]
            from profiling_config import extra_profile_options
            profile_options=extra_profile_options(torch_npu.profiler,a.profile_pipe_metrics)
            print('Profiler metrics:', 'Level1 PipeUtilization' if a.profile_pipe_metrics else 'default',flush=True)
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=1,warmup=1,active=3,repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/'profile')),
                record_shapes=True,profile_memory=False,with_stack=False,**profile_options) as prof:
                for _ in range(5):
                    for _ in range(10): fn()
                    torch.npu.synchronize();prof.step()
            return
        rows=[]
        with (a.output/'timings.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['round','implementation','event_us','wall_us'])
            writer.writeheader()
            for rnd in range(a.rounds):
                order=implementation_order(functions,rnd)
                for label in order:
                    fn=functions[label];torch.npu.synchronize()
                    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    begin=time.perf_counter();start.record()
                    for _ in range(a.iters): fn()
                    end.record();end.synchronize()
                    row=dict(round=rnd,implementation=label,event_us=start.elapsed_time(end)*1000/a.iters,
                             wall_us=(time.perf_counter()-begin)*1e6/a.iters)
                    rows.append(row);writer.writerow(row);f.flush();print(row,flush=True)
        summary={label:{key:statistics.median(r[key] for r in rows if r['implementation']==label)
                       for key in ['event_us','wall_us']} for label in functions}
        if len(functions)==2:
            first,second=list(functions)
            summary['ratio']=dict(numerator=second,denominator=first,
                                  event_ratio=summary[second]['event_us']/summary[first]['event_us'])
        if a.suite=='mhc_original_big5120':
            summary['ac_over_tl']={label:summary['ascendc_original_pre_sinkhorn']['event_us']/summary[label]['event_us']
                                   for label in functions if label.startswith('tilelang_')}
        if a.suite=='mhc_original_fused':
            summary['ac_over_tl']={label:summary['ascendc_original_fused_h4096']['event_us']/summary[label]['event_us']
                                   for label in functions if label.startswith('tilelang_')}
        if a.suite=='mhc_shifted_fusion':
            summary['ratios']={label:dict(
                original_over_candidate=summary['deepgemm_tilelang']['event_us']/summary[label]['event_us'],
                ascendc_over_candidate=summary['ascendc_shifted_native']['event_us']/summary[label]['event_us'])
                for label in ('tilelang_coeff_fused','tilelang_apply_fused')}
        if a.suite=='mhc_shifted_bound':
            summary['ratios']=dict(
                fused_over_bound=summary['tilelang_apply_fused']['event_us']/summary['tilelang_apply_bound']['event_us'],
                ascendc_over_bound=summary['ascendc_shifted_native']['event_us']/summary['tilelang_apply_bound']['event_us'])
        if a.suite=='mhc_shifted_split':
            summary['ratios']=dict(
                two_over_three=summary['tilelang_apply_bound']['event_us']/summary['tilelang_coeff_bound']['event_us'],
                ascendc_over_three=summary['ascendc_shifted_native']['event_us']/summary['tilelang_coeff_bound']['event_us'])
        if a.suite=='mha_q64':
            summary['ratios']=dict(
                q128_over_q64=summary['tilelang_q128']['event_us']/summary['tilelang_q64']['event_us'],
                ascendc_over_q64=summary['ascendc']['event_us']/summary['tilelang_q64']['event_us'])
        if a.suite=='mha_fma':
            summary['ratios']=dict(
                q128_over_fma=summary['tilelang_q128']['event_us']/summary['tilelang_fma']['event_us'],
                ascendc_over_fma=summary['ascendc']['event_us']/summary['tilelang_fma']['event_us'])
        if a.suite=='mha_stats_split':
            summary['ratios']={}
            for candidate in ('tilelang_s2buf','tilelang_alpha2buf'):
                summary['ratios'][candidate]=dict(
                    matched_best_over_candidate=summary['tilelang_kv2']['event_us']/summary[candidate]['event_us'],
                    original_over_candidate=summary['tilelang_q128']['event_us']/summary[candidate]['event_us'],
                    ascendc_over_candidate=summary['ascendc']['event_us']/summary[candidate]['event_us'])
        if a.suite=='mha_l2':
            summary['ratios']={}
            for candidate in ('tilelang_q_l2','tilelang_o_l2','tilelang_qo_l2'):
                summary['ratios'][candidate]=dict(
                    matched_best_over_candidate=summary['tilelang_kv2']['event_us']/summary[candidate]['event_us'],
                    original_over_candidate=summary['tilelang_q128']['event_us']/summary[candidate]['event_us'],
                    ascendc_over_candidate=summary['ascendc']['event_us']/summary[candidate]['event_us'])
        if a.suite=='mha_pbuffer':
            summary['ratios']=dict(
                matched_best_over_pbuffer=summary['tilelang_kv2']['event_us']/summary['tilelang_pbuffer']['event_us'],
                original_over_pbuffer=summary['tilelang_q128']['event_us']/summary['tilelang_pbuffer']['event_us'],
                ascendc_over_pbuffer=summary['ascendc']['event_us']/summary['tilelang_pbuffer']['event_us'])
        if a.suite=='mha_firstpv':
            summary['ratios']=dict(
                matched_best_over_firstpv=summary['tilelang_kv2']['event_us']/summary['tilelang_firstpv']['event_us'],
                original_over_firstpv=summary['tilelang_q128']['event_us']/summary['tilelang_firstpv']['event_us'],
                ascendc_over_firstpv=summary['ascendc']['event_us']/summary['tilelang_firstpv']['event_us'])
        if a.suite=='mha_buffers':
            summary['ratios']={}
            for candidate in ('tilelang_kv2','tilelang_stats2','tilelang_buffers2'):
                summary['ratios'][candidate]=dict(
                    matched_best_over_candidate=summary['tilelang_finalize']['event_us']/summary[candidate]['event_us'],
                    original_over_candidate=summary['tilelang_q128']['event_us']/summary[candidate]['event_us'],
                    ascendc_over_candidate=summary['ascendc']['event_us']/summary[candidate]['event_us'])
        if a.suite=='mha_qresident':
            summary['ratios']=dict(
                matched_best_over_qresident=summary['tilelang_finalize']['event_us']/summary['tilelang_qresident']['event_us'],
                original_over_qresident=summary['tilelang_q128']['event_us']/summary['tilelang_qresident']['event_us'],
                ascendc_over_qresident=summary['ascendc']['event_us']/summary['tilelang_qresident']['event_us'])
        if a.suite=='mha_finalize':
            summary['ratios']=dict(
                matched_best_over_finalize=summary['tilelang_nostore_s2']['event_us']/summary['tilelang_finalize']['event_us'],
                original_over_finalize=summary['tilelang_q128']['event_us']/summary['tilelang_finalize']['event_us'],
                ascendc_over_finalize=summary['ascendc']['event_us']/summary['tilelang_finalize']['event_us'])
        if a.suite=='mha_stages':
            summary['ratios']=dict(
                original_s3_over_s2=summary['tilelang_q128']['event_us']/summary['tilelang_s2']['event_us'],
                nostore_s3_over_s2=summary['tilelang_nostore']['event_us']/summary['tilelang_nostore_s2']['event_us'],
                best_s3_over_original_s2=summary['tilelang_nostore']['event_us']/summary['tilelang_s2']['event_us'],
                ascendc_over_original_s2=summary['ascendc']['event_us']/summary['tilelang_s2']['event_us'],
                ascendc_over_nostore_s2=summary['ascendc']['event_us']/summary['tilelang_nostore_s2']['event_us'])
        if a.suite=='mha_rowbatch':
            summary['ratios']=dict(
                q128_over_rowbatch=summary['tilelang_q128']['event_us']/summary['tilelang_rowbatch']['event_us'],
                ascendc_over_rowbatch=summary['ascendc']['event_us']/summary['tilelang_rowbatch']['event_us'])
        if a.suite=='mha_rowfused':
            summary['ratios']=dict(
                q128_over_rowfused=summary['tilelang_q128']['event_us']/summary['tilelang_rowfused']['event_us'],
                ascendc_over_rowfused=summary['ascendc']['event_us']/summary['tilelang_rowfused']['event_us'])
        if a.suite=='mha_nostore':
            summary['ratios']=dict(
                q128_over_nostore=summary['tilelang_q128']['event_us']/summary['tilelang_nostore']['event_us'],
                ascendc_over_nostore=summary['ascendc']['event_us']/summary['tilelang_nostore']['event_us'])
        if a.suite=='mha_manual':
            summary['ratios']=dict(
                q128_over_manual=summary['tilelang_q128']['event_us']/summary['tilelang_manual']['event_us'],
                ascendc_over_manual=summary['ascendc']['event_us']/summary['tilelang_manual']['event_us'])
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)


if __name__=='__main__': main()

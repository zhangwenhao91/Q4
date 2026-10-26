"""Backward validation, API timing and profiler driver. --help needs no torch."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['post','sinkhorn','pre','pre_sinkhorn'],required=True)
    p.add_argument('--mode',choices=['backward','train'],default='backward')
    p.add_argument('--tokens',type=int,default=512)
    p.add_argument('--hidden',type=int,default=4096)
    p.add_argument('--repeat',type=int,default=20)
    p.add_argument('--eps',type=float,default=1e-6)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--iters',type=int,default=100)
    p.add_argument('--rounds',type=int,default=9)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--profile',choices=['tilelang','ascendc'])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.tokens,a.hidden,a.repeat,a.iters,a.rounds)<=0 or a.warmup<0 or a.eps<=0:
        p.error('sizes/repeat/iters/rounds/eps must be positive; warmup >= 0')
    if a.tokens%128 or a.hidden%128:
        p.error('this comparison uses tokens and hidden divisible by 128')
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    import torch_npu
    from cases import make_case
    torch.manual_seed(a.seed)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32=True
    tl,ac,refs,names,label=make_case(a,a.tokens,a.hidden)
    meta=vars(a).copy();meta['output']=str(a.output)
    meta.update(torch=torch.__version__,torch_npu=torch_npu.__version__,python=platform.python_version(),
                device=torch.npu.get_device_name(0),tilelang_label=label,hf32=True,
                timing='API stream event and synchronized host wall; allocations included',
                reference='CPU FP32 autograd with BF16 input/output casts',
                pre_optional_inputs='no gamma or external grad_x_post')
    meta['q4_revision']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for package in ['tilelang','tile-kernels','deep-gemm','cann-ops-transformer']:
        try: meta[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    if a.suite.startswith('pre'):
        import deep_gemm
        src=Path(deep_gemm.__file__).parent/'tilelang_ops'/'hc_prenorm.py'
        if src.exists(): meta['hc_prenorm_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    records=[]
    def validate(values,reference,impl):
        if len(values)!=len(reference): raise AssertionError('wrong number of gradients')
        for name,value,ref in zip(names,values,reference):
            if value.shape!=ref.shape: raise AssertionError(f'{impl}/{name}: {value.shape} != {ref.shape}')
            v=value.detach().cpu();r=ref.detach().cpu()
            delta=v.double()-r.double()
            bf16=r.dtype==torch.bfloat16
            rtol,atol=(.02,.02) if bf16 else (1e-3,1e-4)
            ok=bool(torch.isfinite(v).all() and torch.isfinite(r).all()) and torch.allclose(v.float(),r.float(),rtol=rtol,atol=atol)
            records.append(dict(implementation=impl,gradient=name,shape=list(v.shape),dtype=str(v.dtype),
                rtol=rtol,atol=atol,max_abs=delta.abs().max().item(),rmse=delta.square().mean().sqrt().item(),
                passed=ok))
    with torch.no_grad():
        tv=tuple(tl());av=tuple(ac());torch.npu.synchronize()
        validate(tv,refs,label);validate(av,refs,'ascendc');validate(tv,av,'pairwise')
        # A second invocation catches accidental accumulation into reused gradient buffers.
        validate(tuple(tl()),refs,label+'_second_call')
        validate(tuple(ac()),refs,'ascendc_second_call')
    (a.output/'correctness.json').write_text(json.dumps(records,indent=2))
    for r in records: print(r,flush=True)
    if not all(r['passed'] for r in records): raise AssertionError('gradient check failed; timing skipped')
    del tv,av,refs
    if a.check_only: return
    funcs={'tilelang':tl,'ascendc':ac}
    with torch.no_grad():
        for fn in funcs.values():
            for _ in range(a.warmup): fn()
            torch.npu.synchronize()
        if a.profile:
            fn=funcs[a.profile]
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=1,warmup=1,active=3,repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/'profile')),
                record_shapes=True,profile_memory=False,with_stack=False) as prof:
                for _ in range(5):
                    for _ in range(10): fn()
                    torch.npu.synchronize();prof.step()
            return
        rows=[]
        with (a.output/'timings.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['round','implementation','event_us','wall_us'])
            writer.writeheader()
            for rnd in range(a.rounds):
                order=['tilelang','ascendc'] if rnd%2==0 else ['ascendc','tilelang']
                for impl in order:
                    fn=funcs[impl];torch.npu.synchronize()
                    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    wall=time.perf_counter();start.record()
                    for _ in range(a.iters): fn()
                    end.record();end.synchronize()
                    row=dict(round=rnd,implementation=impl,event_us=start.elapsed_time(end)*1000/a.iters,
                             wall_us=(time.perf_counter()-wall)*1e6/a.iters)
                    writer.writerow(row);f.flush();rows.append(row);print(row,flush=True)
        summary={impl:{key:statistics.median(r[key] for r in rows if r['implementation']==impl)
                       for key in ['event_us','wall_us']} for impl in funcs}
        summary['ascendc_over_tilelang']=summary['ascendc']['event_us']/summary['tilelang']['event_us']
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)


if __name__=='__main__': main()

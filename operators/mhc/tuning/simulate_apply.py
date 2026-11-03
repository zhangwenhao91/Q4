"""One compiled apply launch for msprof op simulator; no timing or warmup loops."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--impl',choices=['original','nested','flat','token'],required=True)
    p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.tokens <= 0:
        p.error('tokens must be positive')
    import torch
    import torch_npu
    from tile_kernels.config import get_num_vec_cores
    torch.manual_seed(42)
    torch.npu.set_device(0)
    x_cpu = torch.randn(a.tokens,4,5120,dtype=torch.bfloat16)
    mix_cpu = torch.rand(a.tokens,4,dtype=torch.float32)+1e-6
    x, mix = x_cpu.npu(), mix_cpu.npu()
    out = torch.empty(a.tokens,5120,dtype=torch.bfloat16,device=x.device)
    if a.impl == 'original':
        from tile_kernels.mhc.pre_apply_mix_asc import mhc_pre_apply_mix_fwd_asc as fn
        config = dict(h_blk=5120,num_vec_cores=get_num_vec_cores())
    else:
        if a.impl == 'nested':
            from apply_pipeline import mhc_pre_apply_pipeline_asc as fn
        else:
            from apply_flat_pipeline import mhc_pre_apply_flat_pipeline_asc as fn
        config = dict(h_blk=5120 if a.impl == 'token' else 1280,
                      num_vec_cores=get_num_vec_cores(),pipeline_stages=3)
    a.output.mkdir(parents=True,exist_ok=True)
    kernel = fn.compile(x,mix,out,**config)
    source = kernel.get_kernel_source()
    (a.output/'kernel.cpp').write_text(source,encoding='utf-8')
    metadata = dict(implementation=a.impl,tokens=a.tokens,hidden=5120,config=config,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),target_launches=1,
        note='Only apply; input transfers are outside target kernel, no projection or tail')
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    torch.npu.synchronize()
    print('Launching target once:',metadata,flush=True)
    with torch.no_grad():
        kernel(x,mix,out)
    torch.npu.synchronize()
    actual = out.cpu()
    expected = (x_cpu.float()*mix_cpu.unsqueeze(-1)).sum(1).bfloat16()
    passed = bool(torch.isfinite(actual).all() and torch.allclose(actual,expected,rtol=.02,atol=.02))
    check = dict(passed=passed,max_abs=(actual.float()-expected.float()).abs().max().item())
    (a.output/'correctness.json').write_text(json.dumps(check),encoding='utf-8')
    print('APPLY CHECK:',check,flush=True)
    if not passed:
        raise AssertionError('Apply simulator output failed CPU reference')


if __name__ == '__main__':
    main()

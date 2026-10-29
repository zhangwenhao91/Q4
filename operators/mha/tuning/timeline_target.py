"""Run exactly one selected attention implementation inside a profiling range."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

IMPLEMENTATIONS = ('tilelang_s2buf', 'tilelang_kv2', 'ascendc')
RANGE_NAME = 'Q4_TIMELINE'


def capture_one(fn, mstx, synchronize, stream):
    range_id = mstx.range_start(RANGE_NAME, stream)
    if not isinstance(range_id, int) or range_id <= 0:
        raise RuntimeError('MSTX range_start failed; refusing an unscoped collection')
    try:
        output = fn()
        synchronize()
        return output
    finally:
        mstx.range_end(range_id)


def capture_kernel(fn, synchronize):
    """One application launch; name filtering and replay belong to msprof op."""
    output = fn()
    synchronize()
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', choices=IMPLEMENTATIONS, required=True)
    p.add_argument('--mode', choices=('validate', 'capture'), default='capture')
    p.add_argument('--selection', choices=('mstx', 'kernel'), default='mstx')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=100)
    a = p.parse_args()
    if a.selection=='kernel' and a.implementation=='ascendc':
        p.error('Kernel-only capture currently supports TileLang only; FlashAttnMetadata matching must be resolved first')
    if a.warmup < 0:
        p.error('--warmup must be nonnegative')
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    import tilelang
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root/'operators/spec_v1'))
    from mha_bf16 import make_case
    from stats_split_variant import transform
    torch.manual_seed(42)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = True

    def selected_builder(source, q, k, v):
        data = transform((Path(source)/'core.py').read_bytes(), a.implementation)
        file = (a.output/'selected_core.py').resolve()
        file.write_bytes(data)
        spec = importlib.util.spec_from_file_location('q4_profile_'+a.implementation, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        program = module.flash_attention_fwd(
            q_len=4096, kv_len=4096, head_dim=128, out_dtype='bfloat16',
            softmax_scale=1/math.sqrt(128), manual_schedule=False,
            tiling=module.FwdTiling(block_q=128, block_kv=128, num_stages=2))
        kernel = tilelang.compile(program, out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
        cpp = kernel.get_kernel_source()
        if not isinstance(cpp, str) or not cpp.strip():
            raise RuntimeError('Empty compiled device source')
        (a.output/'selected_kernel.cpp').write_text(cpp, encoding='utf-8')
        return {a.implementation: lambda: (kernel(q,k,v),)}, dict(
            python_sha256=hashlib.sha256(data).hexdigest(),
            cpp_sha256=hashlib.sha256(cpp.encode()).hexdigest())

    with torch.no_grad():
        # Setup may compile an unused baseline, but that baseline is never executed.
        functions, refs, names, meta = make_case(
            SimpleNamespace(ac_metadata='cached'),
            extra_builder=None if a.implementation=='ascendc' else selected_builder)
        fn = functions[a.implementation]
        if a.mode=='validate' or a.selection=='mstx':
            outputs = tuple(fn())
            torch.npu.synchronize()
            checks = []
            for name, output, ref in zip(names, outputs, refs):
                output = output.detach().cpu()
                ref = ref.detach().cpu()
                delta = output.double()-ref.double()
                err = delta.norm().item()/max(ref.double().norm().item(),1e-30)
                max_abs = delta.abs().max().item()
                passed = (output.shape==ref.shape and output.dtype==ref.dtype
                          and bool(torch.isfinite(output).all()) and bool(torch.isfinite(ref).all())
                          and err<=.01 and max_abs<=.02)
                checks.append(dict(output=name, relative_l2=err, max_abs=max_abs, passed=passed))
            passed = (len(outputs)==len(refs)==len(names) and bool(checks)
                      and all(check['passed'] for check in checks))
            (a.output/'correctness.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
            print('TARGET CHECK:', json.dumps(dict(implementation=a.implementation, checks=checks)), flush=True)
            if not passed:
                raise AssertionError('Target precision check failed; no MSTX range opened')
            del outputs, refs
            for _ in range(a.warmup):
                fn()
            torch.npu.synchronize()
        props = torch_npu.npu.get_device_properties(0)
        meta.update(implementation=a.implementation, range_name=RANGE_NAME, seed=42,
                    q4_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                    torch=torch.__version__,torch_npu=torch_npu.__version__,
                    warmup=(a.warmup if a.mode=='validate' or a.selection=='mstx' else 0),
                    selection=a.selection, mode=a.mode, capture_succeeded=False,
                    device=dict(name=props.name, cube_core_num=props.cube_core_num,
                                vector_core_num=props.vector_core_num),
                    scope='Only selected attention call inside MSTX. Metadata cached. Setup, validation and warmup outside range. Profiling/replay data is diagnostic, not ordinary API latency.')
        metadata_file = a.output/'metadata.json'
        metadata_file.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
        if a.mode=='capture' and a.selection=='kernel':
            meta['scope']='One selected TileLang call; no app warmup or correctness kernels. Collector filters main_kernel and performs 50 replay warmups. Setup and cached metadata occur before the call. Diagnostic data only.'
            metadata_file.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
            print('TARGET KERNEL START:',a.implementation,flush=True)
            capture_kernel(fn, torch.npu.synchronize)
            meta['capture_succeeded']=True
            metadata_file.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
            print('TARGET KERNEL END:',a.implementation,flush=True)
        elif a.mode=='capture':
            from native_mstx import load_native_mstx
            mstx=load_native_mstx()
            meta['mstx_backend']=mstx.description
            metadata_file.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
            print('MSTX BACKEND:',json.dumps(mstx.description),flush=True)
            print('TARGET RANGE START:', a.implementation, flush=True)
            capture_one(fn, mstx, torch.npu.synchronize, None)
            meta['capture_succeeded'] = True
            metadata_file.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
            print('TARGET RANGE END:', a.implementation, flush=True)
        else:
            print('Target validated and warmed; no profiling range opened.', flush=True)


if __name__=='__main__':
    main()

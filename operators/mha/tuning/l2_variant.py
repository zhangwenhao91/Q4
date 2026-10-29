"""Independent Q/O L2 no-allocation hints on the verified KV2 best kernel."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

from buffers_variant import transform as buffer_transform
from nostore_variant import ORIGINAL_SHA256

Q_LOAD = 'T.copy(Q[mt * BR : (mt + 1) * BR, 0:D], Q_shared[0:BR, 0:D])'
O_STORE = 'T.dual_copy(O_cast_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])'
CONFIGS = {
    'tilelang_kv2': (),
    'tilelang_q_l2': ('Q',),
    'tilelang_o_l2': ('O',),
    'tilelang_qo_l2': ('Q', 'O'),
}


def transform(original, label):
    if label not in CONFIGS:
        raise ValueError('Unknown cache configuration: '+label)
    text = buffer_transform(original, 'tilelang_kv2').decode('utf-8')
    for name,old,count in [('Q',Q_LOAD,2), ('O',O_STORE,1)]:
        if name in CONFIGS[label]:
            if text.count(old) != count:
                raise RuntimeError('Unexpected pinned copy: '+name)
            text = text.replace(old,old[:-1]+', l2_cache_ctrl=4)')
    return text.encode('utf-8')


def build_candidate(source,q,k,v):
    import tilelang
    original = (Path(source)/'core.py').read_bytes()
    functions = {}
    details = {}
    for label,changed in CONFIGS.items():
        data = transform(original,label)
        target = Path(__file__).resolve().parents[3]/'third_party'/'mha-l2-candidate'/(label+'.py')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f'Candidate source modified; preserve it before proceeding: {target}')
        target.write_bytes(data)
        spec = importlib.util.spec_from_file_location('q4_'+label+'_cache',target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        program = module.flash_attention_fwd(
            q_len=4096,kv_len=4096,head_dim=128,out_dtype='bfloat16',
            softmax_scale=1/math.sqrt(128),manual_schedule=False,
            tiling=module.FwdTiling(block_q=128,block_kv=128,num_stages=2))
        kernel = tilelang.compile(program,out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
        functions[label] = lambda kernel=kernel:(kernel(q,k,v),)
        cpp = kernel.get_kernel_source()
        if not isinstance(cpp,str) or not cpp.strip():
            raise RuntimeError('Empty compiled source: '+label)
        cpp_path = target.with_suffix('.cpp')
        cpp_path.write_text(cpp,encoding='utf-8')
        for line_no,line in enumerate(cpp.splitlines(),1):
            if any(op in line for op in ('asc_copy_gm2l1','asc_copy_ub2gm')):
                print(f'CODEGEN {label}:{line_no}: {line.strip()}',flush=True)
        details[label] = dict(
            changed_to_mode4=list(changed),sha256=hashlib.sha256(data).hexdigest(),
            generated_source=str(target),compiled_source=str(cpp_path),
            cpp_sha256=hashlib.sha256(cpp.encode()).hexdigest())
    return functions,dict(
        candidate='Q/O independent mode4 no-allocation hints',original_sha256=ORIGINAL_SHA256,
        sources=details,
        scope='Matched best KV2 arithmetic, buffer versions and pipeline. Only Q GM-to-L1 and/or BF16 O UB-to-GM cache hints change. K/V unchanged. Mode4 differs by load/store direction; do not interpret as mandatory HBM access or proof that all L2 hits are bypassed.',
        workload='Same input addresses reused during warmup/timing, as in all previous suites; not a cold-cache benchmark.')

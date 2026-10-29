"""Initialize O directly from the first PV tile, on top of the best KV2 kernel."""
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import sys

from buffers_variant import transform as buffer_transform
from nostore_variant import ORIGINAL_SHA256

PV = '''    @T.macro
    def pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub):
        T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_shared[0:D, 0:BC], transpose=True)
        T.copy(P_shared[0:BR, 0:BC], pv_a_l0)
        T.copy(V_shared[0:D, 0:BC], pv_b_l0)
        T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
        T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D])

'''
FIRST_PV = PV.replace('def pv(', 'def pv_first(').replace(
    'pv_acc_l0c, O_tmp_ub):', 'pv_acc_l0c, O_tmp_ub, O_ub):').replace(
    '        T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D])',
    '''        if kv == 0:
            T.dual_copy(pv_acc_l0c, O_ub[0:ROWS, 0:D])
        else:
            T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D])''')
OLD_AUTO = '''        pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)
        accumulate_output(O_tmp_ub, O_ub, alpha_ub)'''
NEW_AUTO = '''        pv_first(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub, O_ub)
        if kv > 0:
            accumulate_output(O_tmp_ub, O_ub, alpha_ub)'''


def best_transform(original):
    return buffer_transform(original, 'tilelang_kv2')


def transform(original):
    text = best_transform(original).decode('utf-8')
    if text.count(PV) != 1 or text.count(OLD_AUTO) != 1:
        raise RuntimeError('Unexpected pinned PV implementation')
    text = text.replace(PV, PV+FIRST_PV).replace(OLD_AUTO, NEW_AUTO)
    # Manual path still needs its original clear; constant False removes this for auto.
    text, count = re.subn(
        r'(?m)^([ ]+)clear_O_ub\(O_ub\)$',
        lambda m: m[1]+'if manual_schedule:\n'+m[1]+'    clear_O_ub(O_ub)', text)
    if count != 2:
        raise RuntimeError('Unexpected output initialization structure')
    return text.encode('utf-8')


def build_candidate(source, q, k, v):
    import tilelang
    original = (Path(source)/'core.py').read_bytes()
    functions = {}
    details = {}
    for label, data in [('tilelang_kv2', best_transform(original)),
                        ('tilelang_firstpv', transform(original))]:
        target = Path(__file__).resolve().parents[3]/'third_party'/'mha-firstpv-candidate'/(label+'.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f'Candidate source modified; preserve it before proceeding: {target}')
        target.write_bytes(data)
        spec = importlib.util.spec_from_file_location('q4_'+label+'_firstpv', target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        program = module.flash_attention_fwd(
            q_len=4096, kv_len=4096, head_dim=128, out_dtype='bfloat16',
            softmax_scale=1/math.sqrt(128), manual_schedule=False,
            tiling=module.FwdTiling(block_q=128, block_kv=128, num_stages=2))
        kernel = tilelang.compile(program, out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
        functions[label] = lambda kernel=kernel: (kernel(q,k,v),)
        cpp = kernel.get_kernel_source()
        if not isinstance(cpp, str) or not cpp.strip():
            raise RuntimeError('Empty compiled source: '+label)
        cpp_path = target.with_suffix('.cpp')
        cpp_path.write_text(cpp, encoding='utf-8')
        for line_no, line in enumerate(cpp.splitlines(), 1):
            if ('for (' in line and 'kv' in line) or 'asc_copy_l0c2ub' in line:
                print(f'CODEGEN {label}:{line_no}: {line.strip()}', flush=True)
        details[label] = dict(
            sha256=hashlib.sha256(data).hexdigest(), generated_source=str(target),
            compiled_source=str(cpp_path), cpp_sha256=hashlib.sha256(cpp.encode()).hexdigest())
    return functions, dict(
        candidate='first PV directly initializes O; later blocks rescale/accumulate',
        original_sha256=ORIGINAL_SHA256, sources=details,
        scope='Matched nostore+stages2+fused finalization+KV2, S/alpha versions3. First PV bypasses O_tmp and accumulation; remove auto O clear. Arithmetic of later blocks and softmax retained. Conditional copies may alter scheduling; NPU validation required.')

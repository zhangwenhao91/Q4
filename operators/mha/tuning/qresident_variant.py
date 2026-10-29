"""Keep Q in L0A across KV iterations, on top of the validated best kernel."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

from finalize_variant import transform as best_transform
from nostore_variant import ORIGINAL_SHA256

Q_COPY = 'T.copy(Q_shared[0:BR, 0:D], qk_a_l0)'
Q_LOAD = 'T.copy(Q[mt * BR : (mt + 1) * BR, 0:D], Q_shared[0:BR, 0:D])'
OLD_QK = '''    def qk(kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub):
        T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_shared[0:BC, 0:D])
        T.copy(Q_shared[0:BR, 0:D], qk_a_l0)
'''
NEW_QK = OLD_QK.replace('        '+Q_COPY+'\n', '')


def transform(original):
    text = best_transform(original).decode('utf-8')
    if text.count(OLD_QK) != 1 or text.count(Q_LOAD) != 2:
        raise RuntimeError('Unexpected pinned Q load/copy structure')
    text = text.replace(OLD_QK, NEW_QK)
    lines = []
    for line in text.splitlines(keepends=True):
        lines.append(line)
        if line.strip() == Q_LOAD:
            indent = line[:len(line)-len(line.lstrip())]
            lines.append(indent+Q_COPY+'\n')
    return ''.join(lines).encode('utf-8')


def build_candidate(source, q, k, v):
    import tilelang
    original = (Path(source)/'core.py').read_bytes()
    functions = {}
    details = {}
    for label, data in [('tilelang_finalize', best_transform(original)),
                        ('tilelang_qresident', transform(original))]:
        target = Path(__file__).resolve().parents[3]/'third_party'/'mha-qresident-candidate'/(label+'.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f'Candidate source modified; preserve it before proceeding: {target}')
        target.write_bytes(data)
        spec = importlib.util.spec_from_file_location('q4_'+label+'_qresident', target)
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
        # These lines assist codegen review; they are not a proof of instruction counts.
        for line_no, line in enumerate(cpp.splitlines(), 1):
            if 'asc_copy_l12l0a' in line or ('for (' in line and 'kv' in line):
                print(f'CODEGEN {label}:{line_no}: {line.strip()}', flush=True)
        details[label] = dict(
            sha256=hashlib.sha256(data).hexdigest(), generated_source=str(target),
            compiled_source=str(cpp_path), cpp_sha256=hashlib.sha256(cpp.encode()).hexdigest())
    return functions, dict(
        candidate='Q L1-to-L0A copy before automatic KV loop',
        original_sha256=ORIGINAL_SHA256, sources=details,
        scope='Matched nostore+stages2+fused finalization. Only move automatic Q copy to immediately after Q GM-to-L1 load, once per Q tile. Q/P use separate declared L0A buffers. Compiler resource placement/synchronization and numerical correctness require NPU validation. Manual scheduling not tested.')

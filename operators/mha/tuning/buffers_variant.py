"""Factorial buffer-version experiment on the validated best stages2 kernel."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

from finalize_variant import transform as best_transform
from nostore_variant import ORIGINAL_SHA256

CONFIGS = {
    'tilelang_finalize': (),
    'tilelang_kv2': ('K_shared', 'V_shared'),
    'tilelang_stats2': ('S_ub', 'alpha_ub'),
    'tilelang_buffers2': ('K_shared', 'V_shared', 'S_ub', 'alpha_ub'),
}


def transform(original, label):
    if label not in CONFIGS:
        raise ValueError('Unknown buffer configuration: '+label)
    text = best_transform(original).decode('utf-8')
    for name in CONFIGS[label]:
        old = name+': 3,'
        if text.count(old) != 2:
            raise RuntimeError('Unexpected pinned buffer versions: '+name)
        text = text.replace(old, name+': 2,')
    return text.encode('utf-8')


def build_candidate(source, q, k, v):
    import tilelang
    original = (Path(source)/'core.py').read_bytes()
    functions = {}
    details = {}
    for label, changed in CONFIGS.items():
        data = transform(original, label)
        target = Path(__file__).resolve().parents[3]/'third_party'/'mha-buffers-candidate'/(label+'.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f'Candidate source modified; preserve it before proceeding: {target}')
        target.write_bytes(data)
        spec = importlib.util.spec_from_file_location('q4_'+label+'_buffers', target)
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
        # Print the actual emitted loop bounds as a small diagnostic, not a latency model.
        for line_no, line in enumerate(cpp.splitlines(), 1):
            if 'for (' in line and 'kv' in line:
                print(f'CODEGEN {label}:{line_no}: {line.strip()}', flush=True)
        details[label] = dict(
            changed_to_two=list(changed), sha256=hashlib.sha256(data).hexdigest(),
            generated_source=str(target), compiled_source=str(cpp_path),
            cpp_sha256=hashlib.sha256(cpp.encode()).hexdigest())
    return functions, dict(
        candidate='K/V and S/alpha buffer versions: independent changes and combination',
        original_sha256=ORIGINAL_SHA256, sources=details,
        scope='Matched nostore+stages2+fused finalization. Only selected explicit buffer versions change from 3 to 2. Q copy stays inside the KV loop. Arithmetic, tiling and other versions retained. Actual allocation, schedule legality and accuracy require server validation.')

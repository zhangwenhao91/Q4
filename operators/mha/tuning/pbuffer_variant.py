"""Explicit P L1 double-buffering on top of the verified best KV2 kernel."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

from buffers_variant import transform as buffer_transform
from nostore_variant import ORIGINAL_SHA256

OLD_VERSION = 'P_nz_ub: 2,'
NEW_VERSION = 'P_shared: 2, P_nz_ub: 2,'


def best_transform(original):
    return buffer_transform(original, 'tilelang_kv2')


def transform(original):
    text = best_transform(original).decode('utf-8')
    if text.count(OLD_VERSION) != 2 or 'P_shared:' in text:
        raise RuntimeError('Unexpected pinned P buffer annotations')
    return text.replace(OLD_VERSION, NEW_VERSION).encode('utf-8')


def build_candidate(source, q, k, v):
    import tilelang
    original = (Path(source)/'core.py').read_bytes()
    functions = {}
    details = {}
    compiled = {}
    for label, data in [('tilelang_kv2', best_transform(original)),
                        ('tilelang_pbuffer', transform(original))]:
        target = Path(__file__).resolve().parents[3]/'third_party'/'mha-pbuffer-candidate'/(label+'.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f'Candidate source modified; preserve it before proceeding: {target}')
        target.write_bytes(data)
        spec = importlib.util.spec_from_file_location('q4_'+label+'_pbuffer', target)
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
        compiled[label] = cpp
        cpp_path = target.with_suffix('.cpp')
        cpp_path.write_text(cpp, encoding='utf-8')
        for line_no, line in enumerate(cpp.splitlines(), 1):
            if ('for (' in line and 'kv' in line) or any(
                    op in line for op in ('asc_copy_ub2l1', 'asc_copy_l12l0a')):
                print(f'CODEGEN {label}:{line_no}: {line.strip()}', flush=True)
        details[label] = dict(
            sha256=hashlib.sha256(data).hexdigest(), generated_source=str(target),
            compiled_source=str(cpp_path), cpp_sha256=hashlib.sha256(cpp.encode()).hexdigest())
    identical = compiled['tilelang_kv2'] == compiled['tilelang_pbuffer']
    print(f'CODEGEN_IDENTICAL_PBUFFER: {identical}', flush=True)
    return functions, dict(
        candidate='explicit P_shared versions2', original_sha256=ORIGINAL_SHA256,
        sources=details, generated_cpp_identical=identical,
        scope='Only add P_shared:2 to both buffer-annotation branches of the best KV2 source. No arithmetic, tiling, cache policy or other version changes. Inspect emitted P write/read addresses to verify actual slots. Compiled C++ equality is a diagnostic, not a machine-code comparison.')

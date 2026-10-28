"""Q64 candidate: pinned upstream math, full-width-safe per-row SIMD storage."""
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

ORIGINAL_SHA256='0c1af6f07f2c727ca6ef6d09d4ce3c21ce16286d5931e75e8a9c6f2248ed23b0'
STAT_NAMES=('m_ub','l_ub','alpha_ub','old_m_ub','row_sum_ub')


def transform(original):
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_SHA256:
        raise RuntimeError('Q64 adapter requires the exact pinned upstream core.py')
    text=original.decode('utf-8')
    marker='    D = head_dim\n'
    if text.count(marker)!=1: raise RuntimeError('Unexpected builder shape')
    text=text.replace(marker,'    ROW_STATS = max(ROWS, VL)  # Q64: full-width SIMD loads/stores need 64 FP32 slots.\n'+marker)
    changed={}
    for name in STAT_NAMES:
        pattern=rf'(\b{name} = T\.alloc_shared\()\(ROWS,\)(, accum_dtype\))'
        text,count=re.subn(pattern,rf'\g<1>(ROW_STATS,)\g<2>',text)
        expected=2
        if count!=expected: raise RuntimeError(f'Unexpected allocation count for {name}: {count}')
        changed[name]=count
    return text.encode('utf-8'),changed


def build_candidate(source,q,k,v):
    import tilelang
    import math
    original=(Path(source)/'core.py').read_bytes()
    data,changed=transform(original)
    root=Path(__file__).resolve().parents[3]
    target=root/'third_party'/'mha-q64-candidate'/'core_q64.py'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=data:
        raise RuntimeError(f'Candidate source modified; preserve it before proceeding: {target}')
    target.write_bytes(data)
    spec=importlib.util.spec_from_file_location('q4_mha_core_q64',target)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    program=module.flash_attention_fwd(q_len=4096,kv_len=4096,head_dim=128,out_dtype='bfloat16',
              softmax_scale=1/math.sqrt(128),
              tiling=module.FwdTiling(block_q=64,block_kv=128,num_stages=3))
    kernel=tilelang.compile(program,out_idx=-1,
                  pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
    return {'tilelang_q64':lambda:(kernel(q,k,v),)},dict(
        candidate='Q64 with full-width-safe scalar statistics allocation',
        baseline=dict(block_q=128,block_kv=128,num_stages=3,blocks=32),
        candidate_tiling=dict(block_q=64,block_kv=128,num_stages=3,blocks=64),
        logical_rows_per_aiv=32,statistics_storage_fp32_elements=64,
        original_sha256=ORIGINAL_SHA256,candidate_sha256=hashlib.sha256(data).hexdigest(),
        generated_source=str(target),padded_allocations=changed,
        scope='Only Q tile changes; required statistics allocation padding is explicit. No formula, dtype, KV tile, pipeline or synchronization change.')

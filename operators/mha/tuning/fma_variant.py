"""Q128 Vector candidate: one fused multiply-add for each online-output accumulation."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

ORIGINAL_SHA256='0c1af6f07f2c727ca6ef6d09d4ce3c21ce16286d5931e75e8a9c6f2248ed23b0'
OLD='''            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    prev = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, rescale_mask), cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")'''
NEW='''            rescale_mask = S.pset(32, "PAT_ALL")
            accumulated = S.alloc_var(T.float32)
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    prev = S.vld(O_ub[i, c], "NORM")
                    accumulated = S.vld(O_tmp_ub[i, c], "NORM")
                    S.vmula(accumulated, alpha, prev, rescale_mask)
                    S.vsts(O_ub[i, c], accumulated, rescale_mask, "NORM_B32")'''


def transform(original):
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_SHA256:
        raise RuntimeError('FMA candidate requires exact pinned original core.py')
    text=original.decode('utf-8')
    if text.count(OLD)!=1: raise RuntimeError('Unexpected accumulation implementation')
    return text.replace(OLD,NEW).encode('utf-8')


def build_candidate(source,q,k,v):
    import tilelang
    data=transform((Path(source)/'core.py').read_bytes())
    target=Path(__file__).resolve().parents[3]/'third_party'/'mha-fma-candidate'/'core_fma.py'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=data:
        raise RuntimeError(f'Candidate source modified; preserve it before proceeding: {target}')
    target.write_bytes(data)
    spec=importlib.util.spec_from_file_location('q4_mha_core_fma',target)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    program=module.flash_attention_fwd(q_len=4096,kv_len=4096,head_dim=128,out_dtype='bfloat16',
               softmax_scale=1/math.sqrt(128),tiling=module.FwdTiling(block_q=128,block_kv=128,num_stages=3))
    kernel=tilelang.compile(program,out_idx=-1,
                           pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
    return {'tilelang_fma':lambda:(kernel(q,k,v),)},dict(
        candidate='Q128 Vector FMA output accumulation',
        tiling=dict(block_q=128,block_kv=128,num_stages=3,blocks=32),
        original_sha256=ORIGINAL_SHA256,candidate_sha256=hashlib.sha256(data).hexdigest(),
        generated_source=str(target),
        changed_math='O <- alpha*O + P@V uses one FP32 FMA rounding instead of separate multiply/add rounding',
        scope='Only accumulate_output macro arithmetic; all data shapes, buffers, pipeline, synchronization and output dtype retained. Not stacked on Q64.')

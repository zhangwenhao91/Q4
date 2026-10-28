"""Q128 softmax candidate without scaled-score UB writeback."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

ORIGINAL_SHA256='0c1af6f07f2c727ca6ef6d09d4ce3c21ce16286d5931e75e8a9c6f2248ed23b0'
CHANGES=[('                s00 = S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), softmax_full)\n                s01 = S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), softmax_full)\n                s10 = S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), softmax_full)\n                s11 = S.vmuls(S.vld(S_ub[r + 1, VL], "NORM"), T.float32(scale), softmax_full)\n                S.vsts(S_ub[r, 0], s00, softmax_full, "NORM_B32")\n                S.vsts(S_ub[r, VL], s01, softmax_full, "NORM_B32")\n                S.vsts(S_ub[r + 1, 0], s10, softmax_full, "NORM_B32")\n                S.vsts(S_ub[r + 1, VL], s11, softmax_full, "NORM_B32")\n\n                max0 = S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full)\n                max1 = S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full)\n', '                s00 = S.vld(S_ub[r, 0], "NORM")\n                s01 = S.vld(S_ub[r, VL], "NORM")\n                s10 = S.vld(S_ub[r + 1, 0], "NORM")\n                s11 = S.vld(S_ub[r + 1, VL], "NORM")\n                max0 = S.vmuls(S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full), T.float32(scale), softmax_one)\n                max1 = S.vmuls(S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full), T.float32(scale), softmax_one)\n'), ('                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")', '                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")\n                p00 = S.vmuls(p00, T.float32(scale), softmax_full)\n                p01 = S.vmuls(p01, T.float32(scale), softmax_full)\n                p10 = S.vmuls(p10, T.float32(scale), softmax_full)\n                p11 = S.vmuls(p11, T.float32(scale), softmax_full)')]


def transform(original):
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_SHA256:
        raise RuntimeError('no-store candidate requires exact pinned original core.py')
    text=original.decode('utf-8')
    for old,new in CHANGES:
        if text.count(old)!=1: raise RuntimeError('Unexpected softmax implementation')
        text=text.replace(old,new)
    return text.encode('utf-8')


def build_candidate(source,q,k,v):
    import tilelang
    data=transform((Path(source)/'core.py').read_bytes())
    target=Path(__file__).resolve().parents[3]/'third_party'/'mha-nostore-candidate'/'core_nostore.py'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=data:
        raise RuntimeError(f'Candidate source modified; preserve it before proceeding: {target}')
    target.write_bytes(data)
    spec=importlib.util.spec_from_file_location('q4_mha_core_nostore',target)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    program=module.flash_attention_fwd(q_len=4096,kv_len=4096,head_dim=128,out_dtype='bfloat16',
               softmax_scale=1/math.sqrt(128),tiling=module.FwdTiling(block_q=128,block_kv=128,num_stages=3))
    kernel=tilelang.compile(program,out_idx=-1,
                           pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
    return {'tilelang_nostore':lambda:(kernel(q,k,v),)},dict(
        candidate='Q128 softmax without scaled-score UB writeback',
        tiling=dict(block_q=128,block_kv=128,num_stages=3,blocks=32),
        original_sha256=ORIGINAL_SHA256,candidate_sha256=hashlib.sha256(data).hexdigest(),
        generated_source=str(target),
        changed_math='Positive scale commutes with finite row maximum; scores scale in registers before exponentiation.',
        scope='Only softmax score handling; no scaled-score UB stores. Retain two passes, barriers, stats, packing, pipeline and buffers. No Q64/FMA/manual changes.')

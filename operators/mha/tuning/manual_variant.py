"""Compare the pinned core's existing manual stage schedule against automatic scheduling."""
import hashlib
import importlib
import math
from pathlib import Path

ORIGINAL_SHA256='0c1af6f07f2c727ca6ef6d09d4ce3c21ce16286d5931e75e8a9c6f2248ed23b0'


def program(core):
    return core.flash_attention_fwd(q_len=4096,kv_len=4096,head_dim=128,out_dtype='bfloat16',
        softmax_scale=1/math.sqrt(128),tiling=core.FwdTiling(block_q=128,block_kv=128,num_stages=3),
        manual_schedule=True)


def build_candidate(source,q,k,v):
    import tilelang
    digest=hashlib.sha256((Path(source)/'core.py').read_bytes()).hexdigest()
    if digest!=ORIGINAL_SHA256: raise RuntimeError('Manual candidate requires exact pinned core.py')
    # make_case has imported this exact module for the unchanged automatic baseline.
    core=importlib.import_module('core')
    if Path(core.__file__).resolve()!=(Path(source)/'core.py').resolve():
        raise RuntimeError('Unexpected core module; refusing a different source implementation')
    kernel=tilelang.compile(program(core),out_idx=-1,
                           pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
    return {'tilelang_manual':lambda:(kernel(q,k,v),)},dict(
        candidate='pinned upstream kv_manual stages; no source patch',source_sha256=digest,
        tiling=dict(block_q=128,block_kv=128,num_stages=3,blocks=32),
        manual_schedule=True,scope='Same QK/softmax/PV/output macros and buffers; existing T.Stage ordering replaces automatic placement. No Q64/FMA changes.')

"""Factorial Q128 experiment: original/nostore softmax x two/three stages."""
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path
import sys

from nostore_variant import ORIGINAL_SHA256, build_candidate as build_nostore


def program(core, stages):
    if stages not in (2, 3):
        raise ValueError('This experiment only compares stages 2 and 3')
    return core.flash_attention_fwd(
        q_len=4096, kv_len=4096, head_dim=128, out_dtype='bfloat16',
        softmax_scale=1/math.sqrt(128),
        tiling=core.FwdTiling(block_q=128, block_kv=128, num_stages=stages))


def build_candidate(source, q, k, v):
    import tilelang
    source=Path(source)
    digest=hashlib.sha256((source/'core.py').read_bytes()).hexdigest()
    if digest != ORIGINAL_SHA256:
        raise RuntimeError('Stages experiment requires exact pinned core.py')
    core=importlib.import_module('core')
    if Path(core.__file__).resolve() != (source/'core.py').resolve():
        raise RuntimeError('Unexpected original core module')
    functions, nostore_meta=build_nostore(source,q,k,v)
    generated=Path(nostore_meta['generated_source'])
    spec=importlib.util.spec_from_file_location('q4_mha_nostore_stages',generated)
    nostore=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=nostore
    spec.loader.exec_module(nostore)
    for label, module in [('tilelang_s2',core), ('tilelang_nostore_s2',nostore)]:
        kernel=tilelang.compile(program(module,2),out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
        functions[label]=lambda kernel=kernel:(kernel(q,k,v),)
    return functions, dict(
        candidate='original/nostore x stages 2/3; Q128 KV128, automatic scheduling',
        original_sha256=digest, nostore=nostore_meta,
        labels=dict(tilelang_q128='original, stages3', tilelang_s2='original, stages2',
                    tilelang_nostore='nostore, stages3', tilelang_nostore_s2='nostore, stages2'),
        scope='Only tiling.num_stages changes within each matched pair. No manual/FMA/row fusion changes. Stage count may affect compiler scheduling and buffer versions; it does not guarantee actual overlap.')

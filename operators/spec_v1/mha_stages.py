"""Five-way stage-depth experiment, including the current best candidate."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata != 'cached':
        raise ValueError('Stages experiment requires cached metadata')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'operators/mha/tuning'))
    from stages_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions,refs,names,meta=native_case(a,extra_builder=build_candidate)
    meta['native_kernels']={
        'tilelang_q128':'original Q128/KV128/stages3',
        'tilelang_nostore':'nostore Q128/KV128/stages3',
        'tilelang_s2':'original Q128/KV128/stages2',
        'tilelang_nostore_s2':'nostore Q128/KV128/stages2',
        'ascendc':'native FlashAttn, cached metadata'}
    return functions,refs,names,meta

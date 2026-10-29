"""Compare Q residency against the original, current best, and native AscendC."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata != 'cached':
        raise ValueError('Q residency experiment requires cached native metadata')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'operators/mha/tuning'))
    from qresident_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions, refs, names, meta = native_case(a, extra_builder=build_candidate)
    meta['native_kernels'] = {
        'tilelang_q128': 'original Q128/KV128/stages3',
        'tilelang_finalize': 'nostore+stages2+fused finalization, previous best',
        'tilelang_qresident': 'previous best plus Q L0A residency',
        'ascendc': 'native FlashAttn, cached metadata'}
    return functions, refs, names, meta

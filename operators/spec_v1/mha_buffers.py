"""Six-way explicit buffer-version comparison including original and best kernels."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata != 'cached':
        raise ValueError('Buffer experiment requires cached native metadata')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'operators/mha/tuning'))
    from buffers_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions, refs, names, meta = native_case(a, extra_builder=build_candidate)
    meta['native_kernels'] = {
        'tilelang_q128': 'original Q128/KV128/stages3',
        'tilelang_finalize': 'current best: nostore+stages2+fused finalization',
        'tilelang_kv2': 'current best with K/V versions2',
        'tilelang_stats2': 'current best with S/alpha versions2',
        'tilelang_buffers2': 'current best with K/V/S/alpha versions2',
        'ascendc': 'native FlashAttn, cached metadata'}
    return functions, refs, names, meta

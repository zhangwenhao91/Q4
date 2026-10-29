"""Four-way first-PV initialization experiment against the verified best kernel."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata != 'cached':
        raise ValueError('First-PV experiment requires cached native metadata')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'operators/mha/tuning'))
    from firstpv_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions, refs, names, meta = native_case(a, extra_builder=build_candidate)
    meta['native_kernels'] = {
        'tilelang_q128': 'original Q128/KV128/stages3',
        'tilelang_kv2': 'current best: nostore+stages2+fused finalization+KV2',
        'tilelang_firstpv': 'current best plus first PV directly initializes O',
        'ascendc': 'native FlashAttn, cached metadata'}
    return functions, refs, names, meta

"""Five-way split S/alpha buffer-version comparison on the verified KV2 best."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata != 'cached':
        raise ValueError('Buffer experiment requires cached native metadata')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'operators/mha/tuning'))
    from stats_split_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions, refs, names, meta = native_case(a, extra_builder=build_candidate)
    meta['native_kernels'] = {
        'tilelang_q128': 'original Q128/KV128/stages3',
        'tilelang_kv2': 'current best, S/alpha both versions3',
        'tilelang_s2buf': 'current best, S versions2, alpha versions3',
        'tilelang_alpha2buf': 'current best, S versions3, alpha versions2',
        'ascendc': 'native FlashAttn, cached metadata'}
    return functions, refs, names, meta

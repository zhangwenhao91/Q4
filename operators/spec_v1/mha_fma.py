"""Three-way original Q128 / Q128 FMA / native cached-metadata AC comparison."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata!='cached': raise ValueError('FMA tuning requires cached native metadata')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'operators'/'mha'/'tuning'))
    from fma_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions,refs,names,meta=native_case(a,extra_builder=build_candidate)
    meta['native_kernels']={
        'tilelang_q128':'unchanged pinned Q128/KV128/stages3',
        'tilelang_fma':'Q128/KV128/stages3, explicit SIMD FMA output accumulation',
        'ascendc':'unchanged native FlashAttn, cached metadata'}
    return functions,refs,names,meta

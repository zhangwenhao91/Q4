"""Three-way original Q128 / Q128 row-fused softmax / native cached-metadata AC comparison."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata!='cached': raise ValueError('row-fused softmax tuning requires cached native metadata')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'operators'/'mha'/'tuning'))
    from rowfused_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions,refs,names,meta=native_case(a,extra_builder=build_candidate)
    meta['native_kernels']={
        'tilelang_q128':'unchanged pinned Q128/KV128/stages3',
        'tilelang_rowfused':'Q128/KV128/stages3, single-pass row softmax',
        'ascendc':'unchanged native FlashAttn, cached metadata'}
    return functions,refs,names,meta

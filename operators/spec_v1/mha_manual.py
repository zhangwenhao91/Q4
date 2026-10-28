"""Three-way original Q128 / upstream manual-stage Q128 / cached native AC."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata!='cached': raise ValueError('Manual scheduling comparison requires cached metadata')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'operators'/'mha'/'tuning'))
    from manual_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions,refs,names,meta=native_case(a,extra_builder=build_candidate)
    meta['native_kernels']={
        'tilelang_q128':'unchanged Q128/KV128/stages3, kv_auto',
        'tilelang_manual':'same pinned source and shapes, kv_manual T.Stage constraints',
        'ascendc':'unchanged native FlashAttn, cached metadata'}
    return functions,refs,names,meta

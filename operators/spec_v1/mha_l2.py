"""Six-way Q/O cache-policy comparison against the verified KV2 best."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata != 'cached':
        raise ValueError('L2 experiment requires cached native metadata')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'operators/mha/tuning'))
    from l2_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions,refs,names,meta = native_case(a,extra_builder=build_candidate)
    meta['native_kernels'] = {
        'tilelang_q128':'original Q128/KV128/stages3',
        'tilelang_kv2':'current best KV2, original Q/O cache hints',
        'tilelang_q_l2':'current best with Q load cache mode4',
        'tilelang_o_l2':'current best with O store cache mode4',
        'tilelang_qo_l2':'current best with Q/O cache mode4',
        'ascendc':'native FlashAttn, cached metadata'}
    return functions,refs,names,meta

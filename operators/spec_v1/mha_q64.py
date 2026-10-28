"""Three-way Q128 / Q64 candidate / native AscendC comparison with shared inputs."""
from pathlib import Path
import sys


def make_case(a):
    if a.ac_metadata!='cached':
        raise ValueError('Q64 tuning compares against the faster cached-metadata native baseline')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'operators'/'mha'/'tuning'))
    from q64_variant import build_candidate
    from mha_bf16 import make_case as native_case
    functions,refs,names,meta=native_case(a,extra_builder=build_candidate)
    meta['native_kernels']={
        'tilelang_q128':'unchanged pinned core.py, default FwdTiling(128,128,3)',
        'tilelang_q64':'generated core_q64.py, FwdTiling(64,128,3), stats storage padded to 64',
        'ascendc':'unchanged native FlashAttn, cached metadata'}
    return functions,refs,names,meta

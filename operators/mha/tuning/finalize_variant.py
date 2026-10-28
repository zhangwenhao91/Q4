"""Fuse final normalization/cast on the validated nostore stages2 configuration."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from nostore_variant import transform as nostore_transform, ORIGINAL_SHA256

OLD='    @T.macro\n    def finalize_cast(mt, O, O_ub, O_cast_ub, l_ub):\n        with T.SimdVF(latency=1780):\n            normalize_mask = S.pset(32, "PAT_ALL")\n            for i in range(ROWS):\n                denominator = S.vld(l_ub[i], "BRC_B32")\n                for c in range(0, D, VL):\n                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)\n                    S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")\n        with T.SimdVF(latency=193):\n            cast_full = S.pset(32, "PAT_ALL")\n            for r in range(ROWS):\n                for c in range(0, D, VL):\n                    ob = S.vcvt(S.vld(O_ub[r, c], "NORM"), out_dtype, cast_full)\n                    S.vsts(O_cast_ub[r, c], ob, cast_full, "PK_B32")\n        T.dual_copy(O_cast_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])\n\n'
NEW='    @T.macro\n    def finalize_cast(mt, O, O_ub, O_cast_ub, l_ub):\n        with T.SimdVF(latency=1973):\n            normalize_mask = S.pset(32, "PAT_ALL")\n            for i in range(ROWS):\n                denominator = S.vld(l_ub[i], "BRC_B32")\n                for c in range(0, D, VL):\n                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)\n                    ob = S.vcvt(normalized, out_dtype, normalize_mask)\n                    S.vsts(O_cast_ub[i, c], ob, normalize_mask, "PK_B32")\n        T.dual_copy(O_cast_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])\n\n'


def transform(original):
    # nostore_transform verifies the exact original hash before changing softmax.
    text=nostore_transform(original).decode('utf-8')
    if text.count(OLD)!=1: raise RuntimeError('Unexpected finalize_cast implementation')
    return text.replace(OLD,NEW).encode('utf-8')


def build_candidate(source,q,k,v):
    import tilelang
    original=(Path(source)/'core.py').read_bytes()
    functions={};details={}
    for label,data in [('tilelang_nostore_s2',nostore_transform(original)),
                       ('tilelang_finalize',transform(original))]:
        target=Path(__file__).resolve().parents[3]/'third_party'/'mha-finalize-candidate'/(label+'.py')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=data:
            raise RuntimeError(f'Candidate source modified; preserve before proceeding: {target}')
        target.write_bytes(data)
        spec=importlib.util.spec_from_file_location('q4_'+label+'_finalize',target)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        program=module.flash_attention_fwd(q_len=4096,kv_len=4096,head_dim=128,out_dtype='bfloat16',
            softmax_scale=1/math.sqrt(128),tiling=module.FwdTiling(block_q=128,block_kv=128,num_stages=2))
        kernel=tilelang.compile(program,out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
        functions[label]=lambda kernel=kernel:(kernel(q,k,v),)
        details[label]=dict(sha256=hashlib.sha256(data).hexdigest(),generated_source=str(target))
    return functions,dict(candidate='nostore stages2 + final normalize/cast fusion',
        original_sha256=ORIGINAL_SHA256,sources=details,
        scope='Only finalize_cast differs from matched nostore stages2. Keep FP32 division, BF16 conversion, output alias and forward traversal. One VF replaces two; hint 1973 is sum of original hints, not a measured latency. Automatic scheduling may change.')

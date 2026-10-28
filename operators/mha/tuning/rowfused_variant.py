"""Q128 single-pass row softmax candidate; NPU validation required."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

ORIGINAL_SHA256='0c1af6f07f2c727ca6ef6d09d4ce3c21ce16286d5931e75e8a9c6f2248ed23b0'
NEW_SOFTMAX='''    @T.macro
    def softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        with T.SimdVF(latency=735):
            softmax_full = S.pset(32, "PAT_ALL")
            softmax_one = S.pset(32, "PAT_VL1")
            softmax_b16 = S.pset(16, "PAT_VL128")
            dst_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC),
                "bfloat16",
            )

            for r in range(0, ROWS, 2):
                p00, p01 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                s00 = S.vmuls(p00, T.float32(scale), softmax_full)
                s01 = S.vmuls(p01, T.float32(scale), softmax_full)
                local_max0 = S.vdupv(S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full), softmax_full)
                old_max0 = S.vld(m_ub[r], "BRC_B32")
                max0 = S.vmax(old_max0, local_max0, softmax_full)
                e00 = S.vexpdif(s00, max0, softmax_full)
                e01 = S.vexpdif(s01, max0, softmax_full)
                row_sum0 = S.vcadd(S.vadd(e00, e01, softmax_full), softmax_full)
                alpha0 = S.vexpdif(old_max0, max0, softmax_one)
                old_l0 = S.vld(l_ub[r], "BRC_B32")
                new_l0 = S.vadd(S.vmul(old_l0, alpha0, softmax_one), row_sum0, softmax_one)
                S.vsts(m_ub[r], max0, softmax_one, "ONEPT_B32")
                S.vsts(l_ub[r], new_l0, softmax_one, "ONEPT_B32")
                S.vsts(alpha_ub[r], alpha0, softmax_one, "ONEPT_B32")
                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")
                s10 = S.vmuls(p10, T.float32(scale), softmax_full)
                s11 = S.vmuls(p11, T.float32(scale), softmax_full)
                local_max1 = S.vdupv(S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full), softmax_full)
                old_max1 = S.vld(m_ub[r + 1], "BRC_B32")
                max1 = S.vmax(old_max1, local_max1, softmax_full)
                e10 = S.vexpdif(s10, max1, softmax_full)
                e11 = S.vexpdif(s11, max1, softmax_full)
                row_sum1 = S.vcadd(S.vadd(e10, e11, softmax_full), softmax_full)
                alpha1 = S.vexpdif(old_max1, max1, softmax_one)
                old_l1 = S.vld(l_ub[r + 1], "BRC_B32")
                new_l1 = S.vadd(S.vmul(old_l1, alpha1, softmax_one), row_sum1, softmax_one)
                S.vsts(m_ub[r + 1], max1, softmax_one, "ONEPT_B32")
                S.vsts(l_ub[r + 1], new_l1, softmax_one, "ONEPT_B32")
                S.vsts(alpha_ub[r + 1], alpha1, softmax_one, "ONEPT_B32")
                even_bf16_0 = S.vcvt(e00, "bfloat16", softmax_full, sat=False, part=0)
                even_bf16_1 = S.vcvt(e10, "bfloat16", softmax_full, sat=False, part=0)
                odd_bf16_0 = S.vcvt(e01, "bfloat16", softmax_full, sat=False, part=1)
                odd_bf16_1 = S.vcvt(e11, "bfloat16", softmax_full, sat=False, part=1)
                merged_bits0 = S.vor(
                    T.reinterpret(even_bf16_0, "uint16x128"),
                    T.reinterpret(odd_bf16_0, "uint16x128"),
                    softmax_b16,
                )
                merged_bits1 = S.vor(
                    T.reinterpret(even_bf16_1, "uint16x128"),
                    T.reinterpret(odd_bf16_1, "uint16x128"),
                    softmax_b16,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits0, "bfloat16x128"),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits1, "bfloat16x128"),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )


'''


def transform(original):
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_SHA256:
        raise RuntimeError('row-fused candidate requires exact pinned original core.py')
    text=original.decode('utf-8')
    start=text.index('    @T.macro\n    def softmax(')
    end=text.index('    @T.macro\n    def pack_p(',start)
    return (text[:start]+NEW_SOFTMAX+text[end:]).encode('utf-8')


def build_candidate(source,q,k,v):
    import tilelang
    data=transform((Path(source)/'core.py').read_bytes())
    target=Path(__file__).resolve().parents[3]/'third_party'/'mha-rowfused-candidate'/'core_rowfused.py'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=data:
        raise RuntimeError(f'Candidate source modified; preserve it before proceeding: {target}')
    target.write_bytes(data)
    spec=importlib.util.spec_from_file_location('q4_mha_core_rowfused',target)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    program=module.flash_attention_fwd(q_len=4096,kv_len=4096,head_dim=128,out_dtype='bfloat16',
               softmax_scale=1/math.sqrt(128),tiling=module.FwdTiling(block_q=128,block_kv=128,num_stages=3))
    kernel=tilelang.compile(program,out_idx=-1,
                           pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
    return {'tilelang_rowfused':lambda:(kernel(q,k,v),)},dict(
        candidate='Q128 single-pass row softmax',
        tiling=dict(block_q=128,block_kv=128,num_stages=3,blocks=32),
        original_sha256=ORIGINAL_SHA256,candidate_sha256=hashlib.sha256(data).hexdigest(),
        generated_source=str(target),
        changed_math='Same online softmax equations and probability packing; max reduction groups even/odd lanes.',
        scope='Replace only softmax macro: keep scores and row stats in registers, scalar-store final m/l/alpha once. Original buffer declarations and latency hint retained; automatic scheduler may choose different placement. No FMA/Q64/manual changes.')

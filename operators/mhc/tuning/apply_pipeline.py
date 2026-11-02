# SPDX-License-Identifier: MIT
# Adapted from TileKernels c1cf914 pre_apply_mix_asc.py; see LICENSE_TILEKERNELS.
# Experimental: token ownership retained, hidden tiles software-pipelined.
import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


# Each bf16 chunk is unpacked into a 64-lane fp32 vector register.
_VEC = 64


@tilelang.jit()
def mhc_pre_apply_pipeline_asc(x, mix, o, h_blk, num_vec_cores: int, pipeline_stages: int):
    """Ascend forward: ``o[n, h] = sum_mhc mix[n, mhc] * x[n, mhc, h]``."""
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    x: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    mix: T.Tensor[(num_tokens, mhc), T.float32]
    o: T.Tensor[(num_tokens, hidden), T.bfloat16]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC
    assert hidden % h_blk == 0 and h_blk % 64 == 0
    num_stages = pipeline_stages

    with T.Kernel(num_vec_cores) as core_id:
        x_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        o_ub = T.alloc_shared((h_blk,), T.bfloat16)
        T.annotate_buffer_versions({x_ub: num_stages, mix_ub: num_stages, o_ub: num_stages})

        for pid_n in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=1):
            for pid_h in T.Pipelined(num_hidden_tiles, num_stages=num_stages):
                T.copy(mix[pid_n, 0], mix_ub[0:mhc])
                T.copy(x[pid_n, 0, pid_h * h_blk], x_ub)

                with T.SimdVF():
                    # Preload the mix scalars as broadcast registers once per token-tile.
                    mix_reg = S.alloc_local((mhc,), T.float32)
                    for m in T.unroll(mhc, explicit=True):
                        mix_reg[m] = S.vld(mix_ub[m], dist='BRC_B32')
                    xs = S.alloc_local((mhc,), T.float32)
                    acc = S.alloc_var(T.float32)
                    for c in T.serial(num_chunks):
                        col = c * _VEC
                        for m in T.unroll(mhc, explicit=True):
                            xs[m] = S.vcvt(S.vld(x_ub[m, col], dist='UNPK_B16'), T.float32, part=0)
                        acc = S.vmul(xs[0], mix_reg[0])
                        for m in T.unroll(mhc - 1, explicit=True):
                            S.vmula(acc, xs[m + 1], mix_reg[m + 1])
                        S.vsts(o_ub[col], S.vcvt(acc, T.bfloat16), dist='PK_B32')

                T.copy(o_ub, o[pid_n, pid_h * h_blk])

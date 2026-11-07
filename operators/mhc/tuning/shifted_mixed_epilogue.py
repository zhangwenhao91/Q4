"""Vector epilogue for the token-owned shifted mHC mixed kernel."""
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from shifted_pre_three_kernels import compute_coefficients


@T.macro
def shifted_coeff_epilogue(x, prev_pre, alpha, bias, hi, pre, post, comb,
                           accum, sums, token_begin, token_stage, aiv_id, hidden, l0c_stages,
                           aiv_tokens):
    # Fixpipe splits the tile along M between the two local AIVs.
    proj_ub = T.alloc_shared((aiv_tokens * 24,), T.float32)
    scale_ub = T.alloc_shared((3, 8), T.float32)
    base_ub = T.alloc_shared((24, 8), T.float32)
    mixes_ub = T.alloc_shared((16 * 24,), T.float32)
    pre_ub = T.alloc_shared((16, 4), T.float32)
    post_ub = T.alloc_shared((16, 4), T.float32)
    comb_in_ub = T.alloc_shared((16, 16), T.float32)
    comb_out_ub = T.alloc_shared((16, 16), T.float32)

    T.dual_copy(accum[token_stage, 0:2 * aiv_tokens, 0:24],
                T.view(proj_ub, (aiv_tokens, 24)), unit_flag_ctrl=3)
    with T.SimtVF(threads=32):
        for i in T.Parallel(3):
            scale_ub[i, 0] = alpha[i]
        for i in T.Parallel(24):
            base_ub[i, 0] = bias[i]

    aiv_begin = token_begin + aiv_id * aiv_tokens
    for tile in T.Serial(aiv_tokens // 16):
        start = aiv_begin + tile * 16
        compute_coefficients(
            proj_ub, T.view(sums, (l0c_stages * 64,)), scale_ub, base_ub, mixes_ub,
            pre_ub, post_ub, comb_in_ub, comb_out_ub,
            tile * 16 * 24, token_stage * 64 + tile * 16, 16, 4, 16, 24,
            T.float32(1.0 / (4 * hidden)), 1e-20, 1e-6, 1e-6, 2., 20)
        T.copy(pre_ub, pre[start : start + 16, 0:4])
        T.copy(post_ub, post[start : start + 16, 0:4])
        T.copy(comb_out_ub, comb[start : start + 16, 0:16])


@T.macro
def shifted_epilogue(x, prev_pre, alpha, bias, hi, pre, post, comb,
                     accum, sums, token_begin, token_stage, aiv_id, hidden, l0c_stages,
                     aiv_tokens):
    shifted_coeff_epilogue(x, prev_pre, alpha, bias, hi, pre, post, comb,
                           accum, sums, token_begin, token_stage, aiv_id, hidden, l0c_stages,
                           aiv_tokens)
    residual_ub = T.alloc_shared((4, 256), T.bfloat16)
    old_mix_ub = T.alloc_shared((8,), T.float32)
    hi_ub = T.alloc_shared((256,), T.bfloat16)
    aiv_begin = token_begin + aiv_id * 64
    for row in T.Serial(64):
        token = aiv_begin + row
        T.copy(prev_pre[token, 0:4], old_mix_ub[0:4])
        for block in T.Pipelined(hidden // 256, num_stages=2):
            col = block * 256
            for stream in T.unroll(4):
                offset = stream * hidden + col
                T.copy(x[token, offset : offset + 256], residual_ub[stream, 0:256])
            with T.SimdVF():
                weights = S.alloc_local((4,), T.float32)
                for stream in T.unroll(4, explicit=True):
                    weights[stream] = S.vld(old_mix_ub[stream], dist='BRC_B32')
                for lane_block in T.Serial(4):
                    lane = lane_block * 64
                    first = S.vcvt(S.vld(residual_ub[0, lane], dist='UNPK_B16'), T.float32, part=0)
                    value = S.alloc_var(T.float32)
                    value = S.vmul(first, weights[0])
                    for stream in T.unroll(3, explicit=True):
                        other = S.vcvt(S.vld(residual_ub[stream + 1, lane], dist='UNPK_B16'), T.float32, part=0)
                        S.vmula(value, other, weights[stream + 1])
                    S.vsts(hi_ub[lane], S.vcvt(value, T.bfloat16), dist='PK_B32')
            T.copy(hi_ub, hi[token, col : col + 256])

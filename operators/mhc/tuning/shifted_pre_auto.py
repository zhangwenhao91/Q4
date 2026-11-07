# SPDX-License-Identifier: MIT
# Copyright (c) 2026 DeepSeek. Portions adapted for Q4 shifted mHC.
"""Standalone preallocated H5120 shifted Pre with N-based dispatch.

N < 512 uses two K8 kernels. N >= 512 uses the original three K8 kernels.
N must be a positive multiple of 128. Apply consumes the previous layer's pre.
Returned outputs are (layer_input, next_pre_mix, post_mix, comb_mix).
Bind once per input set and call the returned launcher on one stream.
Outputs and workspace are reused, so every call overwrites the previous outputs.

Adapted from DeepGEMM-Ascend 3a353262 and TileKernels c1cf914.
See LICENSE_DEEPGEMM, LICENSE_TILEKERNELS and source.json in this directory.

Usage from the repository root:
    from operators.mhc.tuning.shifted_pre_auto import bind_shifted_pre
    run, metadata = bind_shifted_pre(x, prev_pre, phi, alpha, bias)
    hi, next_pre, post, comb = run()
"""

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout


def _u32_index(index_i32):
    return T.reinterpret(index_i32, 'uint32x64')


def _row_reduce(mat, mask_lo, mask_hi, bcast_idx2):
    """Row reduce: vcgadd ×2 + vselr ×2 + vsel → broadcast row sum."""
    return S.vsel(
        S.vselr(S.vcgadd(mat, mask_lo), bcast_idx2),
        S.vselr(S.vcgadd(mat, mask_hi), bcast_idx2),
        mask_lo,
    )


def _row_max(mat, mask_lo, mask_hi, bcast_idx2):
    """Row max: vcgmax ×2 + vselr ×2 + vsel → broadcast row max."""
    return S.vsel(
        S.vselr(S.vcgmax(mat, mask_lo), bcast_idx2),
        S.vselr(S.vcgmax(mat, mask_hi), bcast_idx2),
        mask_lo,
    )


@T.macro
def compute_coefficients(
    mul_flat_ub,
    sq_ub,
    scale_ub,
    base_ub,
    mixes_flat_ub,
    pre_ub,
    post_ub,
    comb_in_ub,
    comb_out_ub,
    mul_offset,
    sq_offset,
    tb,
    mhc,
    mhc2,
    mhc3,
    inv_rms_group_size,
    rms_eps,
    mhc_pre_eps,
    mhc_sinkhorn_eps,
    mhc_post_mult_value,
    sinkhorn_repeat,
):
    with T.SimdVF():
        mask = S.pset(32)
        ones = S.vdup(T.float32(1), T.float32, mask)
        zeros = S.vdup(T.float32(0), T.float32, mask)
        inv_group = S.vdup(inv_rms_group_size, T.float32, mask)
        lane_idx = S.vci(T.int32(0), T.int32)

        ##################################################################
        # Phase A: mixes = gemm_out_mul * rms, vectorized with the 6-VR
        # compact token layout (copied from the reduce kernel).
        # 16 tokens x 24 mixes = 384 fp32 = 6 VRs of 64 lanes; each
        # token spans 24 lanes of its VR.
        for vec_idx in T.unroll(tb * mhc3 // 64, explicit=True):
            m_vec = S.vld(mul_flat_ub[mul_offset + vec_idx * 64])
            rms_vec = S.alloc_var(T.float32)
            rms_vec = zeros
            elem_start = vec_idx * 64
            elem_end = elem_start + 64
            token_start = elem_start // mhc3
            token_end = (elem_end - 1) // mhc3
            for token_idx in range(token_start, token_end + 1):
                sq_val = S.vld(sq_ub[sq_offset + token_idx], dist='BRC_B32')
                rms_val = S.vdiv(
                    ones,
                    S.vsqrt(
                        S.vadds(S.vmul(sq_val, inv_group, mask), T.float32(rms_eps), mask), mask
                    ),
                    mask,
                )
                lane_start = token_idx * mhc3 - elem_start
                lane_end = lane_start + mhc3
                token_mask = S.pand(
                    S.vcmps(lane_idx, T.int32(lane_start), mask, 'ge'),
                    S.vcmps(lane_idx, T.int32(lane_end), mask, 'lt'),
                    mask,
                )
                rms_vec = S.vsel(rms_val, rms_vec, token_mask)
            S.vsts(mixes_flat_ub[vec_idx * 64], S.vmul(m_vec, rms_vec, mask), mask, 'NORM_B32')

        S.mem_bar('VST_VLD')

        ##################################################################
        # Phase B: split mixes into pre/post/comb (copied from
        # pre_split_mixes_fwd_simd_mhc4_asc, adjusted for tb tokens).
        idx = S.vci(T.int32(0), T.int32)
        col3 = S.vand(idx, S.vdup(T.int32(3), T.int32, mask), mask)
        col15 = S.vand(idx, S.vdup(T.int32(15), T.int32, mask), mask)
        scale0 = S.vld(scale_ub[0, 0], dist='BRC_B32')
        scale1 = S.vld(scale_ub[1, 0], dist='BRC_B32')
        scale2 = S.vld(scale_ub[2, 0], dist='BRC_B32')
        pre_base_vec = S.alloc_var(T.float32)
        post_base_vec = S.alloc_var(T.float32)
        comb_base_vec = S.alloc_var(T.float32)
        pre_base_vec = zeros
        post_base_vec = zeros
        comb_base_vec = zeros
        for base_id in range(mhc):
            base_mask = S.vcmps(col3, T.int32(base_id), mask, 'eq')
            pre_base_vec = S.vsel(
                S.vld(base_ub[base_id, 0], dist='BRC_B32'), pre_base_vec, base_mask
            )
            post_base_vec = S.vsel(
                S.vld(base_ub[base_id + mhc, 0], dist='BRC_B32'), post_base_vec, base_mask
            )
        for base_id in range(mhc2):
            base_mask = S.vcmps(col15, T.int32(base_id), mask, 'eq')
            comb_base_vec = S.vsel(
                S.vld(base_ub[base_id + mhc * 2, 0], dist='BRC_B32'),
                comb_base_vec,
                base_mask,
            )

        # pre/post: 16 tokens per VR (4 lanes each)
        for vec_id in range(tb // 16):
            token = S.vadd(
                S.vshrs(idx, T.int32(2), mask), S.vdup(T.int32(vec_id * 16), T.int32, mask), mask
            )
            token_offset = S.vadd(
                S.vshls(token, T.int32(4), mask), S.vshls(token, T.int32(3), mask), mask
            )
            pre_index = S.vadd(token_offset, col3, mask)
            post_index = S.vadds(pre_index, T.int32(4), mask)

            pre_val = S.vadd(
                S.vmul(S.vgather2(mixes_flat_ub[0], _u32_index(pre_index), mask), scale0, mask),
                pre_base_vec,
                mask,
            )
            pre_sig = S.vdiv(
                ones, S.vadds(S.vexpdif(zeros, pre_val, mask), T.float32(1), mask), mask
            )
            S.vsts(
                pre_ub[vec_id * 16, 0],
                S.vadds(pre_sig, T.float32(mhc_pre_eps), mask),
                mask,
                'NORM_B32',
            )

            post_val = S.vadd(
                S.vmul(S.vgather2(mixes_flat_ub[0], _u32_index(post_index), mask), scale1, mask),
                post_base_vec,
                mask,
            )
            post_sig = S.vdiv(
                ones, S.vadds(S.vexpdif(zeros, post_val, mask), T.float32(1), mask), mask
            )
            S.vsts(
                post_ub[vec_id * 16, 0],
                S.vmuls(post_sig, T.float32(mhc_post_mult_value), mask),
                mask,
                'NORM_B32',
            )

        # comb: 4 tokens per VR (16 lanes each)
        for vec_id in range(tb // 4):
            token = S.vadd(
                S.vshrs(idx, T.int32(4), mask), S.vdup(T.int32(vec_id * 4), T.int32, mask), mask
            )
            token_offset = S.vadd(
                S.vshls(token, T.int32(4), mask), S.vshls(token, T.int32(3), mask), mask
            )
            comb_index = S.vadd(S.vadds(token_offset, T.int32(8), mask), col15, mask)
            comb_val = S.vadd(
                S.vmul(S.vgather2(mixes_flat_ub[0], _u32_index(comb_index), mask), scale2, mask),
                comb_base_vec,
                mask,
            )
            S.vsts(comb_in_ub[vec_id * 4, 0], comb_val, mask, 'NORM_B32')

        S.mem_bar('VST_VLD')

        ##################################################################
        # _mhc_sinkhorn_fwd (4 tokens packed per VR, copied from sinkhorn_asc)
        mask2 = mask
        idx = S.vci(T.int32(0), T.int32)
        const_7 = S.vdup(T.int32(7), T.int32, mask2)
        mod8 = S.vand(idx, const_7, mask2)
        mask_lo = S.vcmps(mod8, T.int32(4), mask2, 'lt')
        mask_hi = S.vcmps(mod8, T.int32(4), mask2, 'ge')

        two = T.int32(2)
        three = T.int32(3)
        four = T.int32(4)
        const_15 = S.vdup(T.int32(15), T.int32, mask2)
        token2 = S.vshls(S.vshrs(idx, four, mask2), T.int32(1), mask2)
        local = S.vand(idx, const_15, mask2)
        local_div8 = S.vshrs(local, three, mask2)
        bcast_idx2 = S.vadd(token2, local_div8, mask2)

        token_base_tidx = S.vshls(S.vshrs(idx, four, mask2), four, mask2)
        col = S.vand(idx, S.vdup(T.int32(3), T.int32, mask2), mask2)
        col4 = S.vshls(col, two, mask2)
        t_idx = S.vadd(S.vadd(token_base_tidx, col4, mask2), S.vshrs(local, two, mask2), mask2)

        cur_a = S.alloc_var(T.float32)
        cur_b = S.alloc_var(T.float32)
        cur_c = S.alloc_var(T.float32)
        cur_d = S.alloc_var(T.float32)
        for g in range(tb // 16):
            g0 = 4 * g
            mat_a_0 = S.vld(comb_in_ub[g0 * 4, 0])
            mat_b_0 = S.vld(comb_in_ub[(g0 + 1) * 4, 0])
            mat_c_0 = S.vld(comb_in_ub[(g0 + 2) * 4, 0])
            mat_d_0 = S.vld(comb_in_ub[(g0 + 3) * 4, 0])

            rmax_a = _row_max(mat_a_0, mask_lo, mask_hi, bcast_idx2)
            rmax_b = _row_max(mat_b_0, mask_lo, mask_hi, bcast_idx2)
            rmax_c = _row_max(mat_c_0, mask_lo, mask_hi, bcast_idx2)
            rmax_d = _row_max(mat_d_0, mask_lo, mask_hi, bcast_idx2)
            mat_a_1 = S.vexpdif(mat_a_0, rmax_a, mask2)
            mat_b_1 = S.vexpdif(mat_b_0, rmax_b, mask2)
            mat_c_1 = S.vexpdif(mat_c_0, rmax_c, mask2)
            mat_d_1 = S.vexpdif(mat_d_0, rmax_d, mask2)
            rsum_a = _row_reduce(mat_a_1, mask_lo, mask_hi, bcast_idx2)
            rsum_b = _row_reduce(mat_b_1, mask_lo, mask_hi, bcast_idx2)
            rsum_c = _row_reduce(mat_c_1, mask_lo, mask_hi, bcast_idx2)
            rsum_d = _row_reduce(mat_d_1, mask_lo, mask_hi, bcast_idx2)
            mat_a_2 = S.vdiv(mat_a_1, rsum_a, mask2)
            mat_b_2 = S.vdiv(mat_b_1, rsum_b, mask2)
            mat_c_2 = S.vdiv(mat_c_1, rsum_c, mask2)
            mat_d_2 = S.vdiv(mat_d_1, rsum_d, mask2)
            mat_a_3 = S.vadds(mat_a_2, T.float32(mhc_sinkhorn_eps), mask2)
            mat_b_3 = S.vadds(mat_b_2, T.float32(mhc_sinkhorn_eps), mask2)
            mat_c_3 = S.vadds(mat_c_2, T.float32(mhc_sinkhorn_eps), mask2)
            mat_d_3 = S.vadds(mat_d_2, T.float32(mhc_sinkhorn_eps), mask2)

            tr_a = S.vselr(mat_a_3, t_idx)
            tr_b = S.vselr(mat_b_3, t_idx)
            tr_c = S.vselr(mat_c_3, t_idx)
            tr_d = S.vselr(mat_d_3, t_idx)
            csum_a = S.vsel(
                S.vselr(S.vcgadd(tr_a, mask_lo), bcast_idx2),
                S.vselr(S.vcgadd(tr_a, mask_hi), bcast_idx2),
                mask_lo,
            )
            csum_b = S.vsel(
                S.vselr(S.vcgadd(tr_b, mask_lo), bcast_idx2),
                S.vselr(S.vcgadd(tr_b, mask_hi), bcast_idx2),
                mask_lo,
            )
            csum_c = S.vsel(
                S.vselr(S.vcgadd(tr_c, mask_lo), bcast_idx2),
                S.vselr(S.vcgadd(tr_c, mask_hi), bcast_idx2),
                mask_lo,
            )
            csum_d = S.vsel(
                S.vselr(S.vcgadd(tr_d, mask_lo), bcast_idx2),
                S.vselr(S.vcgadd(tr_d, mask_hi), bcast_idx2),
                mask_lo,
            )
            tr_a = S.vdiv(tr_a, S.vadds(csum_a, T.float32(mhc_sinkhorn_eps), mask2), mask2)
            tr_b = S.vdiv(tr_b, S.vadds(csum_b, T.float32(mhc_sinkhorn_eps), mask2), mask2)
            tr_c = S.vdiv(tr_c, S.vadds(csum_c, T.float32(mhc_sinkhorn_eps), mask2), mask2)
            tr_d = S.vdiv(tr_d, S.vadds(csum_d, T.float32(mhc_sinkhorn_eps), mask2), mask2)
            cur_a = S.vselr(tr_a, t_idx)
            cur_b = S.vselr(tr_b, t_idx)
            cur_c = S.vselr(tr_c, t_idx)
            cur_d = S.vselr(tr_d, t_idx)

            for _step in range(sinkhorn_repeat - 1):
                rsum_a = S.vsel(
                    S.vselr(S.vcgadd(cur_a, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(cur_a, mask_hi), bcast_idx2),
                    mask_lo,
                )
                rsum_b = S.vsel(
                    S.vselr(S.vcgadd(cur_b, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(cur_b, mask_hi), bcast_idx2),
                    mask_lo,
                )
                rsum_c = S.vsel(
                    S.vselr(S.vcgadd(cur_c, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(cur_c, mask_hi), bcast_idx2),
                    mask_lo,
                )
                rsum_d = S.vsel(
                    S.vselr(S.vcgadd(cur_d, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(cur_d, mask_hi), bcast_idx2),
                    mask_lo,
                )
                cur_a = S.vdiv(cur_a, S.vadds(rsum_a, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                cur_b = S.vdiv(cur_b, S.vadds(rsum_b, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                cur_c = S.vdiv(cur_c, S.vadds(rsum_c, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                cur_d = S.vdiv(cur_d, S.vadds(rsum_d, T.float32(mhc_sinkhorn_eps), mask2), mask2)

                tr_a = S.vselr(cur_a, t_idx)
                tr_b = S.vselr(cur_b, t_idx)
                tr_c = S.vselr(cur_c, t_idx)
                tr_d = S.vselr(cur_d, t_idx)
                csum_a = S.vsel(
                    S.vselr(S.vcgadd(tr_a, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(tr_a, mask_hi), bcast_idx2),
                    mask_lo,
                )
                csum_b = S.vsel(
                    S.vselr(S.vcgadd(tr_b, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(tr_b, mask_hi), bcast_idx2),
                    mask_lo,
                )
                csum_c = S.vsel(
                    S.vselr(S.vcgadd(tr_c, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(tr_c, mask_hi), bcast_idx2),
                    mask_lo,
                )
                csum_d = S.vsel(
                    S.vselr(S.vcgadd(tr_d, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(tr_d, mask_hi), bcast_idx2),
                    mask_lo,
                )
                tr_a = S.vdiv(tr_a, S.vadds(csum_a, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                tr_b = S.vdiv(tr_b, S.vadds(csum_b, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                tr_c = S.vdiv(tr_c, S.vadds(csum_c, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                tr_d = S.vdiv(tr_d, S.vadds(csum_d, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                cur_a = S.vselr(tr_a, t_idx)
                cur_b = S.vselr(tr_b, t_idx)
                cur_c = S.vselr(tr_c, t_idx)
                cur_d = S.vselr(tr_d, t_idx)

            S.vsts(comb_out_ub[g0 * 4, 0], cur_a, mask2)
            S.vsts(comb_out_ub[(g0 + 1) * 4, 0], cur_b, mask2)
            S.vsts(comb_out_ub[(g0 + 2) * 4, 0], cur_c, mask2)
            S.vsts(comb_out_ub[(g0 + 3) * 4, 0], cur_d, mask2)


@T.macro
def _reduce_split_coefficients(
    partial_proj,
    partial_sq,
    alpha,
    bias,
    pre,
    post,
    comb,
    tokens,
    splits,
    cube_cores,
    cube_id,
    vector_id,
    flat_hidden,
):
    proj = T.alloc_shared((384,), T.float32)
    sq = T.alloc_shared((16,), T.float32)
    batch_proj = T.alloc_shared((splits, 384), T.float32)
    batch_sq = T.alloc_shared((splits, 64), T.float32)
    sq_offset = 0
    scale_ub = T.alloc_shared((3, 8), T.float32)
    base_ub = T.alloc_shared((24, 8), T.float32)
    mixes = T.alloc_shared((384,), T.float32)
    pre_tile = T.alloc_shared((16, 4), T.float32)
    post_tile = T.alloc_shared((16, 4), T.float32)
    comb_in = T.alloc_shared((16, 16), T.float32)
    comb_out = T.alloc_shared((16, 16), T.float32)
    first_start = T.min(cube_id, tokens // 32 - 1) * 16
    first_output = first_start // 64 * 128 + vector_id * 64 + first_start % 64
    with T.PerCoreTask():
        T.ascend_sync_inter_wait('PIPE_MTE2', 8)
        with T.Task():
            raw_rows = T.view(partial_proj, (splits, tokens * 24))
            first_raw = first_output * 24
            T.copy(raw_rows[:, first_raw : first_raw + 384], batch_proj[:, 0:384])
            T.copy(
                partial_sq[:, first_output : first_output + 16],
                batch_sq[:, sq_offset : sq_offset + 16],
            )
    with T.SimtVF(threads=32):
        for i in T.Parallel(3):
            scale_ub[i, 0] = alpha[i]
        for i in T.Parallel(24):
            base_ub[i, 0] = bias[i]
    jobs = T.ceildiv(T.max(tokens // 32 - cube_id, 0), cube_cores)
    for job in T.Serial(jobs):
        start = (cube_id + job * cube_cores) * 16
        output_start = start // 64 * 128 + vector_id * 64 + start % 64
        if job != 0:
            raw_begin = output_start * 24
            T.copy(raw_rows[:, raw_begin : raw_begin + 384], batch_proj[:, 0:384])
            T.copy(
                partial_sq[:, output_start : output_start + 16],
                batch_sq[:, sq_offset : sq_offset + 16],
            )
        with T.SimdVF():
            mask = S.pset(32)
            sq_mask = S.vcmps(S.vci(T.int32(0), T.int32), T.int32(16), mask, 'lt')
            for vec in T.unroll(6, explicit=True):
                offset = vec * 64
                sum_proj = S.alloc_var(T.float32)
                sum_proj = S.vld(batch_proj[0, offset])
                for k in T.unroll(splits - 1, explicit=True):
                    sum_proj = S.vadd(sum_proj, S.vld(batch_proj[k + 1, offset]), mask)
                S.vsts(proj[offset], sum_proj, mask)
            sum_sq = S.alloc_var(T.float32)
            sum_sq = S.vld(batch_sq[0, sq_offset])
            for k in T.unroll(splits - 1, explicit=True):
                sum_sq = S.vadd(sum_sq, S.vld(batch_sq[k + 1, sq_offset]), sq_mask)
            S.vsts(sq[0], sum_sq, sq_mask, extent=16)
        compute_coefficients(
            proj,
            sq,
            scale_ub,
            base_ub,
            mixes,
            pre_tile,
            post_tile,
            comb_in,
            comb_out,
            0,
            0,
            16,
            4,
            16,
            24,
            T.float32(1.0 / flat_hidden),
            1e-20,
            1e-06,
            1e-06,
            2.0,
            20,
        )
        T.copy(pre_tile, pre[output_start : output_start + 16, 0:4])
        T.copy(post_tile, post[output_start : output_start + 16, 0:4])
        T.copy(comb_out, comb[output_start : output_start + 16, 0:16])


@T.macro
def _projection_fused_body(
    x,
    fn,
    out_mul,
    sqrsum,
    alpha,
    bias,
    pre,
    post,
    comb,
    num_tokens,
    mhc_mult3,
    flat_hidden,
    block_mhc_mult3,
    num_aic_cores,
    num_k_splits,
    cores_per_split,
    num_hidden_blocks_per_split,
    num_l0b_stages,
    num_l0c_stages,
    num_token_blocks_per_chunk,
):
    block_hidden = 256
    mad_hidden = 64
    num_aivs = 2
    half_block_num_tokens = 128 // num_aivs
    num_mads_per_block = block_hidden // mad_hidden
    num_load_stages = 2
    num_cast_stages = 2
    num_l1_x_stages = 3
    num_mad_stages = 2
    nz_stage_rows = half_block_num_tokens + 1
    with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
        split_id = core_id // cores_per_split
        core_in_split = core_id % cores_per_split
        num_token_blocks = T.ceildiv(num_tokens, 128)
        num_token_chunks = T.ceildiv(num_token_blocks, num_token_blocks_per_chunk)
        x_ub = T.alloc_shared((half_block_num_tokens, block_hidden), T.bfloat16)
        x_nz_ub = T.alloc_shared((nz_stage_rows, block_hidden), T.float32)
        T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})
        sqrsum_ub = T.alloc_shared((num_l0c_stages, 64), T.float32)
        partial_ub = T.alloc_shared((num_l0c_stages, 64, 24), T.float32)
        x_l1 = T.alloc_l1((128, block_hidden), T.float32)
        fn_l1 = T.alloc_l1((block_mhc_mult3, block_hidden), T.float32)
        x_l0 = T.alloc_l0a((128, mad_hidden), T.float32)
        fn_l0 = T.alloc_l0b((num_mads_per_block, block_mhc_mult3, mad_hidden), T.float32)
        out_mul_l0 = T.alloc_l0c((num_l0c_stages, 128, block_mhc_mult3), T.float32)
        T.annotate_buffer_versions(
            {
                x_ub: num_load_stages,
                x_nz_ub: num_cast_stages,
                x_l1: num_l1_x_stages,
                fn_l1: num_l0b_stages,
                x_l0: num_mad_stages,
                fn_l0: num_l0b_stages,
            }
        )
        T.set_hf32_mode('nearest_even')
        for chunk_idx in T.Serial(num_token_chunks):
            token_block_base = chunk_idx * num_token_blocks_per_chunk + core_in_split
            num_valid_token_stages = T.min(
                num_l0c_stages,
                T.max(T.ceildiv(num_token_blocks - token_block_base, cores_per_split), 0),
            )
            aiv_token_begin_base = token_block_base * 128 + aiv_id * half_block_num_tokens
            num_valid_sqrsum_stages = T.min(
                num_l0c_stages,
                T.max(T.ceildiv(num_tokens - aiv_token_begin_base, cores_per_split * 128), 0),
            )
            for local_hidden_block in T.Pipelined(num_hidden_blocks_per_split, num_stages=2):
                global_hidden_block = split_id * num_hidden_blocks_per_split + local_hidden_block
                hidden_begin = global_hidden_block * block_hidden
                T.copy(
                    fn[0:mhc_mult3, hidden_begin : hidden_begin + block_hidden],
                    fn_l1[0:mhc_mult3, :],
                )
                for hidden_mad in T.Serial(num_mads_per_block):
                    T.copy(
                        fn_l1[0:mhc_mult3, hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden],
                        fn_l0[hidden_mad, 0:mhc_mult3, :],
                    )
                for token_stage in T.Pipelined(
                    num_valid_token_stages,
                    num_stages=2,
                    annotations={'multi_buffer_eligible': [x_l1]},
                ):
                    token_block = token_block_base + token_stage * cores_per_split
                    token_begin = token_block * 128
                    actual_num_tokens = T.min(num_tokens - token_begin, 128)
                    aiv_token_begin = token_begin + aiv_id * half_block_num_tokens
                    if aiv_token_begin < num_tokens:
                        actual_aiv_num_tokens = T.min(
                            num_tokens - aiv_token_begin, half_block_num_tokens
                        )
                        T.copy(
                            x[
                                aiv_token_begin : aiv_token_begin + actual_aiv_num_tokens,
                                hidden_begin : hidden_begin + block_hidden,
                            ],
                            x_ub[0:actual_aiv_num_tokens, :],
                            l2_cache_ctrl=None,
                            pad_value=0,
                        )
                        with T.SimdVF(latency=640):
                            bf16_mask = T.simd.pset(16)
                            f32_mask = T.simd.pset(32)
                            nz_stride = T.int32(nz_stage_rows << 16 | 8 * nz_stage_rows)
                            zero_bf16 = T.simd.vdup(T.bfloat16(0), 'bfloat16', bf16_mask)
                            row_indices = T.simd.vci(T.float32(0), 'float32')
                            current_row = T.simd.alloc_var('float32')
                            current_row = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                            one = T.simd.vdup(T.float32(1), 'float32', f32_mask)
                            sqr_sums = T.simd.alloc_var('float32')
                            sqr_sums = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                            for row in T.Serial(half_block_num_tokens):
                                row_acc = T.simd.alloc_var('float32')
                                row_acc = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                                nz_ptr = T.simd.make_ubuf_ptr(
                                    T.access_ptr(x_nz_ub[row, 0], 'w', 1, block_hidden), 'float32'
                                )
                                for pass_id in T.Serial(block_hidden // 128):
                                    packed = T.simd.vld(x_ub[row, pass_id * 128])
                                    lo_bits, hi_bits = T.simd.vintlv(zero_bf16, packed)
                                    unpacked_lo = T.reinterpret(lo_bits, 'float32x64')
                                    unpacked_hi = T.reinterpret(hi_bits, 'float32x64')
                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(unpacked_lo, unpacked_lo, f32_mask),
                                        f32_mask,
                                    )
                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(unpacked_hi, unpacked_hi, f32_mask),
                                        f32_mask,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        unpacked_lo, nz_ptr, nz_stride, f32_mask, update=True
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        unpacked_hi, nz_ptr, nz_stride, f32_mask, update=True
                                    )
                                row_sum = T.simd.alloc_var('float32')
                                row_sum = T.simd.vcadd(row_acc, f32_mask)
                                row_sum_broadcast = T.simd.vdupv(row_sum, f32_mask)
                                row_mask = T.simd.vcmp(row_indices, current_row, f32_mask, 'eq')
                                sqr_sums = T.simd.vsel(row_sum_broadcast, sqr_sums, row_mask)
                                current_row = T.simd.vadd(current_row, one, f32_mask)
                            if local_hidden_block != 0:
                                previous = T.simd.vld(sqrsum_ub[token_stage, 0])
                                sqr_sums = T.simd.vadd(sqr_sums, previous, f32_mask)
                            T.simd.vsts(sqrsum_ub[token_stage, 0], sqr_sums, f32_mask)
                        T.dual_copy(
                            x_nz_ub[0:half_block_num_tokens, 0:block_hidden],
                            x_l1[0:128, 0:block_hidden],
                        )
                    for hidden_mad in T.Serial(num_mads_per_block):
                        T.copy(
                            x_l1[
                                0:actual_num_tokens,
                                hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden,
                            ],
                            x_l0[0:actual_num_tokens, :],
                        )
                        T.gemm(
                            x_l0[0:actual_num_tokens, :],
                            fn_l0[hidden_mad, 0:mhc_mult3, :],
                            out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                            transpose_B=True,
                            clear_accum=local_hidden_block == 0 and hidden_mad == 0,
                            unit_flag_ctrl=T.Select(
                                local_hidden_block == num_hidden_blocks_per_split - 1
                                and hidden_mad == num_mads_per_block - 1,
                                3,
                                2,
                            ),
                        )
            for drain_slot in T.Serial(num_valid_token_stages):
                T.dual_copy(
                    out_mul_l0[drain_slot, 0:128, 0:24],
                    partial_ub[drain_slot, 0:64, 0:24],
                    unit_flag_ctrl=3,
                )
            with T.PerCoreTask():
                with T.Task():
                    for partial_slot in T.Serial(num_valid_token_stages):
                        partial_token = aiv_token_begin_base + partial_slot * cores_per_split * 128
                        T.copy(
                            partial_ub[partial_slot, 0:64, 0:24],
                            out_mul[split_id, partial_token : partial_token + 64, 0:24],
                        )
                    for sum_slot in T.Serial(num_valid_sqrsum_stages):
                        sum_token = aiv_token_begin_base + sum_slot * cores_per_split * 128
                        T.copy(
                            sqrsum_ub[sum_slot, 0:64], sqrsum[split_id, sum_token : sum_token + 64]
                        )
                for _ in T.Serial(T.if_then_else(chunk_idx + 1 == num_token_chunks, 1, 0)):
                    T.ascend_sync_inter_arrive('PIPE_MTE3', 8)
        _reduce_split_coefficients(
            out_mul,
            sqrsum,
            alpha,
            bias,
            pre,
            post,
            comb,
            num_tokens,
            num_k_splits,
            num_aic_cores,
            core_id,
            aiv_id,
            flat_hidden,
        )


@tilelang.jit
def mhc_projection_coeff_k8_kernel(
    x,
    fn,
    alpha,
    bias,
    pre,
    post,
    comb,
    partial_proj,
    partial_sq,
    block_mhc_mult3: int,
    num_aic_cores: int,
    num_k_splits: int,
    cores_per_split: int,
    num_hidden_blocks_per_split: int,
    num_l0b_stages: int,
    num_l0c_stages: int,
    num_token_blocks_per_chunk: int,
):
    tokens = T.const('tokens')
    flat_hidden = T.const('flat_hidden')
    x: T.Tensor[(tokens, flat_hidden), T.bfloat16]
    fn: T.Tensor[(24, flat_hidden), T.float32]
    alpha: T.Tensor[(3,), T.float32]
    bias: T.Tensor[(24,), T.float32]
    pre: T.Tensor[(tokens, 4), T.float32]
    post: T.Tensor[(tokens, 4), T.float32]
    comb: T.Tensor[(tokens, 16), T.float32]
    partial_proj: T.Tensor[(8, tokens, 24), T.float32]
    partial_sq: T.Tensor[(8, tokens), T.float32]
    assert num_k_splits == 8 and num_l0c_stages == 1
    assert tokens >= cores_per_split * 128
    assert tokens % (cores_per_split * 128) == 0
    assert flat_hidden == num_hidden_blocks_per_split * num_k_splits * 256
    _projection_fused_body(
        x,
        fn,
        partial_proj,
        partial_sq,
        alpha,
        bias,
        pre,
        post,
        comb,
        tokens,
        24,
        flat_hidden,
        block_mhc_mult3,
        num_aic_cores,
        num_k_splits,
        cores_per_split,
        num_hidden_blocks_per_split,
        num_l0b_stages,
        num_l0c_stages,
        num_token_blocks_per_chunk,
    )


@T.macro
def _projection_three_body(
    x,
    fn,
    out_mul,
    sqrsum,
    num_tokens,
    mhc_mult3,
    block_mhc_mult3,
    num_aic_cores,
    num_k_splits,
    cores_per_split,
    num_hidden_blocks_per_split,
    num_l0b_stages,
    num_l0c_stages,
    num_token_blocks_per_chunk,
):
    block_num_tokens = 128
    block_hidden = 256
    mad_hidden = 64
    half_block_num_tokens = block_num_tokens // 2
    num_mads_per_block = block_hidden // mad_hidden
    nz_stage_rows = half_block_num_tokens + 1

    with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
        split_id = core_id // cores_per_split
        core_in_split = core_id % cores_per_split
        num_token_blocks = T.ceildiv(num_tokens, block_num_tokens)
        num_token_chunks = T.ceildiv(num_token_blocks, num_token_blocks_per_chunk)

        x_ub = T.alloc_shared((half_block_num_tokens, block_hidden), T.bfloat16)
        x_nz_ub = T.alloc_shared((nz_stage_rows, block_hidden), T.float32)
        T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})
        sqrsum_ub = T.alloc_shared((num_l0c_stages, half_block_num_tokens), T.float32)

        x_l1 = T.alloc_l1((block_num_tokens, block_hidden), T.float32)
        fn_l1 = T.alloc_l1((block_mhc_mult3, block_hidden), T.float32)
        x_l0 = T.alloc_l0a((block_num_tokens, mad_hidden), T.float32)
        fn_l0 = T.alloc_l0b((num_mads_per_block, block_mhc_mult3, mad_hidden), T.float32)
        out_mul_l0 = T.alloc_l0c((num_l0c_stages, block_num_tokens, block_mhc_mult3), T.float32)

        T.annotate_buffer_versions(
            {
                x_ub: 3,
                x_nz_ub: 2,
                x_l1: 3,
                fn_l1: num_l0b_stages,
                x_l0: 2,
                fn_l0: num_l0b_stages,
            }
        )

        T.set_hf32_mode('nearest_even')
        for chunk_idx in T.Serial(num_token_chunks):
            token_block_base = chunk_idx * num_token_blocks_per_chunk + core_in_split
            num_valid_token_stages = T.min(
                num_l0c_stages,
                T.max(T.ceildiv(num_token_blocks - token_block_base, cores_per_split), 0),
            )
            aiv_token_begin_base = (
                token_block_base * block_num_tokens + aiv_id * half_block_num_tokens
            )
            num_valid_sqrsum_stages = T.min(
                num_l0c_stages,
                T.max(
                    T.ceildiv(
                        num_tokens - aiv_token_begin_base,
                        cores_per_split * block_num_tokens,
                    ),
                    0,
                ),
            )

            for local_hidden_block in T.Pipelined(num_hidden_blocks_per_split, num_stages=2):
                global_hidden_block = split_id * num_hidden_blocks_per_split + local_hidden_block
                hidden_begin = global_hidden_block * block_hidden

                # Keep physical buffers block_mhc_mult3-aligned, but restrict
                # data movement and MAD regions to the logical mhc_mult3.
                T.copy(
                    fn[0:mhc_mult3, hidden_begin : hidden_begin + block_hidden],
                    fn_l1[0:mhc_mult3, :],
                )
                for hidden_mad in T.Serial(num_mads_per_block):
                    T.copy(
                        fn_l1[0:mhc_mult3, hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden],
                        fn_l0[hidden_mad, 0:mhc_mult3, :],
                    )

                for token_stage in T.Pipelined(
                    num_valid_token_stages,
                    num_stages=2,
                    annotations={'multi_buffer_eligible': [x_l1]},
                ):
                    token_block = token_block_base + token_stage * cores_per_split
                    token_begin = token_block * block_num_tokens
                    actual_num_tokens = T.min(num_tokens - token_begin, block_num_tokens)
                    aiv_token_begin = token_begin + aiv_id * half_block_num_tokens

                    if aiv_token_begin < num_tokens:
                        actual_aiv_num_tokens = T.min(
                            num_tokens - aiv_token_begin,
                            half_block_num_tokens,
                        )
                        T.copy(
                            x[
                                aiv_token_begin : aiv_token_begin + actual_aiv_num_tokens,
                                hidden_begin : hidden_begin + block_hidden,
                            ],
                            x_ub[0:actual_aiv_num_tokens, :],
                            l2_cache_ctrl=None,
                            pad_value=0,
                        )

                        # Widen one AIV half-tile, pack padded NZ, and update row sums.
                        with T.SimdVF(latency=640):
                            bf16_mask = T.simd.pset(16)
                            f32_mask = T.simd.pset(32)
                            nz_stride = T.int32((nz_stage_rows << 16) | (8 * nz_stage_rows))
                            zero_bf16 = T.simd.vdup(T.bfloat16(0), 'bfloat16', bf16_mask)
                            row_indices = T.simd.vci(T.float32(0), 'float32')
                            current_row = T.simd.alloc_var('float32')
                            current_row = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                            one = T.simd.vdup(T.float32(1), 'float32', f32_mask)
                            sqr_sums = T.simd.alloc_var('float32')
                            sqr_sums = T.simd.vdup(T.float32(0), 'float32', f32_mask)

                            for row in T.Serial(half_block_num_tokens):
                                row_acc = T.simd.alloc_var('float32')
                                row_acc = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                                nz_ptr = T.simd.make_ubuf_ptr(
                                    T.access_ptr(
                                        x_nz_ub[row, 0],
                                        'w',
                                        1,
                                        block_hidden,
                                    ),
                                    'float32',
                                )

                                for pass_id in T.Serial(block_hidden // 128):
                                    packed = T.simd.vld(x_ub[row, pass_id * 128])
                                    lo_bits, hi_bits = T.simd.vintlv(zero_bf16, packed)
                                    unpacked_lo = T.reinterpret(lo_bits, 'float32x64')
                                    unpacked_hi = T.reinterpret(hi_bits, 'float32x64')

                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(unpacked_lo, unpacked_lo, f32_mask),
                                        f32_mask,
                                    )
                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(unpacked_hi, unpacked_hi, f32_mask),
                                        f32_mask,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        unpacked_lo,
                                        nz_ptr,
                                        nz_stride,
                                        f32_mask,
                                        update=True,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        unpacked_hi,
                                        nz_ptr,
                                        nz_stride,
                                        f32_mask,
                                        update=True,
                                    )

                                row_sum = T.simd.alloc_var('float32')
                                row_sum = T.simd.vcadd(row_acc, f32_mask)
                                row_sum_broadcast = T.simd.vdupv(row_sum, f32_mask)
                                row_mask = T.simd.vcmp(
                                    row_indices,
                                    current_row,
                                    f32_mask,
                                    'eq',
                                )
                                sqr_sums = T.simd.vsel(row_sum_broadcast, sqr_sums, row_mask)
                                current_row = T.simd.vadd(current_row, one, f32_mask)

                            if local_hidden_block != 0:
                                previous = T.simd.vld(sqrsum_ub[token_stage, 0])
                                sqr_sums = T.simd.vadd(sqr_sums, previous, f32_mask)
                            T.simd.vsts(sqrsum_ub[token_stage, 0], sqr_sums, f32_mask)

                        T.dual_copy(
                            x_nz_ub[0:half_block_num_tokens, 0:block_hidden],
                            x_l1[0:block_num_tokens, 0:block_hidden],
                        )

                    for hidden_mad in T.Serial(num_mads_per_block):
                        T.copy(
                            x_l1[
                                0:actual_num_tokens,
                                hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden,
                            ],
                            x_l0[0:actual_num_tokens, :],
                        )
                        T.gemm(
                            x_l0[0:actual_num_tokens, :],
                            fn_l0[hidden_mad, 0:mhc_mult3, :],
                            out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                            transpose_B=True,
                            clear_accum=(local_hidden_block == 0 and hidden_mad == 0),
                            unit_flag_ctrl=T.Select(
                                local_hidden_block == num_hidden_blocks_per_split - 1
                                and hidden_mad == num_mads_per_block - 1,
                                3,
                                2,
                            ),
                        )

            for partial_slot in T.Serial(num_valid_token_stages):
                partial_token = (
                    token_block_base + partial_slot * cores_per_split
                ) * block_num_tokens
                T.copy(
                    out_mul_l0[partial_slot, 0:128, 0:24],
                    out_mul[split_id, partial_token : partial_token + 128, 0:24],
                    unit_flag_ctrl=3,
                )
            for sum_slot in T.Serial(num_valid_sqrsum_stages):
                sum_token = aiv_token_begin_base + sum_slot * cores_per_split * block_num_tokens
                T.copy(sqrsum_ub[sum_slot, 0:64], sqrsum[split_id, sum_token : sum_token + 64])


@tilelang.jit
def mhc_projection_split_partials(
    x,
    fn,
    out_mul,
    sqrsum,
    block_mhc_mult3: int,
    num_aic_cores: int,
    num_k_splits: int,
    cores_per_split: int,
    num_hidden_blocks_per_split: int,
    num_l0b_stages: int,
    num_l0c_stages: int,
    num_token_blocks_per_chunk: int,
):
    tokens = T.dynamic('tokens')
    flat_hidden = T.const('flat_hidden')
    x: T.Tensor[(tokens, flat_hidden), T.bfloat16]
    fn: T.Tensor[(24, flat_hidden), T.float32]
    out_mul: T.Tensor[(num_k_splits, tokens, 24), T.float32]
    sqrsum: T.Tensor[(num_k_splits, tokens), T.float32]
    _projection_three_body(
        x,
        fn,
        out_mul,
        sqrsum,
        tokens,
        24,
        block_mhc_mult3,
        num_aic_cores,
        num_k_splits,
        cores_per_split,
        num_hidden_blocks_per_split,
        num_l0b_stages,
        num_l0c_stages,
        num_token_blocks_per_chunk,
    )


@tilelang.jit
def mhc_coeff_token_pipeline(
    gemm_out_mul,
    gemm_out_sqrsum,
    mhc_scale,
    mhc_base,
    pre_mix,
    post_mix,
    comb_mix,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    hidden: int,
    n_splits: int,
    token_block: int,
    num_stages: int,
    num_vec_cores: int,
):
    """RMS + split + Sinkhorn; emit next_pre/post/comb, with no feature apply."""
    num_tokens = T.dynamic('num_tokens')
    mhc = 4
    mhc2 = mhc * mhc
    mhc3 = mhc * 2 + mhc2
    assert mhc3 == 24

    gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc3), T.float32]
    gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[(mhc3,), T.float32]
    pre_mix: T.Tensor[(num_tokens, mhc), T.float32]
    post_mix: T.Tensor[(num_tokens, mhc), T.float32]
    comb_mix: T.Tensor[(num_tokens, mhc2), T.float32]

    tb = token_block
    assert token_block == 16, "Initial Q4 fusion candidate uses a 16-token tile"
    # Same compile-time fp32 constant as the reference reduce kernel.
    inv_rms_group_size = T.float32(1.0 / (mhc * hidden))

    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((3, 8), T.float32)
        base_ub = T.alloc_shared((mhc3, 8), T.float32)
        mul_flat_ub = T.alloc_shared((tb * mhc3,), T.float32)
        mixes_flat_ub = T.alloc_shared((tb * mhc3,), T.float32)
        sq_ub = T.alloc_shared((tb,), T.float32)
        pre_ub = T.alloc_shared((tb, mhc), T.float32)
        post_ub = T.alloc_shared((tb, mhc), T.float32)
        comb_in_ub = T.alloc_shared((tb, mhc2), T.float32)
        comb_out_ub = T.alloc_shared((tb, mhc2), T.float32)
        batch_mul_ub = T.alloc_shared((n_splits, tb * mhc3), T.float32)
        # Pad the UB row to a full SIMD vector. DMA and arithmetic use
        # only the first tb lanes.
        batch_sq_ub = T.alloc_shared((n_splits, 64), T.float32)
        T.annotate_buffer_versions(
            {
                mul_flat_ub: num_stages,
                mixes_flat_ub: num_stages,
                sq_ub: num_stages,
                pre_ub: num_stages,
                post_ub: num_stages,
                comb_in_ub: num_stages,
                comb_out_ub: num_stages,
            }
        )

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = mhc_scale[i]
            for i in T.Parallel(mhc3):
                base_ub[i, 0] = mhc_base[i]

        jobs = T.ceildiv(T.max(T.ceildiv(num_tokens, tb) - core_id, 0), num_vec_cores)
        for step in T.Pipelined(jobs, num_stages=num_stages):
            pid = core_id + step * num_vec_cores
            token_base = pid * tb
            raw_rows = T.view(gemm_out_mul, (n_splits, num_tokens * mhc3))
            raw_begin = token_base * mhc3
            T.copy(raw_rows[0:n_splits, raw_begin : raw_begin + tb * mhc3], batch_mul_ub)
            T.copy(
                gemm_out_sqrsum[0:n_splits, token_base : token_base + tb],
                batch_sq_ub[0:n_splits, 0:tb],
            )
            with T.SimdVF():
                mask = S.pset(32)
                sq_mask = S.vcmps(S.vci(T.int32(0), T.int32), T.int32(tb), mask, 'lt')
                for vec in T.unroll(tb * mhc3 // 64, explicit=True):
                    offset = vec * 64
                    sum_mul = S.alloc_var(T.float32)
                    sum_mul = S.vld(batch_mul_ub[0, offset])
                    for k in T.unroll(n_splits - 1, explicit=True):
                        sum_mul = S.vadd(sum_mul, S.vld(batch_mul_ub[k + 1, offset]), mask)
                    S.vsts(mul_flat_ub[offset], sum_mul, mask)
                sum_sq = S.alloc_var(T.float32)
                sum_sq = S.vld(batch_sq_ub[0, 0])
                for k in T.unroll(n_splits - 1, explicit=True):
                    sum_sq = S.vadd(sum_sq, S.vld(batch_sq_ub[k + 1, 0]), sq_mask)
                S.vsts(sq_ub[0], sum_sq, sq_mask, extent=tb)

            ##################################################################
            # _mhc_reduce_partials_and_rmsnorm_fwd + _mhc_pre_split_mixes_fwd
            # + _mhc_sinkhorn_fwd in a single SimdVF phase: the whole latency
            # path is one unit so the scheduler can overlap it with the
            # pre-apply memory stream of the previous token block.
            compute_coefficients(
                mul_flat_ub,
                sq_ub,
                scale_ub,
                base_ub,
                mixes_flat_ub,
                pre_ub,
                post_ub,
                comb_in_ub,
                comb_out_ub,
                0,
                0,
                tb,
                mhc,
                mhc2,
                mhc3,
                inv_rms_group_size,
                rms_eps,
                mhc_pre_eps,
                mhc_sinkhorn_eps,
                mhc_post_mult_value,
                sinkhorn_repeat,
            )

            T.copy(pre_ub[:, 0:mhc], pre_mix[token_base : token_base + tb, 0:mhc])
            T.copy(post_ub[:, 0:mhc], post_mix[token_base : token_base + tb, 0:mhc])
            T.copy(comb_out_ub, comb_mix[token_base : token_base + tb, 0:mhc2])


_VEC = 64


@tilelang.jit()
def mhc_pre_apply_flat_pipeline_asc(x, mix, o, h_blk, num_vec_cores: int, pipeline_stages: int):
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

        # Exact per-core job count avoids invalid token loads without a conditional
        # around the pipelined body; cores with no tokens have zero iterations.
        jobs = T.ceildiv(T.max(num_tokens - core_id, 0), num_vec_cores)
        for step in T.Pipelined(jobs * num_hidden_tiles, num_stages=num_stages):
            pid_n = core_id + (step // num_hidden_tiles) * num_vec_cores
            pid_h = step % num_hidden_tiles
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


def bind_shifted_pre(x, prev_pre, phi, alpha, bias):
    """Compile and allocate once, selecting two kernels below N=512."""
    import torch
    from deep_gemm import _C
    from tile_kernels.config import get_num_vec_cores

    if len(x.shape) != 3 or tuple(x.shape[1:]) != (4, 5120):
        raise ValueError('Expected residual[N,4,5120]')
    n, streams, hidden = x.shape
    if n <= 0 or n % 128:
        raise ValueError('N must be a positive multiple of 128')
    if tuple(prev_pre.shape) not in ((n, 4), (n, 4, 1)):
        raise ValueError('Expected prev_pre[N,4] or prev_pre[N,4,1]')
    for tensor, shape, dtype in (
        (x, (n, 4, 5120), torch.bfloat16),
        (prev_pre, tuple(prev_pre.shape), torch.float32),
        (phi, (24, 20480), torch.float32),
        (alpha, (3,), torch.float32),
        (bias, (24,), torch.float32),
    ):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != x.device:
            raise ValueError('Input shape, dtype or device mismatch')
        if not tensor.is_contiguous():
            raise ValueError('Inputs must be contiguous')

    groups = min(_C.get_num_sms() // 8, n // 128)
    if groups == 0:
        raise ValueError('K8 projection requires at least eight Cube cores')
    small = n < 512
    if small:
        while (n // 128) % groups:
            groups -= 1
    l0c_stages = 1 if small else 16
    config = dict(
        block_mhc_mult3=32,
        num_aic_cores=8 * groups,
        num_k_splits=8,
        cores_per_split=groups,
        num_hidden_blocks_per_split=10,
        num_l0b_stages=2,
        num_l0c_stages=l0c_stages,
        num_token_blocks_per_chunk=groups * l0c_stages,
    )
    vec_cores = get_num_vec_cores()
    flat, prev = x.flatten(1), prev_pre.view(n, 4)
    hi = torch.empty(n, hidden, dtype=x.dtype, device=x.device)
    packed = torch.empty(n * 24, dtype=torch.float32, device=x.device)
    pre, post, comb = packed.split((n * 4, n * 4, n * 16))
    pre, post, comb = pre.view(n, 4), post.view(n, 4), comb.view(n, 16)
    raw = torch.empty(8, n, 24, dtype=torch.float32, device=x.device)
    sq = torch.empty(8, n, dtype=torch.float32, device=x.device)
    outputs = hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb.view(n, 4, 4)
    apply = mhc_pre_apply_flat_pipeline_asc.compile(
        x, prev, hi, h_blk=hidden, num_vec_cores=vec_cores, pipeline_stages=3
    )
    sources = {'apply': apply.get_kernel_source()}

    if small:
        fused = mhc_projection_coeff_k8_kernel.compile(
            flat, phi, alpha, bias, pre, post, comb, raw, sq, **config
        )
        sources['projection_coeff'] = fused.get_kernel_source()

        def run():
            apply(x, prev, hi)
            fused(flat, phi, alpha, bias, pre, post, comb, raw, sq)
            return outputs

    else:
        projection = mhc_projection_split_partials.compile(flat, phi, raw, sq, **config)
        coefficients = mhc_coeff_token_pipeline.compile(
            raw,
            sq,
            alpha,
            bias,
            pre,
            post,
            comb,
            rms_eps=1e-20,
            mhc_pre_eps=1e-6,
            mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=20,
            hidden=hidden,
            n_splits=8,
            token_block=16,
            num_stages=2,
            num_vec_cores=vec_cores,
        )
        sources.update(
            projection=projection.get_kernel_source(), tail=coefficients.get_kernel_source()
        )

        def run():
            apply(x, prev, hi)
            projection(flat, phi, raw, sq)
            coefficients(raw, sq, alpha, bias, pre, post, comb)
            return outputs

    implementation = 'two-kernel K8' if small else 'three-kernel K8'
    return run, dict(
        dispatch=dict(tokens=n, threshold=512, implementation=implementation),
        expected_kernels_per_call=2 if small else 3,
        projection_config=config,
        output_preallocation=True,
        workspace_reuse=True,
        per_call_allocations=0,
        generated_sources=sources,
    )

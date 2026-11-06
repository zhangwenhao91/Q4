# SPDX-License-Identifier: MIT
# Copyright (c) 2026 DeepSeek. Portions adapted for Q4 shifted mHC.
"""H5120 shifted mHC Pre with three TileLang kernels and a bound launch path.

The projection is adapted from DeepGEMM-Ascend 3a353262 (LICENSE_DEEPGEMM).
The coefficient and Apply code is adapted from DeepSeek TileKernels c1cf914
(SPDX-License-Identifier: MIT; see LICENSE_TILEKERNELS and source.json).
"""
import hashlib
import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout


def get_config(mhc_hidden_size: int, mhc_mult3: int, num_tokens: int, deterministic: bool = False, *, token_owned=False) -> dict:
    """Return optimal kernel parameters for a given shape."""
    from deep_gemm import _C
    block_num_tokens = 128
    block_hidden = 256
    l0b_size_bytes = 64 * 1024
    l0c_size_bytes = 256 * 1024

    assert num_tokens > 0, f'HC prenorm fwd requires positive num_tokens, got {num_tokens}'
    assert 0 < mhc_mult3 <= 64 and mhc_mult3 % 8 == 0, f'HC prenorm fwd requires mhc_mult3 to be 8-aligned and no larger than 64, got {mhc_mult3}'
    assert mhc_hidden_size > 0 and mhc_hidden_size % block_hidden == 0, (
        f'HC prenorm fwd requires mhc_hidden_size to be a positive multiple of {block_hidden}, got {mhc_hidden_size}'
    )

    num_aic_cores = _C.get_num_sms()
    block_mhc_mult3 = (mhc_mult3 + 15) // 16 * 16
    num_hidden_blocks = mhc_hidden_size // block_hidden
    num_token_blocks = math.ceil(num_tokens / block_num_tokens)
    num_l0b_stages = l0b_size_bytes // (block_mhc_mult3 * block_hidden * 4)
    num_l0c_stages = l0c_size_bytes // (block_num_tokens * block_mhc_mult3 * 4)

    # K partitioning defines the per-row floating-point reduction.
    # M-dependent split selection improves L0C residency/B reuse,
    # but changes the reduction grouping and breaks batch invariance.
    num_k_splits_by_workload = (
        16 if deterministic else max(1, num_aic_cores * num_l0c_stages // num_token_blocks)
    )
    num_k_splits_by_hidden = num_hidden_blocks & -num_hidden_blocks
    num_k_splits_by_cores = num_aic_cores & -num_aic_cores
    split_limit = min(num_k_splits_by_workload, num_k_splits_by_hidden, num_k_splits_by_cores)
    num_k_splits = 1 if token_owned else 1 << (split_limit.bit_length() - 1)

    active_aic_cores = num_k_splits * min(num_aic_cores // num_k_splits, num_token_blocks)
    cores_per_split = active_aic_cores // num_k_splits
    num_hidden_blocks_per_split = num_hidden_blocks // num_k_splits
    num_token_blocks_per_chunk = cores_per_split * num_l0c_stages

    return {
        'block_mhc_mult3': block_mhc_mult3,
        'num_aic_cores': active_aic_cores,
        'num_k_splits': num_k_splits,
        'cores_per_split': cores_per_split,
        'num_hidden_blocks_per_split': num_hidden_blocks_per_split,
        'num_l0b_stages': num_l0b_stages,
        'num_l0c_stages': num_l0c_stages,
        'num_token_blocks_per_chunk': num_token_blocks_per_chunk,
    }

@T.macro
def _projection_body(
    x, fn, out_mul, sqrsum, prev_pre, alpha, bias, hi, pre, post, comb,
    num_tokens, mhc_mult3, mhc_hidden_size,
    block_mhc_mult3, num_aic_cores, num_k_splits, cores_per_split,
    num_hidden_blocks_per_split, num_l0b_stages, num_l0c_stages,
    num_token_blocks_per_chunk, deterministic, residual_l2_cache_ctrl, fused_shifted, epilogue,
    reuse_apply=False,
):
    block_num_tokens = 128
    block_hidden = 256
    mad_hidden = 64
    num_aivs = 2
    half_block_num_tokens = block_num_tokens // num_aivs
    num_mads_per_block = block_hidden // mad_hidden
    num_load_stages = 2 if fused_shifted or reuse_apply else 3
    num_cast_stages = 1 if reuse_apply else 2
    num_l1_x_stages = 3
    num_mad_stages = 2

    num_store_phases = num_k_splits if deterministic else (2 if num_k_splits > 1 else 1)
    # In non-deterministic mode, phase 0 separates split 0's non-atomic store
    # from the remaining splits' atomic adds. The last phase only needs a
    # barrier when another token chunk follows. Encode that optional barrier
    # as a uniform 0/1-trip loop because AutoSchedule rejects inter-core waits
    # under conditional control flow inside T.PerCoreTask.
    num_always_synced_store_phases = num_store_phases if deterministic else (1 if num_k_splits > 1 else 0)
    num_chunk_end_syncs = 0 if deterministic or num_k_splits == 1 else 1
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
        if reuse_apply:
            # Only one token tile is resident while its four-stream sum is live.
            old_mix_ub = T.alloc_shared((64, 4), T.float32)
            apply_acc = T.alloc_shared((64, 256), T.float32)
            apply_out = T.alloc_shared((64, 256), T.bfloat16)
            partial_ub = T.alloc_shared((64, 24), T.float32)
            T.annotate_buffer_versions({apply_acc: 1, apply_out: 1})

        x_l1 = T.alloc_l1((block_num_tokens, block_hidden), T.float32)
        fn_l1 = T.alloc_l1((block_mhc_mult3, block_hidden), T.float32)
        x_l0 = T.alloc_l0a((block_num_tokens, mad_hidden), T.float32)
        fn_l0 = T.alloc_l0b((num_mads_per_block, block_mhc_mult3, mad_hidden), T.float32)
        out_mul_l0 = T.alloc_l0c((num_l0c_stages, block_num_tokens, block_mhc_mult3), T.float32)

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

        if not reuse_apply:
            if split_id != 0:
                T.set_atomic('add', 'float32')
        T.set_hf32_mode('nearest_even')
        for chunk_idx in T.Serial(num_token_chunks):
            token_block_base = chunk_idx * num_token_blocks_per_chunk + core_in_split
            num_valid_token_stages = T.min(
                num_l0c_stages,
                T.max(T.ceildiv(num_token_blocks - token_block_base, cores_per_split), 0),
            )
            aiv_token_begin_base = token_block_base * block_num_tokens + aiv_id * half_block_num_tokens
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
            if reuse_apply:
                if num_valid_token_stages > 0:
                    T.copy(prev_pre[aiv_token_begin_base : aiv_token_begin_base + 64, 0:4], old_mix_ub)

            for local_hidden_block in T.Pipelined(num_hidden_blocks_per_split, num_stages=2):
                if reuse_apply:
                    stream_idx = local_hidden_block % 4
                    hidden_band = split_id * (num_hidden_blocks_per_split // 4) + local_hidden_block // 4
                    global_hidden_block = stream_idx * (mhc_hidden_size // (4 * block_hidden)) + hidden_band
                else:
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
                            l2_cache_ctrl=residual_l2_cache_ctrl,
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

                                    if reuse_apply:
                                        mix_weight = S.vld(old_mix_ub[row, stream_idx], dist='BRC_B32')
                                        band_col = pass_id * 128
                                        apply_lo = S.alloc_var(T.float32)
                                        apply_hi = S.alloc_var(T.float32)
                                        if stream_idx == 0:
                                            apply_lo = S.vmul(unpacked_lo, mix_weight)
                                            apply_hi = S.vmul(unpacked_hi, mix_weight)
                                        else:
                                            apply_lo = S.vld(apply_acc[row, band_col])
                                            apply_hi = S.vld(apply_acc[row, band_col + 64])
                                            S.vmula(apply_lo, unpacked_lo, mix_weight)
                                            S.vmula(apply_hi, unpacked_hi, mix_weight)
                                        if stream_idx == 3:
                                            S.vsts(apply_out[row, band_col], S.vcvt(apply_lo, T.bfloat16), dist='PK_B32')
                                            S.vsts(apply_out[row, band_col + 64], S.vcvt(apply_hi, T.bfloat16), dist='PK_B32')
                                        else:
                                            S.vsts(apply_acc[row, band_col], apply_lo)
                                            S.vsts(apply_acc[row, band_col + 64], apply_hi)

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

                        if reuse_apply:
                            if stream_idx == 3:
                                T.copy(apply_out, hi[aiv_token_begin : aiv_token_begin + 64,
                                                     hidden_band * 256 : (hidden_band + 1) * 256])
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
                                local_hidden_block == num_hidden_blocks_per_split - 1 and hidden_mad == num_mads_per_block - 1,
                                3,
                                2,
                            ),
                        )

            if reuse_apply:
                for partial_stage in T.Serial(num_valid_token_stages):
                    partial_begin = aiv_token_begin_base + partial_stage * cores_per_split * 128
                    T.dual_copy(out_mul_l0[partial_stage, 0:128, 0:24], partial_ub, unit_flag_ctrl=3)
                    T.copy(partial_ub, out_mul[split_id, partial_begin : partial_begin + 64, 0:24])
                    T.copy(sqrsum_ub[partial_stage, 0:64], sqrsum[split_id, partial_begin : partial_begin + 64])
            elif fused_shifted:
                for epilogue_stage in T.Serial(num_valid_token_stages):
                    epilogue_begin = (token_block_base + epilogue_stage * cores_per_split) * block_num_tokens
                    epilogue(x, prev_pre, alpha, bias, hi, pre, post, comb,
                                     out_mul_l0, sqrsum_ub, epilogue_begin, epilogue_stage,
                                     aiv_id, mhc_hidden_size // 4, num_l0c_stages)
            else:
                with T.PerCoreTask():
                    for store_phase in T.Serial(num_store_phases):
                        store_this_phase = T.if_then_else(
                            deterministic,
                            split_id == store_phase,
                            T.if_then_else(store_phase == 0, split_id == 0, split_id != 0),
                        )
                        out_mul_flag = (chunk_idx * num_store_phases + store_phase) % 4

                        if store_this_phase:
                            with T.Task():
                                for token_stage in T.Serial(num_valid_token_stages):
                                    token_block = token_block_base + token_stage * cores_per_split
                                    token_begin = token_block * block_num_tokens
                                    actual_num_tokens = T.min(num_tokens - token_begin, block_num_tokens)
                                    T.copy(
                                        out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                                        out_mul[
                                            token_begin : token_begin + actual_num_tokens,
                                            0:mhc_mult3,
                                        ],
                                        unit_flag_ctrl=3,
                                    )
                        num_store_syncs = T.if_then_else(
                            store_phase < num_always_synced_store_phases,
                            1,
                            T.min(num_token_chunks - chunk_idx - 1, num_chunk_end_syncs),
                        )
                        for _ in T.Serial(num_store_syncs):
                            T.ascend_sync_inter_arrive('PIPE_FIX', out_mul_flag)
                            T.ascend_sync_inter_wait('PIPE_FIX', out_mul_flag)

                with T.PerCoreTask():
                    for store_phase in T.Serial(num_store_phases):
                        store_this_phase = T.if_then_else(
                            deterministic,
                            split_id == store_phase,
                            T.if_then_else(store_phase == 0, split_id == 0, split_id != 0),
                        )
                        sqrsum_flag = 4 + (chunk_idx * num_store_phases + store_phase) % 4

                        if store_this_phase:
                            with T.Task():
                                for token_stage in T.Serial(num_valid_sqrsum_stages):
                                    token_block = token_block_base + token_stage * cores_per_split
                                    token_begin = token_block * block_num_tokens + aiv_id * half_block_num_tokens
                                    actual_aiv_num_tokens = T.min(
                                        num_tokens - token_begin,
                                        half_block_num_tokens,
                                    )
                                    T.copy(
                                        sqrsum_ub[token_stage, 0:actual_aiv_num_tokens],
                                        sqrsum[token_block, aiv_id, 0:actual_aiv_num_tokens],
                                        l2_cache_ctrl='normal_fv',
                                    )
                        num_store_syncs = T.if_then_else(
                            store_phase < num_always_synced_store_phases,
                            1,
                            T.min(num_token_chunks - chunk_idx - 1, num_chunk_end_syncs),
                        )
                        for _ in T.Serial(num_store_syncs):
                            T.ascend_sync_inter_arrive('PIPE_MTE3', sqrsum_flag)
                            T.ascend_sync_inter_wait('PIPE_MTE3', sqrsum_flag)

        if reuse_apply:
            # Both partials are now written by MTE3, so one uniform Vector
            # barrier suffices before any AIV reads another group's partials.
            with T.PerCoreTask():
                T.ascend_sync_inter_arrive('PIPE_MTE3', 8)
                T.ascend_sync_inter_wait('PIPE_MTE3', 8)
            epilogue(out_mul, sqrsum, alpha, bias, pre, post, comb,
                     num_tokens, num_k_splits, num_aic_cores, core_id, aiv_id)
        else:
            if split_id != 0:
                T.set_atomic_none()

@tilelang.jit
def mhc_projection_token_pipeline(
    x, fn, out_mul, sqrsum,
    block_mhc_mult3: int,
    num_aic_cores: int,
    num_k_splits: int,
    cores_per_split: int,
    num_hidden_blocks_per_split: int,
    num_l0b_stages: int,
    num_l0c_stages: int,
    num_token_blocks_per_chunk: int,
    deterministic: bool = False,
    residual_l2_cache_ctrl="notalloc_keep",
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    mhc_hidden_size = T.const('mhc_hidden_size')
    x: T.Tensor[(num_tokens, mhc_hidden_size), T.bfloat16]
    fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32]
    out_mul: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(T.ceildiv(num_tokens, 128), 2, 64), T.float32]
    _projection_body(
        x, fn, out_mul, sqrsum, None, None, None, None, None, None, None,
        num_tokens, mhc_mult3, mhc_hidden_size,
        block_mhc_mult3, num_aic_cores, num_k_splits, cores_per_split,
        num_hidden_blocks_per_split, num_l0b_stages, num_l0c_stages,
        num_token_blocks_per_chunk, deterministic, residual_l2_cache_ctrl, False, None)


def _u32_index(index_i32):
    return T.reinterpret(index_i32, 'uint32x64')


def _row_reduce(mat, mask_lo, mask_hi, bcast_idx2):
    """Row reduce: vcgadd ×2 + vselr ×2 + vsel → broadcast row sum."""
    return S.vsel(S.vselr(S.vcgadd(mat, mask_lo), bcast_idx2), S.vselr(S.vcgadd(mat, mask_hi), bcast_idx2), mask_lo)


def _row_max(mat, mask_lo, mask_hi, bcast_idx2):
    """Row max: vcgmax ×2 + vselr ×2 + vsel → broadcast row max."""
    return S.vsel(S.vselr(S.vcgmax(mat, mask_lo), bcast_idx2), S.vselr(S.vcgmax(mat, mask_hi), bcast_idx2), mask_lo)


@T.macro
def compute_coefficients(mul_flat_ub, sq_ub, scale_ub, base_ub, mixes_flat_ub,
                         pre_ub, post_ub, comb_in_ub, comb_out_ub,
                         mul_offset, sq_offset, tb, mhc, mhc2, mhc3,
                         inv_rms_group_size, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
                         mhc_post_mult_value, sinkhorn_repeat):
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
                    S.vsqrt(S.vadds(S.vmul(sq_val, inv_group, mask), T.float32(rms_eps), mask), mask),
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
            pre_base_vec = S.vsel(S.vld(base_ub[base_id, 0], dist='BRC_B32'), pre_base_vec, base_mask)
            post_base_vec = S.vsel(S.vld(base_ub[base_id + mhc, 0], dist='BRC_B32'), post_base_vec, base_mask)
        for base_id in range(mhc2):
            base_mask = S.vcmps(col15, T.int32(base_id), mask, 'eq')
            comb_base_vec = S.vsel(
                S.vld(base_ub[base_id + mhc * 2, 0], dist='BRC_B32'),
                comb_base_vec,
                base_mask,
            )

        # pre/post: 16 tokens per VR (4 lanes each)
        for vec_id in range(tb // 16):
            token = S.vadd(S.vshrs(idx, T.int32(2), mask), S.vdup(T.int32(vec_id * 16), T.int32, mask), mask)
            token_offset = S.vadd(S.vshls(token, T.int32(4), mask), S.vshls(token, T.int32(3), mask), mask)
            pre_index = S.vadd(token_offset, col3, mask)
            post_index = S.vadds(pre_index, T.int32(4), mask)

            pre_val = S.vadd(
                S.vmul(S.vgather2(mixes_flat_ub[0], _u32_index(pre_index), mask), scale0, mask),
                pre_base_vec,
                mask,
            )
            pre_sig = S.vdiv(ones, S.vadds(S.vexpdif(zeros, pre_val, mask), T.float32(1), mask), mask)
            S.vsts(pre_ub[vec_id * 16, 0], S.vadds(pre_sig, T.float32(mhc_pre_eps), mask), mask, 'NORM_B32')

            post_val = S.vadd(
                S.vmul(S.vgather2(mixes_flat_ub[0], _u32_index(post_index), mask), scale1, mask),
                post_base_vec,
                mask,
            )
            post_sig = S.vdiv(ones, S.vadds(S.vexpdif(zeros, post_val, mask), T.float32(1), mask), mask)
            S.vsts(
                post_ub[vec_id * 16, 0],
                S.vmuls(post_sig, T.float32(mhc_post_mult_value), mask),
                mask,
                'NORM_B32',
            )

        # comb: 4 tokens per VR (16 lanes each)
        for vec_id in range(tb // 4):
            token = S.vadd(S.vshrs(idx, T.int32(4), mask), S.vdup(T.int32(vec_id * 4), T.int32, mask), mask)
            token_offset = S.vadd(S.vshls(token, T.int32(4), mask), S.vshls(token, T.int32(3), mask), mask)
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
    assert n_splits == 1, 'Ascend pre_big_fuse expects the GEMM to reduce split-K internally (n_splits=1)'

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
            if token_base + tb <= num_tokens:
                T.copy(gemm_out_mul[0:1, token_base : token_base + tb, 0:mhc3], T.view(mul_flat_ub, (tb, mhc3)))
                T.copy(gemm_out_sqrsum[0:1, token_base : token_base + tb], sq_ub)
            else:
                # DMA pad_value pads a row's width, not missing token rows.
                # Use bounded scalar loads for the final partial token tile.
                with T.SimtVF(threads=32):
                    for tail_t, tail_c in T.Parallel(tb, mhc3):
                        if token_base + tail_t < num_tokens:
                            mul_flat_ub[tail_t * mhc3 + tail_c] = gemm_out_mul[0, token_base + tail_t, tail_c]
                        else:
                            mul_flat_ub[tail_t * mhc3 + tail_c] = T.float32(0)
                    for tail_t in T.Parallel(tb):
                        if token_base + tail_t < num_tokens:
                            sq_ub[tail_t] = gemm_out_sqrsum[0, token_base + tail_t]
                        else:
                            sq_ub[tail_t] = T.float32(0)

            ##################################################################
            # _mhc_reduce_partials_and_rmsnorm_fwd + _mhc_pre_split_mixes_fwd
            # + _mhc_sinkhorn_fwd in a single SimdVF phase: the whole latency
            # path is one unit so the scheduler can overlap it with the
            # pre-apply memory stream of the previous token block.
            compute_coefficients(
                mul_flat_ub, sq_ub, scale_ub, base_ub, mixes_flat_ub,
                pre_ub, post_ub, comb_in_ub, comb_out_ub, 0, 0, tb, mhc, mhc2, mhc3,
                inv_rms_group_size, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
                mhc_post_mult_value, sinkhorn_repeat)

            T.copy(pre_ub[:, 0:mhc], pre_mix[token_base : token_base + tb, 0:mhc])
            T.copy(post_ub[:, 0:mhc], post_mix[token_base : token_base + tb, 0:mhc])
            T.copy(comb_out_ub, comb_mix[token_base : token_base + tb, 0:mhc2])


# Each bf16 chunk is unpacked into a 64-lane fp32 vector register.
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


@T.macro
def _reduce_split_coefficients(partial_proj, partial_sq, alpha, bias, pre, post, comb,
                               tokens, splits, cube_cores, cube_id, vector_id):
    proj = T.alloc_shared((384,), T.float32)
    proj_next = T.alloc_shared((384,), T.float32)
    sq = T.alloc_shared((16,), T.float32)
    sq_next = T.alloc_shared((16,), T.float32)
    scale_ub = T.alloc_shared((3, 8), T.float32)
    base_ub = T.alloc_shared((24, 8), T.float32)
    mixes = T.alloc_shared((384,), T.float32)
    pre_tile = T.alloc_shared((16, 4), T.float32)
    post_tile = T.alloc_shared((16, 4), T.float32)
    comb_in = T.alloc_shared((16, 16), T.float32)
    comb_out = T.alloc_shared((16, 16), T.float32)

    with T.SimtVF(threads=32):
        for i in T.Parallel(3):
            scale_ub[i, 0] = alpha[i]
        for i in T.Parallel(24):
            base_ub[i, 0] = bias[i]

    worker = cube_id * 2 + vector_id
    jobs = T.ceildiv(T.max(tokens // 16 - worker, 0), cube_cores * 2)
    for job in T.Serial(jobs):
        start = (worker + job * cube_cores * 2) * 16
        T.copy(partial_proj[0, start : start + 16, 0:24], T.view(proj, (16, 24)))
        T.copy(partial_sq[0, start : start + 16], sq)
        for k in T.Serial(1, splits):
            T.copy(partial_proj[k, start : start + 16, 0:24], T.view(proj_next, (16, 24)))
            T.copy(partial_sq[k, start : start + 16], sq_next)
            with T.SimdVF():
                mask = S.pset(32)
                for v in T.unroll(6, explicit=True):
                    off = v * 64
                    S.vsts(proj[off], S.vadd(S.vld(proj[off]), S.vld(proj_next[off]), mask))
            with T.SimtVF(threads=32):
                for row in T.Parallel(16):
                    sq[row] = sq[row] + sq_next[row]
        compute_coefficients(
            proj, sq, scale_ub, base_ub, mixes, pre_tile, post_tile, comb_in, comb_out,
            0, 0, 16, 4, 16, 24, T.float32(1.0 / 20480), 1e-20, 1e-6, 1e-6, 2., 20)
        T.copy(pre_tile, pre[start : start + 16, 0:4])
        T.copy(post_tile, post[start : start + 16, 0:4])
        T.copy(comb_out, comb[start : start + 16, 0:16])


@tilelang.jit
def mhc_shifted_single_reuse_kernel(
    x, fn, prev_pre, alpha, bias, hi, pre, post, comb, partial_proj, partial_sq,
    block_mhc_mult3: int, num_aic_cores: int, num_k_splits: int, cores_per_split: int,
    num_hidden_blocks_per_split: int, num_l0b_stages: int, num_l0c_stages: int,
    num_token_blocks_per_chunk: int,
):
    tokens = T.dynamic('tokens')
    x: T.Tensor[(tokens, 20480), T.bfloat16]
    fn: T.Tensor[(24, 20480), T.float32]
    prev_pre: T.Tensor[(tokens, 4), T.float32]
    alpha: T.Tensor[(3,), T.float32]
    bias: T.Tensor[(24,), T.float32]
    hi: T.Tensor[(tokens, 5120), T.bfloat16]
    pre: T.Tensor[(tokens, 4), T.float32]
    post: T.Tensor[(tokens, 4), T.float32]
    comb: T.Tensor[(tokens, 16), T.float32]
    partial_proj: T.Tensor[(num_k_splits, tokens, 24), T.float32]
    partial_sq: T.Tensor[(num_k_splits, tokens), T.float32]
    assert num_l0c_stages == 1 and num_hidden_blocks_per_split % 4 == 0
    _projection_body(
        x, fn, partial_proj, partial_sq, prev_pre, alpha, bias, hi, pre, post, comb,
        tokens, 24, 20480, block_mhc_mult3, num_aic_cores, num_k_splits, cores_per_split,
        num_hidden_blocks_per_split, num_l0b_stages, num_l0c_stages,
        num_token_blocks_per_chunk, False, None, False, _reduce_split_coefficients, True)


def bind_shifted_pre(x, prev_pre, phi, alpha, bias, projection_config=None, *,
                     projection_l2_default=True, tail_cores=None, tail_stages=2):
    """Compile the three launchers for the supplied tensors and return a callable."""
    import torch
    from tile_kernels.config import get_num_vec_cores

    n, streams, hidden = x.shape
    if (streams, hidden) != (4, 5120) or not n or n % 128:
        raise ValueError('Expected residual[N,4,5120] with positive N divisible by 128')
    if projection_config is None:
        projection_config = get_config(streams * hidden, 24, n)
    flat = x.flatten(1)
    prev = prev_pre.view(n, 4)
    vec_cores = get_num_vec_cores()
    if tail_stages not in (1, 2, 3, 4) or (tail_cores is not None and not 0 < tail_cores <= vec_cores):
        raise ValueError('Invalid tail core/stage configuration')
    settings = dict(rms_eps=1e-20, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
                    mhc_post_mult_value=2., sinkhorn_repeat=20, hidden=hidden, n_splits=1,
                    token_block=16, num_stages=tail_stages, num_vec_cores=tail_cores or vec_cores)

    def allocate_outputs():
        hi = torch.empty(n, hidden, dtype=x.dtype, device=x.device)
        packed = torch.empty(n * 24, dtype=torch.float32, device=x.device)
        pre, post, comb = packed.split((n * 4, n * 4, n * 16))
        return hi, pre.view(n, 4), post.view(n, 4), comb.view(n, 16)

    def allocate_projection():
        raw = torch.empty(1, n, 24, dtype=torch.float32, device=x.device)
        sq = torch.empty(n // 128, 2, 64, dtype=torch.float32, device=x.device)
        return raw, sq

    hi, pre, post, comb = allocate_outputs()
    raw, sq = allocate_projection()
    projection = mhc_projection_token_pipeline.compile(
        flat, phi, raw[0], sq, deterministic=False,
        residual_l2_cache_ctrl=None if projection_l2_default else 'notalloc_keep',
        **projection_config)
    coefficients = mhc_coeff_token_pipeline.compile(
        raw, sq.view(1, n), alpha, bias, pre, post, comb, **settings)
    apply = mhc_pre_apply_flat_pipeline_asc.compile(
        x, prev, hi, h_blk=hidden, num_vec_cores=vec_cores, pipeline_stages=3)

    def run():
        hi, pre, post, comb = allocate_outputs()
        apply(x, prev, hi)
        raw, sq = allocate_projection()
        projection(flat, phi, raw[0], sq)
        coefficients(raw, sq.view(1, n), alpha, bias, pre, post, comb)
        return hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb.view(n, 4, 4)

    sources = {name: kernel.get_kernel_source() for name, kernel in
               (('apply', apply), ('projection', projection), ('tail', coefficients))}
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    return run, dict(
        all_token_pipeline=True, projection_l2_default=projection_l2_default,
        apply_pipeline=(hidden, 3), flat_pipeline=True,
        tail_settings={key: value for key, value in settings.items() if key != 'hidden'},
        operation='public JITImpl.compile -> JITKernel call; no private cache access',
        projection_device_source_sha256=hashes['projection'], tail_device_source_sha256=hashes['tail'],
        numerical_kernel_changes=False, output_preallocation=False,
        per_call_allocations=4, expected_kernels_per_call=3,
        note='compilation excluded for all sides; all device math, output/workspace allocation and views remain timed',
        generated_apply_source=sources['apply'], standalone_apply_source_sha256=hashes['apply'],
        standalone_apply_hidden_tile=hidden, generated_sources=sources)

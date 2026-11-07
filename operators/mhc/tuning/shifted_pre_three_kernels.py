# SPDX-License-Identifier: MIT
# Copyright (c) 2026 DeepSeek. Portions adapted for Q4 shifted mHC.
"""Shared shifted Pre paths for the baseline and experimental benchmarks."""
import hashlib

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from shifted_pre_k8_preallocated import (
    compute_coefficients,
    get_config,
    mhc_pre_apply_flat_pipeline_asc,
)


@T.macro
def _projection_body(
    x, fn, out_mul, sqrsum, prev_pre, alpha, bias, hi, pre, post, comb,
    num_tokens, mhc_mult3, mhc_hidden_size,
    block_mhc_mult3, num_aic_cores, num_k_splits, cores_per_split,
    num_hidden_blocks_per_split, num_l0b_stages, num_l0c_stages,
    num_token_blocks_per_chunk, deterministic, residual_l2_cache_ctrl, fused_shifted, epilogue,
    reuse_apply=False, defer_reduce=False, block_num_tokens=128, fused_split_coeff=False,
):
    reuse_pipeline = reuse_apply and num_l0c_stages > 1
    block_hidden = 128 if reuse_pipeline else 256
    mad_hidden = 64
    num_aivs = 2
    half_block_num_tokens = block_num_tokens // num_aivs
    num_mads_per_block = block_hidden // mad_hidden
    # The 256-wide reuse path needs one input slot to fit the Apply buffers.
    # Two 128-wide slots have the same footprint in the pipelined variant.
    num_load_stages = 2 if reuse_pipeline else (1 if reuse_apply else (2 if fused_shifted else 3))
    num_cast_stages = 2 if reuse_pipeline or not reuse_apply else 1
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
        # The SIMD row-sum accumulator always writes a full 64-lane register.
        sqrsum_ub = T.alloc_shared((num_l0c_stages, 64), T.float32)
        if fused_split_coeff:
            partial_ub = T.alloc_shared((num_l0c_stages, 64, 24), T.float32)
        if reuse_apply:
            # Explicit slots keep each token tile's four-stream sum separate.
            old_mix_ub = T.alloc_shared((num_l0c_stages, 64, 4), T.float32)
            apply_acc = T.alloc_shared((num_l0c_stages, 64, block_hidden), T.float32)
            apply_out = T.alloc_shared((64, block_hidden), T.bfloat16)
            partial_ub = T.alloc_shared((num_l0c_stages, 64, 24), T.float32)
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

        if not reuse_apply and not defer_reduce:
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
                for mix_stage in T.Serial(num_valid_token_stages):
                    mix_begin = aiv_token_begin_base + mix_stage * cores_per_split * 128
                    T.copy(prev_pre[mix_begin : mix_begin + 64, 0:4], old_mix_ub[mix_stage, 0:64, 0:4])

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
                                        mix_weight = S.vld(old_mix_ub[token_stage, row, stream_idx], dist='BRC_B32')
                                        band_col = pass_id * 128
                                        apply_lo = S.alloc_var(T.float32)
                                        apply_hi = S.alloc_var(T.float32)
                                        if stream_idx == 0:
                                            apply_lo = S.vmul(unpacked_lo, mix_weight)
                                            apply_hi = S.vmul(unpacked_hi, mix_weight)
                                        else:
                                            apply_lo = S.vld(apply_acc[token_stage, row, band_col])
                                            apply_hi = S.vld(apply_acc[token_stage, row, band_col + 64])
                                            S.vmula(apply_lo, unpacked_lo, mix_weight)
                                            S.vmula(apply_hi, unpacked_hi, mix_weight)
                                        if stream_idx == 3:
                                            S.vsts(apply_out[row, band_col], S.vcvt(apply_lo, T.bfloat16), dist='PK_B32')
                                            S.vsts(apply_out[row, band_col + 64], S.vcvt(apply_hi, T.bfloat16), dist='PK_B32')
                                        else:
                                            S.vsts(apply_acc[token_stage, row, band_col], apply_lo)
                                            S.vsts(apply_acc[token_stage, row, band_col + 64], apply_hi)

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
                                                     hidden_band * block_hidden : (hidden_band + 1) * block_hidden])
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

            if fused_split_coeff:
                for drain_slot in T.Serial(num_valid_token_stages):
                    T.dual_copy(out_mul_l0[drain_slot, 0:128, 0:24],
                                partial_ub[drain_slot, 0:64, 0:24], unit_flag_ctrl=3)
                with T.PerCoreTask():
                    with T.Task():
                        for partial_slot in T.Serial(num_valid_token_stages):
                            partial_token = aiv_token_begin_base + partial_slot * cores_per_split * block_num_tokens
                            T.copy(partial_ub[partial_slot, 0:64, 0:24],
                                   out_mul[split_id, partial_token : partial_token + 64, 0:24])
                        for sum_slot in T.Serial(num_valid_sqrsum_stages):
                            sum_token = aiv_token_begin_base + sum_slot * cores_per_split * block_num_tokens
                            T.copy(sqrsum_ub[sum_slot, 0:64], sqrsum[split_id, sum_token : sum_token + 64])
                    T.ascend_sync_inter_arrive('PIPE_MTE3', 8)
            elif defer_reduce:
                for partial_slot in T.Serial(num_valid_token_stages):
                    partial_token = (token_block_base + partial_slot * cores_per_split) * block_num_tokens
                    T.copy(out_mul_l0[partial_slot, 0:128, 0:24],
                           out_mul[split_id, partial_token : partial_token + 128, 0:24], unit_flag_ctrl=3)
                for sum_slot in T.Serial(num_valid_sqrsum_stages):
                    sum_token = aiv_token_begin_base + sum_slot * cores_per_split * block_num_tokens
                    T.copy(sqrsum_ub[sum_slot, 0:64], sqrsum[split_id, sum_token : sum_token + 64])
            elif reuse_apply:
                for drain_stage in T.Serial(num_valid_token_stages):
                    T.dual_copy(out_mul_l0[drain_stage, 0:128, 0:24],
                                partial_ub[drain_stage, 0:64, 0:24], unit_flag_ctrl=3)
                with T.PerCoreTask():
                    # Fixpipe publishes the whole drain loop as one task.
                    # Consume its notification once, then store all UB slots.
                    with T.Task():
                        for store_stage in T.Serial(num_valid_token_stages):
                            partial_start = (token_block_base + store_stage * cores_per_split) * 64
                            T.copy(partial_ub[store_stage, 0:64, 0:24], out_mul[split_id, aiv_id, partial_start : partial_start + 64, 0:24])
                            T.copy(sqrsum_ub[store_stage, 0:64], sqrsum[split_id, aiv_id, partial_start : partial_start + 64])
                    # Every AIV signals from MTE3 after its last chunk, including
                    # workers with no tile in that chunk. The reader waits on MTE2.
                    for _ in T.Serial(T.if_then_else(chunk_idx + 1 == num_token_chunks, 1, 0)):
                        T.ascend_sync_inter_arrive('PIPE_MTE3', 8)
            elif fused_shifted:
                for epilogue_stage in T.Serial(num_valid_token_stages):
                    epilogue_begin = (token_block_base + epilogue_stage * cores_per_split) * block_num_tokens
                    epilogue(x, prev_pre, alpha, bias, hi, pre, post, comb,
                                     out_mul_l0, sqrsum_ub, epilogue_begin, epilogue_stage,
                                     aiv_id, mhc_hidden_size // 4, num_l0c_stages,
                                     half_block_num_tokens)
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

        if fused_split_coeff:
            epilogue(out_mul, sqrsum, alpha, bias, pre, post, comb,
                     num_tokens, num_k_splits, num_aic_cores, core_id, aiv_id, True)
        elif reuse_apply:
            epilogue(out_mul, sqrsum, alpha, bias, pre, post, comb,
                     num_tokens, num_k_splits, num_aic_cores, core_id, aiv_id)
        elif not defer_reduce:
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


@tilelang.jit
def mhc_projection_split_partials(
    x, fn, out_mul, sqrsum,
    block_mhc_mult3: int, num_aic_cores: int, num_k_splits: int, cores_per_split: int,
    num_hidden_blocks_per_split: int, num_l0b_stages: int, num_l0c_stages: int,
    num_token_blocks_per_chunk: int,
):
    tokens = T.dynamic('tokens')
    flat_hidden = T.const('flat_hidden')
    x: T.Tensor[(tokens, flat_hidden), T.bfloat16]
    fn: T.Tensor[(24, flat_hidden), T.float32]
    out_mul: T.Tensor[(num_k_splits, tokens, 24), T.float32]
    sqrsum: T.Tensor[(num_k_splits, tokens), T.float32]
    _projection_body(
        x, fn, out_mul, sqrsum, None, None, None, None, None, None, None,
        tokens, 24, flat_hidden, block_mhc_mult3, num_aic_cores, num_k_splits, cores_per_split,
        num_hidden_blocks_per_split, num_l0b_stages, num_l0c_stages,
        num_token_blocks_per_chunk, False, None, False, None, False, True)


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
    batch_partials: bool = False,
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
        if batch_partials:
            batch_mul_ub = T.alloc_shared((n_splits, tb * mhc3), T.float32)
            # Pad the UB row to a full SIMD vector. DMA and arithmetic use
            # only the first tb lanes.
            batch_sq_ub = T.alloc_shared((n_splits, 64), T.float32)
        elif n_splits > 1:
            partial_mul_ub = T.alloc_shared((tb * mhc3,), T.float32)
            partial_sq_ub = T.alloc_shared((tb,), T.float32)
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
            if batch_partials:
                raw_rows = T.view(gemm_out_mul, (n_splits, num_tokens * mhc3))
                raw_begin = token_base * mhc3
                T.copy(raw_rows[0:n_splits, raw_begin : raw_begin + tb * mhc3], batch_mul_ub)
                T.copy(gemm_out_sqrsum[0:n_splits, token_base : token_base + tb],
                       batch_sq_ub[0:n_splits, 0:tb])
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
            elif token_base + tb <= num_tokens:
                T.copy(gemm_out_mul[0:1, token_base : token_base + tb, 0:mhc3], T.view(mul_flat_ub, (tb, mhc3)))
                T.copy(gemm_out_sqrsum[0:1, token_base : token_base + tb], sq_ub)
                if n_splits > 1:
                    for k in T.Serial(1, n_splits):
                        T.copy(gemm_out_mul[k, token_base : token_base + tb, 0:mhc3], T.view(partial_mul_ub, (tb, mhc3)))
                        T.copy(gemm_out_sqrsum[k, token_base : token_base + tb], partial_sq_ub)
                        with T.SimdVF():
                            mask = S.pset(32)
                            for vec in T.unroll(tb * mhc3 // 64, explicit=True):
                                offset = vec * 64
                                S.vsts(mul_flat_ub[offset], S.vadd(S.vld(mul_flat_ub[offset]), S.vld(partial_mul_ub[offset]), mask))
                        with T.SimtVF(threads=32):
                            for row in T.Parallel(tb):
                                sq_ub[row] = sq_ub[row] + partial_sq_ub[row]
            elif n_splits == 1:
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


@T.macro
def _reduce_split_coefficients(partial_proj, partial_sq, alpha, bias, pre, post, comb,
                               tokens, splits, cube_cores, cube_id, vector_id, flat_layout=False):
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

    if flat_layout:
        with T.PerCoreTask():
            T.ascend_sync_inter_wait('PIPE_MTE2', 8)
            with T.Task():
                with T.SimtVF(threads=32):
                    for i in T.Parallel(3):
                        scale_ub[i, 0] = alpha[i]
                    for i in T.Parallel(24):
                        base_ub[i, 0] = bias[i]
    else:
        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = alpha[i]
            for i in T.Parallel(24):
                base_ub[i, 0] = bias[i]

    # Keep AIV0 and AIV1 on disjoint workspace planes. Within a plane
    # consecutive 64-row blocks correspond to alternate halves of each tile.
    jobs = T.ceildiv(T.max(tokens // 32 - cube_id, 0), cube_cores)
    for job in T.Serial(jobs):
        start = (cube_id + job * cube_cores) * 16
        output_start = start // 64 * 128 + vector_id * 64 + start % 64
        with T.PerCoreTask():
            for _ in T.Serial(0 if flat_layout else T.max(1 - job, 0)):
                T.ascend_sync_inter_wait('PIPE_MTE2', 8)
            with T.Task():
                if flat_layout:
                    T.copy(partial_proj[0, output_start : output_start + 16, 0:24], T.view(proj, (16, 24)))
                    T.copy(partial_sq[0, output_start : output_start + 16], sq)
                else:
                    T.copy(partial_proj[0, vector_id, start : start + 16, 0:24], T.view(proj, (16, 24)))
                    T.copy(partial_sq[0, vector_id, start : start + 16], sq)
        for k in T.Serial(1, splits):
            if flat_layout:
                T.copy(partial_proj[k, output_start : output_start + 16, 0:24], T.view(proj_next, (16, 24)))
                T.copy(partial_sq[k, output_start : output_start + 16], sq_next)
            else:
                T.copy(partial_proj[k, vector_id, start : start + 16, 0:24], T.view(proj_next, (16, 24)))
                T.copy(partial_sq[k, vector_id, start : start + 16], sq_next)
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
        T.copy(pre_tile, pre[output_start : output_start + 16, 0:4])
        T.copy(post_tile, post[output_start : output_start + 16, 0:4])
        T.copy(comb_out, comb[output_start : output_start + 16, 0:16])


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
    partial_proj: T.Tensor[(num_k_splits, 2, tokens // 2, 24), T.float32]
    partial_sq: T.Tensor[(num_k_splits, 2, tokens // 2), T.float32]
    assert num_l0c_stages in (1, 2) and num_hidden_blocks_per_split % 4 == 0
    _projection_body(
        x, fn, partial_proj, partial_sq, prev_pre, alpha, bias, hi, pre, post, comb,
        tokens, 24, 20480, block_mhc_mult3, num_aic_cores, num_k_splits, cores_per_split,
        num_hidden_blocks_per_split, num_l0b_stages, num_l0c_stages,
        num_token_blocks_per_chunk, False, None, False, _reduce_split_coefficients, True)


def bind_shifted_pre(x, prev_pre, phi, alpha, bias, projection_config=None, *,
                     projection_l2_default=True, tail_cores=None, tail_stages=2,
                     deferred_splits=None, batch_partials=False, preallocate=False,
                     reuse_workspace=False):
    """Compile the three launchers for the supplied tensors and return a callable."""
    import torch
    from tile_kernels.config import get_num_vec_cores

    n, streams, hidden = x.shape
    if (streams, hidden) != (4, 5120) or not n or n % 128:
        raise ValueError('Expected residual[N,4,5120] with positive N divisible by 128')
    if batch_partials and deferred_splits != 8:
        raise ValueError('Batch reduction is only enabled for deferred K8')
    if deferred_splits is not None:
        projection_config = get_config(streams * hidden, 24, n, split_override=deferred_splits)
    elif projection_config is None:
        projection_config = get_config(streams * hidden, 24, n)
    flat = x.flatten(1)
    prev = prev_pre.view(n, 4)
    vec_cores = get_num_vec_cores()
    if tail_stages not in (1, 2, 3, 4) or (tail_cores is not None and not 0 < tail_cores <= vec_cores):
        raise ValueError('Invalid tail core/stage configuration')
    splits = deferred_splits or 1
    settings = dict(rms_eps=1e-20, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
                    mhc_post_mult_value=2., sinkhorn_repeat=20, hidden=hidden, n_splits=splits,
                    token_block=16, num_stages=tail_stages, num_vec_cores=tail_cores or vec_cores)

    def allocate_outputs():
        hi = torch.empty(n, hidden, dtype=x.dtype, device=x.device)
        packed = torch.empty(n * 24, dtype=torch.float32, device=x.device)
        pre, post, comb = packed.split((n * 4, n * 4, n * 16))
        return hi, pre.view(n, 4), post.view(n, 4), comb.view(n, 16)

    def allocate_projection():
        raw = torch.empty(splits, n, 24, dtype=torch.float32, device=x.device)
        sq_shape = (splits, n) if deferred_splits else (n // 128, 2, 64)
        sq = torch.empty(sq_shape, dtype=torch.float32, device=x.device)
        return raw, sq

    hi, pre, post, comb = allocate_outputs()
    raw, sq = allocate_projection()
    if deferred_splits:
        projection = mhc_projection_split_partials.compile(flat, phi, raw, sq, **projection_config)
    else:
        projection = mhc_projection_token_pipeline.compile(
            flat, phi, raw[0], sq, deterministic=False,
            residual_l2_cache_ctrl=None if projection_l2_default else 'notalloc_keep',
            **projection_config)
    coefficients = mhc_coeff_token_pipeline.compile(
        raw, sq.view(splits, n), alpha, bias, pre, post, comb, batch_partials=batch_partials, **settings)
    apply = mhc_pre_apply_flat_pipeline_asc.compile(
        x, prev, hi, h_blk=hidden, num_vec_cores=vec_cores, pipeline_stages=3)

    # Diagnostic only: outputs alias across calls and are overwritten on reuse.
    # Benchmark calls execute in order on the same stream.
    fixed_outputs = (hi, pre, post, comb) if preallocate else None
    fixed_workspace = (raw, sq) if preallocate or reuse_workspace else None

    def run():
        hi, pre, post, comb = fixed_outputs or allocate_outputs()
        apply(x, prev, hi)
        raw, sq = fixed_workspace or allocate_projection()
        projection(flat, phi, raw if deferred_splits else raw[0], sq)
        coefficients(raw, sq.view(splits, n), alpha, bias, pre, post, comb)
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
        numerical_kernel_changes=False, output_preallocation=preallocate,
        workspace_reuse=fixed_workspace is not None,
        per_call_allocations=0 if preallocate else (2 if reuse_workspace else 4),
        expected_kernels_per_call=3,
        note=('diagnostic single-stream reuse; returned outputs alias across calls; tensor views and device math timed'
              if preallocate else 'single-stream workspace reuse; fresh outputs, views and device math timed'
              if reuse_workspace else 'compilation excluded; all device math, output/workspace allocation and views remain timed'),
        generated_apply_source=sources['apply'], standalone_apply_source_sha256=hashes['apply'],
        standalone_apply_hidden_tile=hidden, generated_sources=sources,
        deferred_splits=deferred_splits, batch_partials=batch_partials, projection_config=projection_config)

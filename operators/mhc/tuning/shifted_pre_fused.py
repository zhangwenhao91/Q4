# SPDX-License-Identifier: MIT
# Copyright (c) 2026 DeepSeek. Portions adapted for Q4 shifted mHC.
"""Shifted mHC Pre for H=5120 and four residual streams.

Two launches: projection (raw + squared norm), then fused coefficients + Apply.
N must be a positive multiple of 128. Defaults target 32 AIC / 64 AIV.
Only PyTorch and TileLang Ascend are required; no other Q4 modules are imported.

Inputs: x BF16[N,4,5120], prev_pre FP32[N,4,1] (or [N,4]),
        phi FP32[24,20480], alpha FP32[3], bias FP32[24].
Outputs: hi BF16[N,5120], next_pre/post FP32[N,4,1], comb FP32[N,4,4].
Apply uses prev_pre; next_pre is produced for the next layer.

Usage on Ascend tensors::

    run, config = bind_shifted_pre(x, prev_pre, phi, alpha, bias)
    hi, next_pre, post, comb = run()  # JIT compilation is done during binding.

Projection adapted from DeepGEMM-Ascend 3a353262; coefficient/Apply math adapted
from DeepSeek TileKernels c1cf914 (MIT).
"""
import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout


HIDDEN = 5120
HC_MULT = 4
RMS_GROUP_SIZE = HC_MULT * HIDDEN
RMS_EPS = 1e-20
PRE_EPS = 1e-6
SINKHORN_EPS = 1e-6
POST_MULTIPLIER = 2.0
SINKHORN_REPEAT = 20


def get_config(mhc_hidden_size: int, mhc_mult3: int, num_tokens: int, deterministic: bool = False, *,
               block_num_tokens=None, num_aic_cores=32) -> dict:
    """H5120 projection tiling for the supplied physical AIC capacity."""
    assert (mhc_hidden_size, mhc_mult3) == (20480, 24), 'Expected H5120, four streams and 24 coefficients'
    assert num_tokens > 0 and num_tokens % 128 == 0, 'Expected a positive multiple of 128 tokens'
    assert not deterministic, 'Only the non-deterministic split-K path is supported'
    if block_num_tokens is None:
        block_num_tokens = 64 if num_tokens == 128 else 128
    assert block_num_tokens in (64, 128), 'Projection M tile must be 64 or 128'
    assert isinstance(num_aic_cores, int) and num_aic_cores > 0, 'Expected positive AIC capacity'
    cores = num_aic_cores
    token_blocks = num_tokens // block_num_tokens
    l0c_stages = 2048 // block_num_tokens  # 256 KiB / (M * 32 * sizeof(FP32)).
    # 20480 / K256 = 80: at most 16 equal power-of-two K splits. Preserve
    # workload-dependent grouping; changing it changes floating-point sums.
    split_limit = min(16, cores & -cores, max(1, cores * l0c_stages // token_blocks))
    splits = 1 << (split_limit.bit_length() - 1)
    return dict(block_num_tokens=block_num_tokens, num_k_splits=splits,
                num_aic_cores=splits * min(cores // splits, token_blocks), num_l0c_stages=l0c_stages)


@tilelang.jit
def mhc_projection_token_pipeline(x, fn, out_mul, sqrsum, num_aic_cores: int,
                                  num_k_splits: int, num_l0c_stages: int,
                                  block_num_tokens: int = 128, deterministic: bool = False,
                                  residual_l2_cache_ctrl="notalloc_keep"):
    assert not deterministic, 'Only the non-deterministic split-K path is supported'
    assert block_num_tokens in (64, 128), 'Projection M tile must be 64 or 128'
    assert num_k_splits in (1, 2, 4, 8, 16) and num_aic_cores % num_k_splits == 0
    assert 0 < num_l0c_stages <= 2048 // block_num_tokens
    num_tokens = T.dynamic('num_tokens')
    x: T.Tensor[(num_tokens, 20480), T.bfloat16]
    fn: T.Tensor[(24, 20480), T.float32]
    out_mul: T.Tensor[(num_tokens, 24), T.float32]
    sqrsum: T.Tensor[(T.ceildiv(num_tokens, block_num_tokens), 2, block_num_tokens // 2), T.float32]
    mhc_mult3 = 24
    block_mhc_mult3 = 32
    cores_per_split = num_aic_cores // num_k_splits
    num_hidden_blocks_per_split = 80 // num_k_splits
    num_token_blocks_per_chunk = cores_per_split * num_l0c_stages
    block_hidden = 256
    mad_hidden = 64
    half_block_num_tokens = block_num_tokens // 2
    num_mads_per_block = block_hidden // mad_hidden

    num_store_phases = 2 if num_k_splits > 1 else 1
    # Initialize split 0 before atomic adds; the last phase syncs only between
    # chunks. Uniform 0/1-trip loops are required because AutoSchedule rejects
    # conditional inter-core waits inside PerCoreTask.
    num_always_synced_store_phases = 1 if num_k_splits > 1 else 0
    num_chunk_end_syncs = 1 if num_k_splits > 1 else 0
    nz_stage_rows = half_block_num_tokens + 1

    with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
        split_id = core_id // cores_per_split
        core_in_split = core_id % cores_per_split
        num_token_blocks = T.ceildiv(num_tokens, block_num_tokens)
        num_token_chunks = T.ceildiv(num_token_blocks, num_token_blocks_per_chunk)

        x_ub = T.alloc_shared((half_block_num_tokens, block_hidden), T.bfloat16)
        x_nz_ub = T.alloc_shared((nz_stage_rows, block_hidden), T.float32)
        T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})
        # One full FP32 vector spill per stage, even for M64's 32 rows/AIV.
        sqrsum_ub = T.alloc_shared((num_l0c_stages, 64), T.float32)

        x_l1 = T.alloc_l1((block_num_tokens, block_hidden), T.float32)
        fn_l1 = T.alloc_l1((block_mhc_mult3, block_hidden), T.float32)
        x_l0 = T.alloc_l0a((block_num_tokens, mad_hidden), T.float32)
        fn_l0 = T.alloc_l0b((num_mads_per_block, block_mhc_mult3, mad_hidden), T.float32)
        out_mul_l0 = T.alloc_l0c((num_l0c_stages, block_num_tokens, block_mhc_mult3), T.float32)

        T.annotate_buffer_versions({x_ub: 3, x_nz_ub: 2, x_l1: 3, fn_l1: 2, x_l0: 2, fn_l0: 2})

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
                T.max(T.ceildiv(num_tokens - aiv_token_begin_base, cores_per_split * block_num_tokens), 0),
            )
            for local_hidden_block in T.Pipelined(num_hidden_blocks_per_split, num_stages=2):
                global_hidden_block = split_id * num_hidden_blocks_per_split + local_hidden_block
                hidden_begin = global_hidden_block * block_hidden

                # Keep physical buffers block_mhc_mult3-aligned, but restrict
                # data movement and MAD regions to the logical mhc_mult3.
                T.copy(fn[0:mhc_mult3, hidden_begin : hidden_begin + block_hidden], fn_l1[0:mhc_mult3, :])
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
                        actual_aiv_num_tokens = T.min(num_tokens - aiv_token_begin, half_block_num_tokens)
                        T.copy(
                            x[aiv_token_begin : aiv_token_begin + actual_aiv_num_tokens,
                              hidden_begin : hidden_begin + block_hidden],
                            x_ub[0:actual_aiv_num_tokens, :],
                            l2_cache_ctrl=residual_l2_cache_ctrl, pad_value=0,
                        )

                        # Widen one AIV half-tile, pack padded NZ, and update row sums.
                        with T.SimdVF(latency=10 * half_block_num_tokens):
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
                                    T.access_ptr(x_nz_ub[row, 0], 'w', 1, block_hidden), 'float32',
                                )

                                for pass_id in T.Serial(block_hidden // 128):
                                    packed = T.simd.vld(x_ub[row, pass_id * 128])
                                    lo_bits, hi_bits = T.simd.vintlv(zero_bf16, packed)
                                    unpacked_lo = T.reinterpret(lo_bits, 'float32x64')
                                    unpacked_hi = T.reinterpret(hi_bits, 'float32x64')

                                    row_acc = T.simd.vadd(
                                        row_acc, T.simd.vmul(unpacked_lo, unpacked_lo, f32_mask), f32_mask,
                                    )
                                    row_acc = T.simd.vadd(
                                        row_acc, T.simd.vmul(unpacked_hi, unpacked_hi, f32_mask), f32_mask,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        unpacked_lo, nz_ptr, nz_stride, f32_mask, update=True,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        unpacked_hi, nz_ptr, nz_stride, f32_mask, update=True,
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
                            x_nz_ub[0:half_block_num_tokens, 0:block_hidden], x_l1[0:block_num_tokens, 0:block_hidden],
                        )

                    for hidden_mad in T.Serial(num_mads_per_block):
                        T.copy(
                            x_l1[0:actual_num_tokens, hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden],
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
                                3, 2,
                            ),
                        )

            with T.PerCoreTask():
                for store_phase in T.Serial(num_store_phases):
                    store_this_phase = T.if_then_else(store_phase == 0, split_id == 0, split_id != 0)
                    out_mul_flag = (chunk_idx * num_store_phases + store_phase) % 4

                    if store_this_phase:
                        with T.Task():
                            for token_stage in T.Serial(num_valid_token_stages):
                                token_block = token_block_base + token_stage * cores_per_split
                                token_begin = token_block * block_num_tokens
                                actual_num_tokens = T.min(num_tokens - token_begin, block_num_tokens)
                                T.copy(
                                    out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                                    out_mul[token_begin : token_begin + actual_num_tokens, 0:mhc_mult3],
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
                    store_this_phase = T.if_then_else(store_phase == 0, split_id == 0, split_id != 0)
                    sqrsum_flag = 4 + (chunk_idx * num_store_phases + store_phase) % 4

                    if store_this_phase:
                        with T.Task():
                            for token_stage in T.Serial(num_valid_sqrsum_stages):
                                token_block = token_block_base + token_stage * cores_per_split
                                token_begin = token_block * block_num_tokens + aiv_id * half_block_num_tokens
                                actual_aiv_num_tokens = T.min(num_tokens - token_begin, half_block_num_tokens)
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

        if split_id != 0:
            T.set_atomic_none()


def _row_reduce(mat, mask_lo, mask_hi, bcast_idx):
    return S.vsel(S.vselr(S.vcgadd(mat, mask_lo), bcast_idx),
                  S.vselr(S.vcgadd(mat, mask_hi), bcast_idx), mask_lo)


def _row_max(mat, mask_lo, mask_hi, bcast_idx):
    return S.vsel(S.vselr(S.vcgmax(mat, mask_lo), bcast_idx),
                  S.vselr(S.vcgmax(mat, mask_hi), bcast_idx), mask_lo)


@T.macro
def sinkhorn_packs(matrices, result, pack_count, mask, mask_lo, mask_hi, bcast_idx, transpose_idx):
    """Normalize packed 4x4 matrices inside one SimdVF.

    This fused tail uses one vector pack with 32 active lanes for two tokens.
    Explicit expansion keeps every register-array index a compile-time constant.
    """
    row_max = S.alloc_local((pack_count,), T.float32)
    exponent = S.alloc_local((pack_count,), T.float32)
    row_sum = S.alloc_local((pack_count,), T.float32)
    normalized = S.alloc_local((pack_count,), T.float32)
    seeded = S.alloc_local((pack_count,), T.float32)
    transposed = S.alloc_local((pack_count,), T.float32)
    column_sum = S.alloc_local((pack_count,), T.float32)
    column_normalized = S.alloc_local((pack_count,), T.float32)
    for pack in T.unroll(pack_count, explicit=True):
        row_max[pack] = _row_max(matrices[pack], mask_lo, mask_hi, bcast_idx)
    for pack in T.unroll(pack_count, explicit=True):
        exponent[pack] = S.vexpdif(matrices[pack], row_max[pack], mask)
    for pack in T.unroll(pack_count, explicit=True):
        row_sum[pack] = _row_reduce(exponent[pack], mask_lo, mask_hi, bcast_idx)
    for pack in T.unroll(pack_count, explicit=True):
        normalized[pack] = S.vdiv(exponent[pack], row_sum[pack], mask, precision='exact')
    for pack in T.unroll(pack_count, explicit=True):
        seeded[pack] = S.vadds(normalized[pack], T.float32(SINKHORN_EPS), mask)
    for pack in T.unroll(pack_count, explicit=True):
        transposed[pack] = S.vselr(seeded[pack], transpose_idx)
    for pack in T.unroll(pack_count, explicit=True):
        column_sum[pack] = _row_reduce(transposed[pack], mask_lo, mask_hi, bcast_idx)
    for pack in T.unroll(pack_count, explicit=True):
        column_normalized[pack] = S.vdiv(
            transposed[pack], S.vadds(column_sum[pack], T.float32(SINKHORN_EPS), mask), mask, precision='exact')
    for pack in T.unroll(pack_count, explicit=True):
        result[pack] = S.vselr(column_normalized[pack], transpose_idx)
    for _ in T.serial(SINKHORN_REPEAT - 1):
        for pack in T.unroll(pack_count, explicit=True):
            row_sum[pack] = _row_reduce(result[pack], mask_lo, mask_hi, bcast_idx)
        for pack in T.unroll(pack_count, explicit=True):
            result[pack] = S.vdiv(
                result[pack], S.vadds(row_sum[pack], T.float32(SINKHORN_EPS), mask), mask, precision='exact')
        for pack in T.unroll(pack_count, explicit=True):
            transposed[pack] = S.vselr(result[pack], transpose_idx)
        for pack in T.unroll(pack_count, explicit=True):
            column_sum[pack] = _row_reduce(transposed[pack], mask_lo, mask_hi, bcast_idx)
        for pack in T.unroll(pack_count, explicit=True):
            column_normalized[pack] = S.vdiv(
                transposed[pack], S.vadds(column_sum[pack], T.float32(SINKHORN_EPS), mask), mask, precision='exact')
        for pack in T.unroll(pack_count, explicit=True):
            result[pack] = S.vselr(column_normalized[pack], transpose_idx)


@T.macro
def _small_coefficients(raw_ub, sq_ub, scale_ub, base_ub, pre_ub, post_ub, comb_ub):
    """Two tokens: 8 valid pre/post lanes, 32 valid combination lanes."""
    with T.SimdVF():
        full = S.pset(32)
        pre_mask = S.update_mask(8, 32)
        comb_mask = S.update_mask(32, 32)
        idx = S.vci(T.int32(0), T.int32)
        one = S.vdup(T.float32(1), T.float32, full)
        zero = S.vdup(T.float32(0), T.float32, full)

        # Both buffers contain 64 initialized floats.  Token indices below
        # are masked to [0,1] on every lane, including inactive output lanes.
        raw = S.vld(raw_ub[0, 0])
        sq = S.vld(sq_ub[0])
        rms = S.vdiv(
            one,
            S.vsqrt(S.vadds(S.vmuls(sq, T.float32(1.0 / RMS_GROUP_SIZE), full),
                           T.float32(RMS_EPS), full), full),
            full, precision='exact')
        bases = S.vld(base_ub[0])
        scale0 = S.vld(scale_ub[0, 0], dist='BRC_B32')
        scale1 = S.vld(scale_ub[1, 0], dist='BRC_B32')
        scale2 = S.vld(scale_ub[2, 0], dist='BRC_B32')

        col3 = S.vand(idx, S.vdup(T.int32(3), T.int32, full), full)
        token4 = S.vand(S.vshrs(idx, T.int32(2), full),
                       S.vdup(T.int32(1), T.int32, full), full)
        raw_pre_idx = S.vadd(S.vshls(token4, T.int32(5), full), col3, full)
        rms_pre = S.vselr(rms, token4)
        pre_raw = S.vmul(S.vselr(raw, raw_pre_idx), rms_pre, pre_mask)
        pre_val = S.vadd(S.vmul(pre_raw, scale0, pre_mask),
                        S.vselr(bases, col3), pre_mask)
        pre = S.vadds(
            S.vdiv(one, S.vadds(S.vexpdif(zero, pre_val, pre_mask),
                               T.float32(1), pre_mask), pre_mask, precision='exact'),
            T.float32(PRE_EPS), pre_mask)
        S.vsts(pre_ub[0, 0], pre, pre_mask)

        post_raw = S.vmul(S.vselr(raw, S.vadds(raw_pre_idx, T.int32(4), full)),
                         rms_pre, pre_mask)
        post_val = S.vadd(S.vmul(post_raw, scale1, pre_mask),
                         S.vselr(bases, S.vadds(col3, T.int32(4), full)), pre_mask)
        post = S.vmuls(
            S.vdiv(one, S.vadds(S.vexpdif(zero, post_val, pre_mask),
                               T.float32(1), pre_mask), pre_mask, precision='exact'),
            T.float32(POST_MULTIPLIER), pre_mask)
        S.vsts(post_ub[0, 0], post, pre_mask)

        col15 = S.vand(idx, S.vdup(T.int32(15), T.int32, full), full)
        token16 = S.vand(S.vshrs(idx, T.int32(4), full),
                        S.vdup(T.int32(1), T.int32, full), full)
        raw_comb_idx = S.vadds(
            S.vadd(S.vshls(token16, T.int32(5), full), col15, full),
            T.int32(8), full)
        comb_raw = S.vmul(S.vselr(raw, raw_comb_idx), S.vselr(rms, token16), comb_mask)
        comb = S.vadd(S.vmul(comb_raw, scale2, comb_mask),
                      S.vselr(bases, S.vadds(col15, T.int32(8), full)), comb_mask)

        mod8 = S.vand(idx, S.vdup(T.int32(7), T.int32, full), full)
        mask_lo = S.pand(S.vcmps(mod8, T.int32(4), full, 'lt'), comb_mask, full)
        mask_hi = S.pand(S.vcmps(mod8, T.int32(4), full, 'ge'), comb_mask, full)
        # Construct all 64 indices with the full mask.  They remain valid
        # even where the following arithmetic has only 32 active lanes.
        token_base = S.vshls(S.vshrs(idx, T.int32(4), full), T.int32(4), full)
        local_div8 = S.vshrs(col15, T.int32(3), full)
        bcast_idx = S.vadd(S.vshls(S.vshrs(idx, T.int32(4), full), T.int32(1), full),
                          local_div8, full)
        trans_idx = S.vadd(
            S.vadd(token_base, S.vshls(col3, T.int32(2), full), full),
            S.vshrs(col15, T.int32(2), full), full)

        matrices = S.alloc_local((1,), T.float32)
        result = S.alloc_local((1,), T.float32)
        matrices[0] = comb
        sinkhorn_packs(matrices, result, 1, comb_mask, mask_lo, mask_hi, bcast_idx, trans_idx)
        S.vsts(comb_ub[0, 0], result[0], comb_mask)


@T.macro
def _load_small_static(scale, base, scale_ub, base_ub):
    with T.SimtVF(threads=32):
        for i in T.Parallel(64):
            if i < 24:
                base_ub[i] = base[i]
            else:
                base_ub[i] = T.float32(0)
        for i in T.Parallel(3):
            scale_ub[i, 0] = scale[i]


@T.macro
def _load_small_dynamic(raw, sq, raw_ub, sq_ub, token_base, num_tokens):
    # Explicit initialization covers both row padding and a partial token
    # tile.  Scalar loading is intentional here: it also reads only two
    # sq elements without requiring a 32-byte GM overread at the final tile.
    with T.SimtVF(threads=32):
        for i in T.Parallel(64):
            token = i // 32
            channel = i % 32
            if token_base + token < num_tokens and channel < 24:
                raw_ub[token, channel] = raw[0, token_base + token, channel]
            else:
                raw_ub[token, channel] = T.float32(0)
            if i < 2 and token_base + i < num_tokens:
                sq_ub[i] = sq[0, token_base + i]
            else:
                sq_ub[i] = T.float32(0)


@T.macro
def _apply_small_tile(x_ub, prev_ub, output_ub, apply_unroll, apply_pack128):
    with T.SimdVF():
        for token in T.serial(2):
            weights = S.alloc_local((4,), T.float32)
            for stream in T.unroll(4, explicit=True):
                weights[stream] = S.vld(prev_ub[token, stream], dist='BRC_B32')
            if apply_pack128:
                zero_bf16 = S.vdup(T.bfloat16(0), T.bfloat16, S.pset(16))
                for pair in T.serial(HIDDEN // 128):
                    pair_col = pair * 128
                    packed_first = S.vld(x_ub[token, 0, pair_col])
                    first_lo, first_hi = S.vintlv(zero_bf16, packed_first)
                    acc_lo = S.alloc_var(T.float32)
                    acc_hi = S.alloc_var(T.float32)
                    acc_lo = S.vmul(T.reinterpret(first_lo, 'float32x64'), weights[0])
                    acc_hi = S.vmul(T.reinterpret(first_hi, 'float32x64'), weights[0])
                    for stream in T.unroll(1, 4, explicit=True):
                        packed_next = S.vld(x_ub[token, stream, pair_col])
                        next_lo, next_hi = S.vintlv(zero_bf16, packed_next)
                        S.vmula(acc_lo, T.reinterpret(next_lo, 'float32x64'), weights[stream])
                        S.vmula(acc_hi, T.reinterpret(next_hi, 'float32x64'), weights[stream])
                    S.vsts(output_ub[token, pair_col], S.vcvt(acc_lo, T.bfloat16), dist='PK_B32')
                    S.vsts(output_ub[token, pair_col + 64], S.vcvt(acc_hi, T.bfloat16), dist='PK_B32')
            elif apply_unroll == 1:
                for chunk in T.serial(HIDDEN // 64):
                    col = chunk * 64
                    acc = S.alloc_var(T.float32)
                    first = S.vcvt(S.vld(x_ub[token, 0, col], dist='UNPK_B16'), T.float32, part=0)
                    acc = S.vmul(first, weights[0])
                    for stream in T.unroll(1, 4, explicit=True):
                        value = S.vcvt(S.vld(x_ub[token, stream, col], dist='UNPK_B16'), T.float32, part=0)
                        S.vmula(acc, value, weights[stream])
                    S.vsts(output_ub[token, col], S.vcvt(acc, T.bfloat16), dist='PK_B32')
            else:
                # Independent chunks keep their accumulators in registers.
                # Within each chunk the stream order and BF16 conversion stay the same.
                accum = S.alloc_local((apply_unroll,), T.float32)
                for group in T.serial(HIDDEN // (64 * apply_unroll)):
                    for part in T.unroll(apply_unroll, explicit=True):
                        first_col = (group * apply_unroll + part) * 64
                        first_vec = S.vcvt(S.vld(x_ub[token, 0, first_col], dist='UNPK_B16'), T.float32, part=0)
                        accum[part] = S.vmul(first_vec, weights[0])
                    for stream in T.unroll(1, 4, explicit=True):
                        for part in T.unroll(apply_unroll, explicit=True):
                            input_col = (group * apply_unroll + part) * 64
                            residual = S.vcvt(S.vld(x_ub[token, stream, input_col], dist='UNPK_B16'), T.float32, part=0)
                            S.vmula(accum[part], residual, weights[stream])
                    for part in T.unroll(apply_unroll, explicit=True):
                        output_col = (group * apply_unroll + part) * 64
                        S.vsts(output_ub[token, output_col], S.vcvt(accum[part], T.bfloat16), dist='PK_B32')


@tilelang.jit
def mhc_coeff_apply_small_pipeline(x, prev, raw, sq, scale, base, hi, pre, post, comb,
                                   num_vec_cores: int = 64, num_stages: int = 2, apply_unroll: int = 1,
                                   apply_pack128: bool = False):
    n = T.dynamic('num_tokens')
    x: T.Tensor[(n, 4, HIDDEN), T.bfloat16]
    prev: T.Tensor[(n, 4), T.float32]
    raw: T.Tensor[(1, n, 24), T.float32]
    sq: T.Tensor[(1, n), T.float32]
    scale: T.Tensor[(3,), T.float32]
    base: T.Tensor[(24,), T.float32]
    hi: T.Tensor[(n, HIDDEN), T.bfloat16]
    pre: T.Tensor[(n, 4), T.float32]
    post: T.Tensor[(n, 4), T.float32]
    comb: T.Tensor[(n, 16), T.float32]
    assert 1 <= num_stages <= 2
    assert apply_unroll in (1, 2, 4)
    with T.Kernel(num_vec_cores) as core:
        raw_ub = T.alloc_shared((2, 32), T.float32)
        sq_ub = T.alloc_shared((64,), T.float32)
        base_ub = T.alloc_shared((64,), T.float32)
        scale_ub = T.alloc_shared((3, 8), T.float32)
        # Keep one 64-FP32 vector of backing storage per output; only the
        # first two tokens are valid. This preserves the verified UB layout.
        pre_ub = T.alloc_shared((16, 4), T.float32)
        post_ub = T.alloc_shared((16, 4), T.float32)
        comb_ub = T.alloc_shared((4, 16), T.float32)
        x_ub = T.alloc_shared((2, 4, HIDDEN), T.bfloat16)
        prev_ub = T.alloc_shared((2, 8), T.float32)
        hi_ub = T.alloc_shared((2, HIDDEN), T.bfloat16)
        T.annotate_buffer_versions({x_ub: num_stages, prev_ub: num_stages, hi_ub: num_stages})
        T.annotate_buffer_versions({raw_ub: num_stages, sq_ub: num_stages,
                                    pre_ub: num_stages, post_ub: num_stages, comb_ub: num_stages})
        _load_small_static(scale, base, scale_ub, base_ub)
        jobs = T.ceildiv(T.max(T.ceildiv(n, 2) - core, 0), num_vec_cores)
        for step in T.Pipelined(jobs, num_stages=num_stages):
            token_base = (core + step * num_vec_cores) * 2
            _load_small_dynamic(raw, sq, raw_ub, sq_ub, token_base, n)
            T.copy(x[token_base:token_base + 2, :, :], x_ub)
            T.copy(prev[token_base:token_base + 2, :], prev_ub[:, 0:4])
            _small_coefficients(raw_ub, sq_ub, scale_ub, base_ub, pre_ub, post_ub, comb_ub)
            T.copy(pre_ub[0:2, :], pre[token_base:token_base + 2, :])
            T.copy(post_ub[0:2, :], post[token_base:token_base + 2, :])
            T.copy(comb_ub[0:2, :], comb[token_base:token_base + 2, :])
            _apply_small_tile(x_ub, prev_ub, hi_ub, apply_unroll, apply_pack128)
            T.copy(hi_ub, hi[token_base:token_base + 2, :])


def bind_shifted_pre(x, prev_pre, phi, alpha, bias, projection_config=None, *,
                     num_aic_cores=32, num_vec_cores=64, num_stages=2, preallocate=False,
                     apply_unroll=1, apply_pack128=False):
    """Compile the two kernels and return ``(run, configuration)``.

    Inputs are bound by reference. Each run launches projection then the fused
    tail on the current stream. With preallocate=True, results alias across calls.
    The callable does not synchronize. Core counts are physical device capacities;
    an explicit projection_config overrides the automatic projection tiling.
    """
    import torch

    if x.ndim != 3 or x.shape[1:] != (4, HIDDEN) or not x.shape[0] or x.shape[0] % 128:
        raise ValueError('Expected x[N,4,5120] with positive N divisible by 128')
    n = x.shape[0]
    if tuple(prev_pre.shape) not in ((n, 4), (n, 4, 1)):
        raise ValueError('Expected prev_pre[N,4] or prev_pre[N,4,1]')
    for name, value, shape, dtype in (
        ('x', x, (n, 4, HIDDEN), torch.bfloat16),
        ('prev_pre', prev_pre, tuple(prev_pre.shape), torch.float32),
        ('phi', phi, (24, 20480), torch.float32),
        ('alpha', alpha, (3,), torch.float32),
        ('bias', bias, (24,), torch.float32),
    ):
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f'{name} must have shape {shape} and dtype {dtype}')
        if value.device != x.device or not value.is_contiguous():
            raise ValueError(f'{name} must be contiguous and on the same device as x')
    if not isinstance(num_vec_cores, int) or num_vec_cores <= 0 or num_stages not in (1, 2):
        raise ValueError('Expected positive vector core count and one or two buffer stages')
    if apply_unroll not in (1, 2, 4):
        raise ValueError('Apply unroll must be 1, 2 or 4')
    if apply_pack128 and apply_unroll != 1:
        raise ValueError('Pack128 and Apply unroll are separate experiments')
    projection_config = dict(projection_config) if projection_config is not None else get_config(
        20480, 24, n, num_aic_cores=num_aic_cores)
    m = projection_config['block_num_tokens']
    if m not in (64, 128):
        raise ValueError('Projection M tile must be 64 or 128')
    flat, prev = x.view(n, 20480), prev_pre.view(n, 4)
    tail_config = dict(num_vec_cores=num_vec_cores, num_stages=num_stages,
                       apply_unroll=apply_unroll, apply_pack128=apply_pack128)

    def allocate():
        hi = torch.empty(n, HIDDEN, dtype=torch.bfloat16, device=x.device)
        packed = torch.empty(n * 24, dtype=torch.float32, device=x.device)
        pre, post, comb = packed.split((n * 4, n * 4, n * 16))
        raw = torch.empty(1, n, 24, dtype=torch.float32, device=x.device)
        sq = torch.empty(n // m, 2, m // 2, dtype=torch.float32, device=x.device)
        return hi, pre.view(n, 4), post.view(n, 4), comb.view(n, 16), raw, sq

    hi, pre, post, comb, raw, sq = allocate()
    projection = mhc_projection_token_pipeline.compile(
        flat, phi, raw[0], sq, residual_l2_cache_ctrl=None, **projection_config)
    fused_tail = mhc_coeff_apply_small_pipeline.compile(
        x, prev, raw, sq.view(1, n), alpha, bias, hi, pre, post, comb, **tail_config)

    if preallocate:
        sq_view = sq.view(1, n)
        raw_view = raw[0]
        outputs = (hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb.view(n, 4, 4))

        def run_preallocated():
            projection(flat, phi, raw_view, sq)
            fused_tail(x, prev, raw, sq_view, alpha, bias, hi, pre, post, comb)
            return outputs

    def run():
        hi, pre, post, comb, raw, sq = allocate()
        projection(flat, phi, raw[0], sq)
        fused_tail(x, prev, raw, sq.view(1, n), alpha, bias, hi, pre, post, comb)
        return hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb.view(n, 4, 4)

    return (run_preallocated if preallocate else run), dict(
                     projection_settings=projection_config, tail_settings=tail_config,
                     output_preallocation=preallocate, per_call_allocations=0 if preallocate else 4,
                     launch_order=('projection', 'fusedtail'), expected_kernels_per_call=2,
                     precision_policy='HF32 nearest-even; FP32 exact division; BF16 RNE Apply')

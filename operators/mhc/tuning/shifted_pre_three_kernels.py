"""Other shifted Pre projection entry points and compatibility imports."""
import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from shifted_pre_k8_preallocated import (
    _projection_body,
    bind_shifted_pre as _bind_shifted_pre,
    compute_coefficients,
    get_config,
    mhc_coeff_token_pipeline,
    mhc_pre_apply_flat_pipeline_asc,
    mhc_projection_split_partials,
)


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

    # Keep AIV0 and AIV1 on disjoint workspace planes. Within a plane
    # consecutive 64-row blocks correspond to alternate halves of each tile.
    jobs = T.ceildiv(T.max(tokens // 32 - cube_id, 0), cube_cores)
    for job in T.Serial(jobs):
        start = (cube_id + job * cube_cores) * 16
        output_start = start // 64 * 128 + vector_id * 64 + start % 64
        with T.PerCoreTask():
            for _ in T.Serial(T.max(1 - job, 0)):
                T.ascend_sync_inter_wait('PIPE_MTE2', 8)
            with T.Task():
                T.copy(partial_proj[0, vector_id, start : start + 16, 0:24], T.view(proj, (16, 24)))
                T.copy(partial_sq[0, vector_id, start : start + 16], sq)
        for k in T.Serial(1, splits):
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


def bind_shifted_pre(*args, **kwargs):
    return _bind_shifted_pre(*args, projection_kernel=mhc_projection_token_pipeline, **kwargs)

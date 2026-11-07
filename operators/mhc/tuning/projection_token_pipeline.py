"""Compatibility imports and the experimental single-kernel entry point."""
import tilelang
from tilelang.ascend import language as T

from shifted_pre_three_kernels import _projection_body, get_config, mhc_projection_token_pipeline
from shifted_mixed_epilogue import shifted_coeff_epilogue, shifted_epilogue


@tilelang.jit
def mhc_shifted_single_kernel(
    x, fn, prev_pre, alpha, bias, hi, pre, post, comb,
    block_mhc_mult3: int,
    num_aic_cores: int,
    num_k_splits: int,
    cores_per_split: int,
    num_hidden_blocks_per_split: int,
    num_l0b_stages: int,
    num_l0c_stages: int,
    num_token_blocks_per_chunk: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    mhc_hidden_size = T.const('mhc_hidden_size')
    x: T.Tensor[(num_tokens, mhc_hidden_size), T.bfloat16]
    fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32]
    prev_pre: T.Tensor[(num_tokens, 4), T.float32]
    alpha: T.Tensor[(3,), T.float32]
    bias: T.Tensor[(24,), T.float32]
    hi: T.Tensor[(num_tokens, mhc_hidden_size // 4), T.bfloat16]
    pre: T.Tensor[(num_tokens, 4), T.float32]
    post: T.Tensor[(num_tokens, 4), T.float32]
    comb: T.Tensor[(num_tokens, 16), T.float32]
    assert num_k_splits == 1
    _projection_body(
        x, fn, None, None, prev_pre, alpha, bias, hi, pre, post, comb,
        num_tokens, mhc_mult3, mhc_hidden_size,
        block_mhc_mult3, num_aic_cores, num_k_splits, cores_per_split,
        num_hidden_blocks_per_split, num_l0b_stages, num_l0c_stages,
        num_token_blocks_per_chunk, False, None, True, shifted_epilogue)


@tilelang.jit
def mhc_projection_coeff_kernel(
    x, fn, alpha, bias, pre, post, comb,
    block_mhc_mult3: int,
    num_aic_cores: int,
    num_k_splits: int,
    cores_per_split: int,
    num_hidden_blocks_per_split: int,
    num_l0b_stages: int,
    num_l0c_stages: int,
    num_token_blocks_per_chunk: int,
    block_num_tokens: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    mhc_hidden_size = T.const('mhc_hidden_size')
    x: T.Tensor[(num_tokens, mhc_hidden_size), T.bfloat16]
    fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32]
    alpha: T.Tensor[(3,), T.float32]
    bias: T.Tensor[(24,), T.float32]
    pre: T.Tensor[(num_tokens, 4), T.float32]
    post: T.Tensor[(num_tokens, 4), T.float32]
    comb: T.Tensor[(num_tokens, 16), T.float32]
    assert num_k_splits == 1 and block_num_tokens in (32, 128)
    _projection_body(
        x, fn, None, None, None, alpha, bias, None, pre, post, comb,
        num_tokens, mhc_mult3, mhc_hidden_size,
        block_mhc_mult3, num_aic_cores, num_k_splits, cores_per_split,
        num_hidden_blocks_per_split, num_l0b_stages, num_l0c_stages,
        num_token_blocks_per_chunk, False, None, True, shifted_coeff_epilogue,
        False, False, block_num_tokens)

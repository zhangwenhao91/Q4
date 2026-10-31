"""Bind compiled launch objects outside timing; retain fresh outputs and all device work."""
import hashlib

from mhc_fusion import SOURCE, allocate_outputs


def make_bound_candidate(projector, x, prev, alpha, bias):
    import torch
    import sys
    from tile_kernels.config import get_num_vec_cores
    sys.path.insert(0, str(SOURCE))
    from apply_tail import mhc_shifted_tail_apply_fwd_asc
    t, padded = projector.t, projector.padded
    prev2d = prev.view(t, 4)
    settings = dict(rms_eps=1e-20, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2., sinkhorn_repeat=20, h_blk=256, n_splits=1,
        token_block=16, num_stages=2, num_vec_cores=get_num_vec_cores())
    # Compilation inspects shape/dtype only; these buffers are not reused by run().
    sample_hi, sample_pre, sample_post, sample_comb = allocate_outputs(torch, t, x)
    sample_raw = torch.empty(1, t, 24, dtype=torch.float32, device=x.device)
    sample_sq = torch.empty(padded // 128, 2, 64, dtype=torch.float32, device=x.device)
    projection = projector.kernel.compile(projector.x, projector.phi, sample_raw[0], sample_sq,
                                          deterministic=False, **projector.config)
    sample_valid_sq = sample_sq.view(1, padded)[:, :t]
    tail = mhc_shifted_tail_apply_fwd_asc.compile(sample_raw, sample_valid_sq, alpha, bias,
        x, prev2d, sample_pre, sample_post, sample_comb, sample_hi, **settings)

    def run():
        hi, pre, post, comb = allocate_outputs(torch, t, x)
        raw = torch.empty(1, t, 24, dtype=torch.float32, device=x.device)
        sq = torch.empty(padded // 128, 2, 64, dtype=torch.float32, device=x.device)
        # The compiled objects accept runtime tensors only; no JIT argument/cache parsing here.
        projection(projector.x, projector.phi, raw[0], sq)
        valid_sq = sq.view(1, padded)
        if t != padded:
            valid_sq = valid_sq[:, :t]
        tail(raw, valid_sq, alpha, bias, x, prev2d, pre, post, comb, hi)
        return hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb.view(t, 4, 4)

    return run, dict(
        operation='public JITImpl.compile -> JITKernel call; no private cache access',
        projection_device_source_sha256=hashlib.sha256(projection.get_kernel_source().encode()).hexdigest(),
        tail_device_source_sha256=hashlib.sha256(tail.get_kernel_source().encode()).hexdigest(),
        numerical_kernel_changes=False, output_preallocation=False,
        per_call_allocations=4, expected_kernels_per_call=2,
        note='compilation excluded for all sides; all device math, output/workspace allocation and views remain timed')

"""Bind compiled launch objects outside timing; retain fresh outputs and all device work."""
import hashlib

from mhc_fusion import SOURCE, allocate_outputs


def make_bound_candidate(projector, x, prev, alpha, bias, split_apply=False, apply_h_blk=5120, tail_cores=None, tail_stages=2):
    if apply_h_blk <= 0 or apply_h_blk % 64 or 5120 % apply_h_blk:
        raise ValueError("apply_h_blk must divide 5120 and be a multiple of 64")
    import torch
    import sys
    from tile_kernels.config import get_num_vec_cores
    sys.path.insert(0, str(SOURCE))
    from apply_tail import mhc_shifted_tail_apply_fwd_asc
    t, padded = projector.t, projector.padded
    prev2d = prev.view(t, 4)
    settings = dict(rms_eps=1e-20, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2., sinkhorn_repeat=20, n_splits=1,
        token_block=16, num_stages=2, num_vec_cores=get_num_vec_cores())
    if tail_stages not in (1,2,3,4) or (tail_cores is not None and not 0 < tail_cores <= settings['num_vec_cores']):
        raise ValueError('Invalid tail core/stage configuration')
    apply_cores = settings['num_vec_cores']
    settings['num_stages'] = tail_stages
    if tail_cores is not None:
        settings['num_vec_cores'] = tail_cores
    # Compilation inspects shape/dtype only; these buffers are not reused by run().
    sample_hi, sample_pre, sample_post, sample_comb = allocate_outputs(torch, t, x)
    sample_raw = torch.empty(1, t, 24, dtype=torch.float32, device=x.device)
    sample_sq = torch.empty(padded // 128, 2, 64, dtype=torch.float32, device=x.device)
    projection = projector.kernel.compile(projector.x, projector.phi, sample_raw[0], sample_sq,
                                          deterministic=False, **projector.config)
    sample_valid_sq = sample_sq.view(1, padded)[:, :t]
    if split_apply:
        from coeff_tail import mhc_coeff_tail_fwd_asc
        from tile_kernels.mhc.pre_apply_mix_asc import mhc_pre_apply_mix_fwd_asc
        standalone_apply = mhc_pre_apply_mix_fwd_asc.compile(x, prev2d, sample_hi,
            h_blk=apply_h_blk, num_vec_cores=apply_cores)
        tail = mhc_coeff_tail_fwd_asc.compile(sample_raw, sample_valid_sq, alpha, bias,
            sample_pre, sample_post, sample_comb, hidden=5120, **settings)
    else:
        tail = mhc_shifted_tail_apply_fwd_asc.compile(sample_raw, sample_valid_sq, alpha, bias,
            x, prev2d, sample_pre, sample_post, sample_comb, sample_hi, h_blk=256, **settings)

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

    def run_split():
        hi, pre, post, comb = allocate_outputs(torch, t, x)
        standalone_apply(x, prev2d, hi)
        raw = torch.empty(1, t, 24, dtype=torch.float32, device=x.device)
        sq = torch.empty(padded // 128, 2, 64, dtype=torch.float32, device=x.device)
        projection(projector.x, projector.phi, raw[0], sq)
        valid_sq = sq.view(1, padded)
        if t != padded:
            valid_sq = valid_sq[:, :t]
        tail(raw, valid_sq, alpha, bias, pre, post, comb)
        return hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb.view(t, 4, 4)

    metadata = dict(
        tail_settings=dict(settings),
        operation='public JITImpl.compile -> JITKernel call; no private cache access',
        projection_device_source_sha256=hashlib.sha256(projection.get_kernel_source().encode()).hexdigest(),
        tail_device_source_sha256=hashlib.sha256(tail.get_kernel_source().encode()).hexdigest(),
        numerical_kernel_changes=False, output_preallocation=False,
        per_call_allocations=4, expected_kernels_per_call=3 if split_apply else 2,
        note='compilation excluded for all sides; all device math, output/workspace allocation and views remain timed')
    if split_apply:
        metadata['standalone_apply_source_sha256'] = hashlib.sha256(standalone_apply.get_kernel_source().encode()).hexdigest()
        metadata['standalone_apply_hidden_tile'] = apply_h_blk
    return (run_split if split_apply else run), metadata

"""Bound three-kernel TL implementation of original current-pre semantics."""
import hashlib
import json
import sys

from mhc_fusion import SOURCE


def make_candidate(x, phi, alpha, bias, x_cpu, phi_cpu, *, all_token_pipeline=False,
                   workspace_k8=False):
    import torch
    from tile_kernels.config import get_num_vec_cores
    from tile_kernels.mhc.pre_apply_mix_asc import mhc_pre_apply_mix_fwd_asc
    from matched_case import DynamicHCProjection
    from fused_projection import check_projection
    manifest = json.loads((SOURCE / 'source.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Fusion source hash mismatch: ' + name)
    sys.path.insert(0, str(SOURCE))
    from coeff_tail import mhc_coeff_tail_fwd_asc
    t, _, hidden = x.shape
    projector = DynamicHCProjection(x, phi)
    projection_fn = projector.kernel
    tail_fn = mhc_coeff_tail_fwd_asc
    apply_fn = mhc_pre_apply_mix_fwd_asc
    projection_options = {}
    apply_options = {}
    projection_config = projector.config
    splits = 8 if workspace_k8 else 1
    if workspace_k8 and not all_token_pipeline:
        raise ValueError('Workspace K8 requires the all-token pipeline')
    if all_token_pipeline:
        if t % 128:
            raise ValueError('All-token current-pre pipeline requires N divisible by 128')
        from apply_flat_pipeline import mhc_pre_apply_flat_pipeline_asc
        from coeff_token_pipeline import mhc_coeff_token_pipeline
        from projection_token_pipeline import mhc_projection_token_pipeline
        projection_fn = mhc_projection_token_pipeline
        tail_fn = mhc_coeff_token_pipeline
        apply_fn = mhc_pre_apply_flat_pipeline_asc
        projection_options['residual_l2_cache_ctrl'] = None
        apply_options['pipeline_stages'] = 3
    if workspace_k8:
        from shifted_pre_three_kernels import get_config, mhc_projection_split_partials
        projection_fn = mhc_projection_split_partials
        projection_config = get_config(4 * hidden, 24, t, split_override=splits)
        projection_options = {}
    padded = projector.padded

    def allocate_outputs():
        hi = torch.empty(t, hidden, dtype=x.dtype, device=x.device)
        size = (t * 4 + 7) // 8 * 8
        packed = torch.empty(size * 2 + t * 16, dtype=torch.float32, device=x.device)
        pre = packed[:t * 4].view(t, 4)
        post = packed[size:size + t * 4].view(t, 4)
        comb = packed[size * 2:].view(t, 16)
        return hi, pre, post, comb

    def allocate_workspace():
        raw = torch.empty(splits, t, 24, dtype=torch.float32, device=x.device)
        sq_shape = (splits, t) if workspace_k8 else (padded // 128, 2, 64)
        return raw, torch.empty(sq_shape, dtype=torch.float32, device=x.device)

    hi, pre, post, comb = allocate_outputs()
    raw, sq = allocate_workspace()
    cores = get_num_vec_cores()
    if workspace_k8:
        projection = projection_fn.compile(projector.x, projector.phi, raw, sq, **projection_config)
    else:
        projection = projection_fn.compile(projector.x, projector.phi, raw[0], sq,
            deterministic=False, **projection_config, **projection_options)

    def checked_projection():
        projection(projector.x, projector.phi, raw if workspace_k8 else raw[0], sq)
        if workspace_k8:
            return raw.sum(0, keepdim=True), sq.sum(0, keepdim=True)
        return raw, sq.view(1, padded)[:, :t]

    checks = check_projection(checked_projection, x_cpu, phi_cpu, t, hidden)
    if not all(item['passed'] for item in checks):
        raise AssertionError('Raw projection check failed: ' + str(checks))

    tail_options = dict(batch_partials=True) if workspace_k8 else {}
    tail = tail_fn.compile(raw, sq if workspace_k8 else sq.view(1, padded)[:, :t], alpha, bias,
        pre, post, comb, rms_eps=1e-20, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2., sinkhorn_repeat=20, hidden=hidden, n_splits=splits,
        token_block=16, num_stages=2, num_vec_cores=cores, **tail_options)
    apply = apply_fn.compile(x, pre, hi, h_blk=hidden, num_vec_cores=cores, **apply_options)

    fixed_workspace = (raw, sq) if workspace_k8 else None

    def run():
        hi, pre, post, comb = allocate_outputs()
        raw, sq = fixed_workspace or allocate_workspace()
        projection(projector.x, projector.phi, raw if workspace_k8 else raw[0], sq)
        tail(raw, sq if workspace_k8 else sq.view(1, padded)[:, :t], alpha, bias, pre, post, comb)
        # Must run AFTER coefficients: original Pre applies the newly computed pre.
        apply(x, pre, hi)
        return hi, post.unsqueeze(-1), comb.view(t, 4, 4)

    return run, dict(projection_checks=checks, hc_source_sha256=projector.source_sha256,
        source=manifest, hidden=hidden, expected_kernels_per_call=3,
        all_token_pipeline=all_token_pipeline,
        semantics='projection -> coefficients/Sinkhorn -> NEW-pre apply; three main outputs',
        output_preallocation=False, workspace_reuse=workspace_k8, projection_config=projection_config,
        per_call_allocations=2 if workspace_k8 else 4,
        device_source_sha256={name:hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
                              for name,kernel in [('projection',projection),('tail',tail),('apply',apply)]})

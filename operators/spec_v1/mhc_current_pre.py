"""Bound three-kernel TL implementation of original current-pre semantics."""
import hashlib
import json
import sys

from mhc_fusion import SOURCE


def make_candidate(x, phi, alpha, bias, x_cpu, phi_cpu):
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
    checks = check_projection(projector, x_cpu, phi_cpu, t, hidden)
    if not all(item['passed'] for item in checks):
        raise AssertionError('Raw projection check failed: ' + str(checks))
    padded = projector.padded

    def allocate():
        hi = torch.empty(t, hidden, dtype=x.dtype, device=x.device)
        size = (t * 4 + 7) // 8 * 8
        packed = torch.empty(size * 2 + t * 16, dtype=torch.float32, device=x.device)
        pre = packed[:t * 4].view(t, 4)
        post = packed[size:size + t * 4].view(t, 4)
        comb = packed[size * 2:].view(t, 16)
        raw = torch.empty(1, t, 24, dtype=torch.float32, device=x.device)
        sq = torch.empty(padded // 128, 2, 64, dtype=torch.float32, device=x.device)
        return hi, pre, post, comb, raw, sq

    hi, pre, post, comb, raw, sq = allocate()
    cores = get_num_vec_cores()
    projection = projector.kernel.compile(projector.x, projector.phi, raw[0], sq,
        deterministic=False, **projector.config)
    tail = mhc_coeff_tail_fwd_asc.compile(raw, sq.view(1, padded)[:, :t], alpha, bias,
        pre, post, comb, rms_eps=1e-20, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2., sinkhorn_repeat=20, hidden=hidden, n_splits=1,
        token_block=16, num_stages=2, num_vec_cores=cores)
    apply = mhc_pre_apply_mix_fwd_asc.compile(x, pre, hi, h_blk=hidden, num_vec_cores=cores)

    def run():
        hi, pre, post, comb, raw, sq = allocate()
        projection(projector.x, projector.phi, raw[0], sq)
        tail(raw, sq.view(1, padded)[:, :t], alpha, bias, pre, post, comb)
        # Must run AFTER coefficients: original Pre applies the newly computed pre.
        apply(x, pre, hi)
        return hi, post.unsqueeze(-1), comb.view(t, 4, 4)

    return run, dict(projection_checks=checks, hc_source_sha256=projector.source_sha256,
        source=manifest, hidden=hidden, expected_kernels_per_call=3,
        semantics='projection -> coefficients/Sinkhorn -> NEW-pre apply; three main outputs',
        output_preallocation=False, per_call_allocations=4,
        device_source_sha256={name:hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
                              for name,kernel in [('projection',projection),('tail',tail),('apply',apply)]})

"""Unmodified pinned TileKernels fused tail: public wrapper and bound launch."""
import hashlib
import importlib
from pathlib import Path


def make_candidates(x, phi, alpha, bias, x_cpu=None, phi_cpu=None):
    import torch
    from matched_case import DynamicHCProjection
    from tile_kernels.config import get_num_vec_cores
    from tile_kernels.mhc import mhc_pre_big_fuse_fwd
    from tile_kernels.mhc.pre_big_fuse_asc import mhc_pre_big_fuse_fwd_asc
    t, _, hidden = x.shape
    if t % 16:
        raise ValueError('Pinned upstream fused tail requires complete 16-token tiles')
    hashes = {
        'pre_big_fuse_kernel':'d3d8f5ec18d426215b1dc2aabb9079e54e951216e971d602c93c5fa98d9afee8',
        'pre_big_fuse_asc':'6b2a575cdce922bddabe35bf7f981073d302d889a7d578f81e2e9d01f4693929'}
    for module, expected in hashes.items():
        path = Path(importlib.import_module('tile_kernels.mhc.' + module).__file__)
        if hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() != expected:
            raise RuntimeError('Pinned upstream source mismatch: ' + str(path))
    projector = DynamicHCProjection(x, phi)
    checks = None
    if x_cpu is not None:
        from fused_projection import check_projection
        checks = check_projection(projector, x_cpu, phi_cpu, t, hidden)
        if not all(item['passed'] for item in checks):
            raise AssertionError('Raw projection check failed: ' + str(checks))
    padded = projector.padded
    settings = dict(rms_eps=1e-20, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2., sinkhorn_repeat=20, n_splits=1)

    def allocate():
        hi = torch.empty(t, hidden, device=x.device, dtype=x.dtype)
        packed = torch.empty(t * 20, device=x.device, dtype=torch.float32)
        post, comb = packed[:t*4].view(t,4), packed[t*4:].view(t,16)
        raw = torch.empty(1,t,24,device=x.device,dtype=torch.float32)
        sq = torch.empty(padded//128,2,64,device=x.device,dtype=torch.float32)
        return hi, post, comb, raw, sq

    hi, post, comb, raw, sq = allocate()
    projection = projector.kernel.compile(projector.x, projector.phi, raw[0], sq,
        deterministic=False, **projector.config)
    tail = mhc_pre_big_fuse_fwd_asc.compile(raw, sq.view(1,padded)[:,:t], alpha, bias,
        x, post, comb, hi, **settings, h_blk=256, token_block=16,
        num_stages=4, num_vec_cores=get_num_vec_cores())

    def public_wrapper():
        hi, post, comb, raw, sq = allocate()
        projector.kernel(projector.x, projector.phi, raw[0], sq,
                         deterministic=False, **projector.config)
        mhc_pre_big_fuse_fwd(raw, sq.view(1,padded)[:,:t], alpha, bias, x, post, comb, hi, **settings)
        return hi, post.unsqueeze(-1), comb.view(t,4,4)

    def bound():
        hi, post, comb, raw, sq = allocate()
        projection(projector.x, projector.phi, raw[0], sq)
        tail(raw, sq.view(1,padded)[:,:t], alpha, bias, x, post, comb, hi)
        return hi, post.unsqueeze(-1), comb.view(t,4,4)

    return {f'tilelang_upstream_big_fuse_h{hidden}':public_wrapper,
            f'tilelang_upstream_big_fuse_bound_h{hidden}':bound}, dict(
        projection_checks=checks, hc_source_sha256=projector.source_sha256,
        revision='c1cf914a82e6901c679445f872051ce44ac2cc63', source_sha256=hashes,
        kernel_changes=False, expected_kernels_per_call=2, per_call_allocations=4,
        semantics='original NEW-pre semantics, three main outputs; complete token tiles only',
        public_wrapper='upstream mhc_pre_big_fuse_fwd plus patched DeepGEMM-Ascend projection',
        bound='same numerical kernels, public compile performed outside timing',
        output_preallocation=False,
        compiled_tail_sha256=hashlib.sha256(tail.get_kernel_source().encode()).hexdigest())

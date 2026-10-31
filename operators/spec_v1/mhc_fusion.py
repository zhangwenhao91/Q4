"""Three-/two-kernel shifted mHC candidates; all numeric work remains inside timing."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'operators/mhc/tuning'


def allocate_outputs(torch, tokens, x):
    hi = torch.empty(tokens, 5120, dtype=x.dtype, device=x.device)
    # Keep every output base 32-byte aligned, including odd tail token counts.
    mix_size = ((tokens * 4 + 7) // 8) * 8
    packed = torch.empty(mix_size * 2 + tokens * 16, dtype=torch.float32, device=x.device)
    pre, post, comb = packed.split((mix_size, mix_size, tokens * 16))
    if mix_size != tokens * 4:
        pre, post = pre.narrow(0, 0, tokens * 4), post.narrow(0, 0, tokens * 4)
    return hi, pre.view(tokens, 4), post.view(tokens, 4), comb.view(tokens, 16)


def make_candidates(projector, x, prev, alpha, bias):
    import torch
    from tile_kernels.config import get_num_vec_cores
    from tile_kernels.mhc import mhc_pre_apply_mix_fwd
    manifest = json.loads((SOURCE / 'source.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Fusion source hash mismatch: ' + name)
    sys.path.insert(0, str(SOURCE))
    from coeff_tail import mhc_coeff_tail_fwd_asc
    from apply_tail import mhc_shifted_tail_apply_fwd_asc
    t = x.shape[0]
    prev2d = prev.view(t, 4)  # View shares the runtime coefficients; no numerical caching.
    settings = dict(n_splits=1, token_block=16, num_stages=2, num_vec_cores=get_num_vec_cores())

    def coeff_fused():
        hi, pre, post, comb = allocate_outputs(torch, t, x)
        mhc_pre_apply_mix_fwd(x, prev2d, hi)
        raw, sq = projector()
        mhc_coeff_tail_fwd_asc(raw, sq, alpha, bias, pre, post, comb,
            rms_eps=1e-20, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=2., sinkhorn_repeat=20, hidden=5120, **settings)
        return hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb.view(t, 4, 4)

    def apply_fused():
        hi, pre, post, comb = allocate_outputs(torch, t, x)
        raw, sq = projector()
        mhc_shifted_tail_apply_fwd_asc(raw, sq, alpha, bias, x, prev2d, pre, post, comb, hi,
            rms_eps=1e-20, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=2., sinkhorn_repeat=20, h_blk=256, **settings)
        return hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb.view(t, 4, 4)

    return {'tilelang_coeff_fused': coeff_fused, 'tilelang_apply_fused': apply_fused}, dict(
        source=manifest, settings=settings, apply_hidden_tile=256,
        expected_kernel_counts=dict(deepgemm_tilelang=5, tilelang_coeff_fused=3,
                                    tilelang_apply_fused=2, ascendc_shifted_native=2),
        allocation_boundary='four tensor allocations per candidate call: feature output, packed coefficient output, raw projection, padded square sum; views included',
        timing='all projection/apply/tail work reruns for every call; no persistent output preallocation')

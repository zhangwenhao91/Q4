"""Unmodified upstream PreV2 + Sinkhorn, with upstream current-pre semantics."""
import hashlib
import os
from pathlib import Path
import subprocess

from cases import ROOT, load_bridge, shifted_reference


def make_case(a):
    import torch
    vendor = ROOT / 'third_party/q4mhc-opp/vendors/q4mhc_transformer'
    if Path(os.environ.get('ASCEND_CUSTOM_OPP_PATH', '/missing')).resolve() != vendor.resolve():
        raise RuntimeError('Select the original q4mhc vendor, in a separate process')
    source = ROOT / 'third_party/ops-transformer-q4mhc'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != 'ba5162411c919e15fbabb28298e8b420c11ad905' or dirty.strip():
        raise RuntimeError('Original source revision mismatch or tracked source modified')
    lib = vendor / 'op_api/lib/libcust_opapi.so'
    x0 = torch.randn(a.tokens, 4, 5120, dtype=torch.bfloat16)
    # Consume the same random draws as shifted_case, for identical benchmark data.
    prev0 = torch.rand(a.tokens, 4, 1) + 1e-6
    phi0 = torch.empty(24, 20480).uniform_(-1, 1) / 20480**.5
    alpha0 = torch.tensor([.1, .1, .1])
    bias0 = torch.randn(24) * .1
    _, pre_ref, post_ref, comb_ref = shifted_reference(x0, prev0, phi0, alpha0, bias0)
    hi_ref = (x0.float() * pre_ref).sum(1).bfloat16()
    x, phi, alpha, bias = [z.npu() for z in (x0, phi0, alpha0, bias0)]
    bridge = load_bridge()

    def original():
        hi, pre, post, logits = bridge.original_pre(x, phi, alpha, bias, 1e-20, 1e-6)
        comb = bridge.sinkhorn(logits, 1e-6, 20)
        return hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb

    return {'ascendc_original_pre_sinkhorn': original}, (hi_ref, pre_ref, post_ref, comb_ref), [
        'current_pre_layer_input', 'next_pre_mix', 'post_mix', 'comb_mix'], dict(
        comparison_available=True, original_source_revision=revision,
        original_library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
        contract=dict(N=a.tokens, hidden=5120, hc_mult=4, norm_eps=1e-20,
                      pre_eps=1e-6, sinkhorn_eps=1e-6, repeat=20, post_mult=2),
        semantics='Unmodified PreV2 uses NEW pre for layer_input; not shifted semantics',
        timing_boundary='PreV2 + Sinkhorn, all output/cache/workspace allocations included; no PyTorch apply',
        weights='synthetic, same seed and random draw order as shifted_case',
        provenance_note='Pinned clean source plus installed library hash; build provenance is not a binary equivalence proof')

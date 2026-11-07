"""Unmodified upstream PreV2 + Sinkhorn, with upstream current-pre semantics."""
import hashlib
import os
from pathlib import Path
import subprocess

from cases import ROOT, load_bridge, shifted_reference


def make_case(a):
    import torch
    fused = a.suite == 'mhc_original_fused'
    big5120 = a.suite == 'mhc_original_big5120'
    fused5120 = fused and 'ascendc_adapted_fused_h5120' in (a.implementations or ())
    hidden = 4096 if fused and not fused5120 else 5120
    vendor_name = 'q4mhc5120' if fused5120 else 'q4mhc'
    vendor = ROOT / 'third_party' / f'{vendor_name}-opp' / 'vendors' / f'{vendor_name}_transformer'
    if Path(os.environ.get('ASCEND_CUSTOM_OPP_PATH', '/missing')).resolve() != vendor.resolve():
        raise RuntimeError(f'Select the {vendor_name} vendor in a separate process')
    source = ROOT / 'third_party' / f'ops-transformer-{vendor_name}'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    expected_dirty = ' M mhc/mhc_pre_sinkhorn/op_host/op_tiling/mhc_pre_sinkhorn_tiling_arch35.cpp' if fused5120 else ''
    if revision != 'ba5162411c919e15fbabb28298e8b420c11ad905' or dirty.rstrip() != expected_dirty:
        raise RuntimeError('Original source revision mismatch or tracked source modified')
    if fused5120:
        patch = ROOT / 'operators/mhc/tuning/fused_h5120.patch'
        subprocess.run(['git', '-C', str(source), 'apply', '--reverse', '--check', str(patch)], check=True)
    lib = vendor / 'op_api/lib/libcust_opapi.so'
    x0 = torch.randn(a.tokens, 4, hidden, dtype=torch.bfloat16)
    # Consume the same random draws as shifted_case, for identical benchmark data.
    prev0 = torch.rand(a.tokens, 4, 1) + 1e-6
    phi0 = torch.empty(24, 4 * hidden).uniform_(-1, 1) / (4 * hidden)**.5
    alpha0 = torch.tensor([.1, .1, .1])
    bias0 = torch.randn(24) * .1
    _, pre_ref, post_ref, comb_ref = shifted_reference(x0, prev0, phi0, alpha0, bias0)
    hi_ref = (x0.float() * pre_ref).sum(1).bfloat16()
    x, phi, alpha, bias = [z.npu() for z in (x0, phi0, alpha0, bias0)]
    if fused:
        import importlib
        module = importlib.import_module('cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn.mhc_pre_sinkhorn')
        # Bind upstream extension once, as for the other prebound API measurements.
        native_fused = module.mhc_pre_sinkhorn_op_builder.load().mhc_pre_sinkhorn
    else:
        bridge = load_bridge()

    def original():
        hi, pre, post, logits = bridge.original_pre(x, phi, alpha, bias, 1e-20, 1e-6)
        comb = bridge.sinkhorn(logits, 1e-6, 20)
        if big5120:
            # Discard next_pre only at the return boundary; all native cache work remains timed.
            return hi, post.unsqueeze(-1), comb
        return hi, pre.unsqueeze(-1), post.unsqueeze(-1), comb

    def original_fused():
        hi, post, comb, *_ = native_fused(x, phi, alpha, bias, 4, 20, 1e-6, 1e-20, False)
        return hi, post.unsqueeze(-1), comb.reshape(a.tokens, 4, 4)

    label = 'ascendc_original_fused_h4096' if fused else 'ascendc_original_pre_sinkhorn'
    if fused5120:
        label = 'ascendc_adapted_fused_h5120'
    refs = (hi_ref, post_ref, comb_ref) if fused or big5120 else (hi_ref, pre_ref, post_ref, comb_ref)
    names = ['current_pre_layer_input', 'post_mix', 'comb_mix'] if fused or big5120 else [
        'current_pre_layer_input', 'next_pre_mix', 'post_mix', 'comb_mix']
    functions = {label: original_fused if fused else original}
    tl_meta = None
    upstream_meta = None
    standalone_label = f'tilelang_current_pre_k8_fused_prealloc_h{hidden}'
    if fused:
        if standalone_label in (a.implementations or ()):
            import sys
            sys.path.insert(0, str(ROOT / 'operators/mhc/tuning'))
            from shifted_pre_k8_fused_preallocated import bind_shifted_pre
            launcher, tl_meta = bind_shifted_pre(x, None, phi, alpha, bias, current_pre=True)
            for name, generated in tl_meta.pop('generated_sources').items():
                (a.output / f'{standalone_label}_{name}.cpp').write_text(generated, encoding='utf-8')
            def standalone_current_pre():
                hi, _, post, comb = launcher()
                return hi, post, comb
            functions[standalone_label] = standalone_current_pre
        from mhc_current_pre import make_candidate
        selected = a.implementations
        pipeline_label = f'tilelang_current_pre_all_token_l2_default_h{hidden}'
        if selected and pipeline_label in selected:
            tl, tl_meta = make_candidate(x, phi, alpha, bias, x0, phi0, all_token_pipeline=True)
            functions[pipeline_label] = tl
        workspace_label = f'tilelang_current_pre_k8_batch_workspace_h{hidden}'
        if selected and workspace_label in selected:
            tl, tl_meta = make_candidate(
                x, phi, alpha, bias, x0, phi0, all_token_pipeline=True, workspace_k8=True)
            functions[workspace_label] = tl
        if not selected or 'tilelang_current_pre_h4096' in selected:
            tl, meta = make_candidate(x, phi, alpha, bias, x0, phi0)
            functions = {'tilelang_current_pre_h4096': tl, **functions}
            if tl_meta is None:
                tl_meta = meta
        want_upstream = not selected or any(name.startswith('tilelang_upstream_big_fuse_') for name in selected)
        if want_upstream and a.tokens % 16 == 0:
            from mhc_upstream_big_fuse import make_candidates
            upstream, upstream_meta = make_candidates(x, phi, alpha, bias)
            functions = {**functions, **upstream}
        elif want_upstream:
            upstream_meta = dict(skipped='Pinned upstream requires complete 16-token tiles; no padding adaptor added')
    if big5120:
        if a.tokens % 16:
            raise ValueError('5120 upstream comparison requires N divisible by 16')
        from mhc_upstream_big_fuse import make_candidates
        upstream, upstream_meta = make_candidates(x, phi, alpha, bias, x0, phi0)
        functions = {**upstream, **functions}
    return functions, refs, names, dict(
        tilelang_current_pre=tl_meta,
        tilelang_upstream_big_fuse=upstream_meta,
        projection_checks=(tl_meta or upstream_meta or {}).get('projection_checks'),
        hc_source_sha256=(tl_meta or upstream_meta or {}).get('hc_source_sha256'),
        comparison_available=True, original_source_revision=revision,
        fused_hidden_allowlist_patch=fused5120,
        original_library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
        contract=dict(N=a.tokens, hidden=hidden, hc_mult=4, norm_eps=1e-20,
                      pre_eps=1e-6, sinkhorn_eps=1e-6, repeat=20, post_mult=2),
        semantics='Unmodified upstream uses NEW pre for layer_input; not shifted semantics',
        output_boundary_note=('Three main outputs compared; AC still computes next_pre and PreV2 caches, fully timed'
                              if big5120 else None),
        timing_boundary=('Fused PreSinkhorn, needBackward=False; three main outputs, no next_pre output' if fused else
                         'PreV2 + Sinkhorn, four outputs plus PreV2 caches') +
                        ('; AC includes allocation/workspace; standalone TL preallocates outputs/workspace; extension bound before timing'
                         if standalone_label in functions else
                         '; all allocation/workspace included; no PyTorch apply; extension bound before timing'),
        weights='synthetic, same seed and random draw order as shifted_case',
        provenance_note='Pinned clean source plus installed library hash; build provenance is not a binary equivalence proof')

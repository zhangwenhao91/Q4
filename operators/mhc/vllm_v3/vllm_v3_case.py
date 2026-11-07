"""Same-input full-API benchmark against pinned upstream HcPre v3."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

from prepare import ROOT, SOURCE, RECORD, VENDOR, REV, verify, sha, extract


def load_v3():
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    if Path(os.environ.get('ASCEND_CUSTOM_OPP_PATH', '/missing')).resolve() != VENDOR.resolve():
        raise RuntimeError('Select only the isolated q4vllmv3 vendor in a fresh process')
    library = VENDOR / 'op_api/lib/libcust_opapi.so'
    if not library.is_file():
        raise RuntimeError('Run bash operators/mhc/vllm_v3/build.sh first')
    build = json.loads((RECORD / 'build.json').read_text(encoding='utf-8'))
    if build.get('operators') != ['hc_pre', 'hc_post']:
        raise RuntimeError('Run bash operators/mhc/vllm_v3/run.sh all to build Pre and Post')
    manifest = json.loads((RECORD / 'source.json').read_text(encoding='utf-8'))
    hashes = verify()
    if manifest['revision'] != REV or manifest['files'] != hashes:
        raise RuntimeError('Upstream manifest mismatch')
    bridge = RECORD / 'v3_binding.cpp'
    if bridge.read_text(encoding='utf-8') != extract((SOURCE / 'csrc/torch_binding.cpp').read_text(encoding='utf-8')):
        raise RuntimeError('Extracted v3 binding changed')
    if sha(bridge) != manifest['binding_sha256']:
        raise RuntimeError('Extracted binding hash mismatch')
    # Refuse a preloaded implementation whose provenance cannot be established.
    if hasattr(torch.ops._C_ascend, 'npu_hc_pre_v3'):
        raise RuntimeError('npu_hc_pre_v3 is already registered; run in a fresh process without importing vllm_ascend')
    npu = Path(torch_npu.__file__).resolve().parent
    cann = Path(os.environ['ASCEND_HOME_PATH'])
    adapter = SOURCE / 'csrc/aclnn_torch_adapter'
    binary = load(name='q4_vllm_hc_pre_v3_' + REV[:8],
        sources=[str(bridge), str(adapter / 'NPUBridge.cpp'), str(adapter / 'NPUStorageImpl.cpp')],
        extra_include_paths=[str(SOURCE / 'csrc'), str(npu / 'include'), str(npu.parent),
                             str(npu / 'include/third_party/acl/inc'), str(cann / 'include')],
        extra_cflags=['-O2', '-std=c++17'],
        extra_ldflags=[f'-L{npu / "lib"}', '-ltorch_npu', f'-L{cann / "lib64"}',
                       '-lascendcl', '-ldl', f'-Wl,-rpath,{npu / "lib"}'],
        is_python_module=False, verbose=True)
    lib = ctypes.CDLL(str(library))
    for symbol, address in (('aclnnHcPre', torch.ops.q4_vllm_v3.api_address()),
                            ('aclnnHcPost', torch.ops.q4_vllm_v3.post_api_address())):
        if address != ctypes.cast(getattr(lib, symbol), ctypes.c_void_p).value:
            raise RuntimeError(f'Upstream adapter resolved {symbol} from an unexpected library')
    if sha(library) != build['library_sha256'] or build['revision'] != REV:
        raise RuntimeError('Installed HcPre library differs from build record')
    return torch.ops._C_ascend.npu_hc_pre_v3, dict(
        upstream_revision=REV, upstream_source_hashes=hashes,
        library=str(library), library_sha256=sha(library), binding_sha256=sha(bridge),
        binding_binary=str(binary), resolved_api_matches_isolated_library=True,
        note=manifest['binding_note'], hf32_rounding='Upstream HcPre uses HF32 RTZ; TL projection uses its original mode; original Q4 precision gates retained')


def check_reuse_partials(x, phi, alpha, bias, raw, sums, pre, post, comb):
    import torch
    from bench_forward import sinkhorn_ref

    n = x.shape[0]
    splits = raw.shape[0]
    x = x.float().reshape(n, 4, splits, -1)
    phi = phi.reshape(24, 4, splits, -1)

    def unpack(values):
        return values.reshape(splits, 2, n // 128, 64, -1).transpose(1, 2).reshape(splits, n, -1)

    def errors(actual, expected):
        finite_actual = torch.isfinite(actual)
        finite_expected = torch.isfinite(expected)
        valid = finite_actual & finite_expected
        relative = ((actual[valid] - expected[valid]).norm() /
                    expected[valid].norm().clamp_min(1e-30)).item()
        return dict(nonfinite=int((~finite_actual).sum()), expected_nonfinite=int((~finite_expected).sum()),
                    rel_l2=round(relative, 7) if valid.any() else None)

    raw, sums = unpack(raw), unpack(sums).squeeze(-1)
    split_errors = []
    for split in range(splits):
        sub_x = x[:, :, split].flatten(1)
        sub_phi = phi[:, :, split].flatten(1)
        split_errors.append(dict(projection=errors(raw[split], sub_x @ sub_phi.T),
                                 square_sum=errors(sums[split], sub_x.square().sum(-1))))
    mix = raw.sum(0) * (sums.sum(0).unsqueeze(-1) / 20480 + 1e-20).rsqrt()
    rebuilt_pre = (mix[:, :4] * alpha[0] + bias[:4]).sigmoid() + 1e-6
    rebuilt_post = 2 * (mix[:, 4:8] * alpha[1] + bias[4:8]).sigmoid()
    rebuilt_comb = sinkhorn_ref((mix[:, 8:] * alpha[2] + bias[8:]).reshape(n, 4, 4), 20, 1e-6)
    print('Split-K diagnostic:', dict(
        partials=split_errors,
        device_vs_rebuilt=dict(pre=errors(pre, rebuilt_pre), post=errors(post, rebuilt_post),
                               comb=errors(comb, rebuilt_comb.reshape(n, 16)))),
        flush=True)


def make_case(a):
    import torch
    import sys
    sys.path.insert(0, str(ROOT / 'operators/mhc/matched'))
    from projection import DynamicHCProjection
    from mhc_bound import make_bound_candidate, make_projection_coeff_candidate, make_shifted_single_candidate
    from cases import shifted_reference
    if a.tokens % 128:
        raise ValueError('Three-kernel N-pipeline comparison requires N divisible by 128')
    l2_ab = a.mhc_projection_l2_ab
    hidden = a.mhc_vllm_hidden
    if hidden == 4096 and (
            a.mhc_vllm_stage != 'pre' or not l2_ab or
            set(a.implementations or ()) != {'ascendc_vllm_v3', 'tilelang_projection_coeff_k8_batch_prealloc'}):
        raise ValueError('H4096 v3 comparison requires Pre and the standalone batched K8 TL/v3 pair')
    include_v3 = not l2_ab or 'ascendc_vllm_v3' in (a.implementations or ())
    if include_v3:
        op, provenance = load_v3()
    t = a.tokens
    x0 = torch.randn(t, 4, hidden, dtype=torch.bfloat16)
    prev0 = torch.rand(t, 4, 1) + 1e-6
    phi0 = torch.empty(24, 4 * hidden).uniform_(-1, 1) / (4 * hidden)**.5
    alpha0 = torch.tensor([.1, .1, .1])
    bias0 = torch.randn(24) * .1
    refs = shifted_reference(x0, prev0, phi0, alpha0, bias0)
    x, prev, phi, alpha, bias = [v.npu() for v in (x0, prev0, phi0, alpha0, bias0)]
    stage = a.mhc_vllm_stage
    if stage != 'pre':
        from tile_kernels.mhc import mhc_post_fwd
        post_op = torch.ops._C_ascend.npu_hc_post
        layer0 = torch.randn(t, 5120, dtype=torch.bfloat16)
        post0, comb0 = refs[2:]
        residual_ref = (torch.einsum('nij,nih->njh', comb0, x0.float())
                        + post0 * layer0.float().unsqueeze(1)).bfloat16()
        layer, residual = layer0.npu().unsqueeze(0), x.unsqueeze(0)

        def ac_post(post, comb):
            return post_op(layer, residual, post.squeeze(-1).unsqueeze(0), comb.unsqueeze(0)).squeeze(0)

        def tl_post(post, comb):
            return mhc_post_fwd(layer, residual, post.unsqueeze(0), comb.unsqueeze(0)).squeeze(0)

        if stage == 'post':
            post, comb = post0.npu(), comb0.npu()
            return dict(ascendc_vllm_post=lambda: (ac_post(post, comb),),
                        tilelang_post=lambda: (tl_post(post, comb),)), (residual_ref,), ['new_residual'], dict(
                comparison_available=True, vllm_v3=provenance, stage=stage,
                contract=dict(N=t, hidden=5120, hc_mult=4),
                timing_boundary='Post API on identical BF16 layer output and residual with shared CPU-reference FP32 coefficients')
    prev2d = prev.view(t, 4)

    def ac():
        # These are metadata-only views. All upstream output/workspace allocation
        # and dispatch still occur on every timed invocation of the original op.
        y, post, comb, pre = op(x, phi, alpha, bias, prev2d,
            hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-20, hc_eps=1e-6)
        return y, pre.unsqueeze(-1), post.unsqueeze(-1), comb

    if hidden == 4096:
        sys.path.insert(0, str(ROOT / 'operators/mhc/tuning'))
        from shifted_pre_k8_fused_preallocated import bind_shifted_pre
        label = 'tilelang_projection_coeff_k8_batch_prealloc'
        launcher, meta = bind_shifted_pre(x, prev, phi, alpha, bias)
        for name, generated in meta.pop('generated_sources').items():
            (a.output / f'{label}_{name}.cpp').write_text(generated, encoding='utf-8')
        return {'ascendc_vllm_v3': ac, label: launcher}, refs, [
            'layer_input', 'next_pre_mix', 'post_mix', 'comb_mix'], dict(
                comparison_available=True, vllm_v3=provenance, stage=stage,
                contract=dict(N=t, hidden=hidden, hc_mult=4, shifted=True),
                variants={label: meta})
    projector = DynamicHCProjection(x, phi)
    if l2_ab:
        functions = {'ascendc_vllm_v3': ac} if include_v3 else {}
        metadata = {}
        variants = [('tilelang_all_token_h5120s3', False, None), ('tilelang_all_token_l2_default', True, None)]
        if a.implementations:
            variants.extend((f'tilelang_deferred_k{splits}', True, splits) for splits in (4, 8, 16))
            variants.append(('tilelang_deferred_k8_batch', True, 8))
            variants.extend((label, True, 8) for label in (
                'tilelang_deferred_k8_prealloc', 'tilelang_deferred_k8_batch_prealloc',
                'tilelang_deferred_k8_batch_workspace'))
        for label, use_default, deferred_splits in variants:
            if a.implementations and label not in a.implementations:
                continue
            functions[label], meta = make_bound_candidate(projector, x, prev, alpha, bias,
                split_apply=True, apply_pipeline=(5120, 3), flat_pipeline=True, all_token_pipeline=True,
                projection_l2_default=use_default, deferred_splits=deferred_splits,
                batch_partials=label.startswith('tilelang_deferred_k8_batch'),
                preallocate=label.endswith('_prealloc'), reuse_workspace=label.endswith('_workspace'))
            for name, source in meta.pop('generated_sources').items():
                (a.output / f'{label}_{name}.cpp').write_text(source, encoding='utf-8')
            metadata[label] = meta
        for projection_coeff, block_num_tokens, k_splits, batch_partials in (
                ('tilelang_projection_coeff_prealloc', 128, 1, False),
                ('tilelang_projection_coeff_m32_prealloc', 32, 1, False),
                ('tilelang_projection_coeff_k8_prealloc', 128, 8, False),
                ('tilelang_projection_coeff_k8_batch_prealloc_reference', 128, 8, True),
                ('tilelang_projection_coeff_k8_batch_prealloc', 128, 8, True)):
            if projection_coeff not in (a.implementations or ()):
                continue
            if projection_coeff == 'tilelang_projection_coeff_k8_batch_prealloc':
                sys.path.insert(0, str(ROOT / 'operators/mhc/tuning'))
                from shifted_pre_k8_fused_preallocated import bind_shifted_pre
                functions[projection_coeff], meta = bind_shifted_pre(x, prev, phi, alpha, bias)
            else:
                functions[projection_coeff], meta = make_projection_coeff_candidate(
                    x, prev, phi, alpha, bias, block_num_tokens=block_num_tokens, k_splits=k_splits,
                    batch_partials=batch_partials)
            for name, source in meta.pop('generated_sources').items():
                (a.output / f'{projection_coeff}_{name}.cpp').write_text(source, encoding='utf-8')
            metadata[projection_coeff] = meta
        reference = 'tilelang_deferred_k8_batch_prealloc_reference'
        if reference in (a.implementations or ()):
            sys.path.insert(0, str(ROOT / 'operators/mhc/tuning'))
            from shifted_pre_three_kernels import bind_shifted_pre
            functions[reference], meta = bind_shifted_pre(
                x, prev, phi, alpha, bias, deferred_splits=8, batch_partials=True, preallocate=True)
            for name, source in meta.pop('generated_sources').items():
                (a.output / f'{reference}_{name}.cpp').write_text(source, encoding='utf-8')
            metadata[reference] = meta
        for label, preallocated, apply_unroll in (
                ('tilelang_fused_small', False, 1),
                ('tilelang_fused_small_prealloc', True, 1),
                ('tilelang_fused_small_prealloc_u2', True, 2),
                ('tilelang_fused_small_prealloc_u4', True, 4),
                ('tilelang_fused_small_prealloc_pack128', True, 1)):
            if label not in (a.implementations or ()):
                continue
            from deep_gemm import _C
            from tile_kernels.config import get_num_vec_cores
            sys.path.insert(0, str(ROOT / 'operators/mhc/tuning'))
            from shifted_pre_fused import bind_shifted_pre
            functions[label], metadata[label] = bind_shifted_pre(
                x, prev, phi, alpha, bias, num_aic_cores=_C.get_num_sms(),
                num_vec_cores=get_num_vec_cores(), preallocate=preallocated, apply_unroll=apply_unroll,
                apply_pack128=label.endswith('_pack128'))
        for single_label, reuse_apply, token_stages in (
                ('tilelang_shifted_single_h5120', False, 1),
                ('tilelang_shifted_single_reuse_h5120', True, 1),
                ('tilelang_shifted_single_reuse_pipeline_h5120', True, 2)):
            if single_label not in (a.implementations or ()):
                continue
            partial_check = None
            if reuse_apply and a.check_only:
                from functools import partial
                partial_check = partial(check_reuse_partials, x0, phi0, alpha0, bias0)
            functions[single_label], meta = make_shifted_single_candidate(
                x, prev, phi, alpha, bias, reuse_apply=reuse_apply,
                reuse_token_stages=token_stages, partial_check=partial_check)
            generated = meta.pop('generated_source')
            (a.output / f'{single_label}.cpp').write_text(generated, encoding='utf-8')
            if partial_check is not None:
                import re
                arrive = re.search(r'asc_sync_inter_arrive\(\s*PIPE_MTE3\s*,\s*8\s*\)', generated)
                wait = re.search(r'asc_sync_inter_wait\(\s*PIPE_MTE2\s*,\s*8\s*\)', generated)
                if not (arrive and wait):
                    raise RuntimeError('Generated reuse kernel is missing the MTE3 arrival or MTE2 wait for flag 8')
                print('Generated reuse kernel includes both arrival and wait for flag 8', flush=True)
            metadata[single_label] = meta
        return functions, refs, ['layer_input', 'next_pre_mix', 'post_mix', 'comb_mix'], dict(
            comparison_available=True, vllm_v3=provenance if include_v3 else None,
            stage=stage, projection_config=projector.config, variants=metadata)
    baseline, baseline_meta = make_bound_candidate(projector, x, prev, alpha, bias, split_apply=True)
    candidate, candidate_meta = make_bound_candidate(projector, x, prev, alpha, bias,
        split_apply=True, apply_pipeline=(5120, 3), flat_pipeline=True, all_token_pipeline=True)
    for name, source in candidate_meta.pop('generated_sources').items():
        (a.output / f'all_token_{name}.cpp').write_text(source, encoding='utf-8')

    names = ['layer_input', 'next_pre_mix', 'post_mix', 'comb_mix']
    functions = dict(ascendc_vllm_v3=ac, tilelang_coeff_bound=baseline,
                     tilelang_all_token_h5120s3=candidate)
    if stage == 'combined':
        def chain(pre_fn, post_fn):
            def run():
                outputs = pre_fn()
                return (*outputs, post_fn(outputs[2], outputs[3]))
            return run

        functions = {name: chain(fn, ac_post if name == 'ascendc_vllm_v3' else tl_post)
                     for name, fn in functions.items()}
        refs = (*refs, residual_ref)
        names.append('new_residual')
    return functions, refs, names, dict(
        comparison_available=True, vllm_v3=provenance,
        stage=stage,
        contract=dict(N=t, hidden=5120, hc_mult=4, shifted=True,
            rms_eps=1e-20, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, sinkhorn_repeat=20, post_mult=2),
        baseline=baseline_meta, candidate=candidate_meta, projection_config=projector.config,
        timing_boundary='Full API including fresh output/workspace allocations and dispatch; input generation/CPU reference/compilation/source validation excluded',
        expected_kernel_counts=dict(ascendc_vllm_v3='verify device launches separately',
            tilelang_coeff_bound=4 if stage == 'combined' else 3,
            tilelang_all_token_h5120s3=4 if stage == 'combined' else 3))

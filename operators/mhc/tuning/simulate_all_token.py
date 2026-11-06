"""Compile three N-pipeline kernels, launch one mHC call, check CPU reference."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=int, default=2048)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=('baseline', 'candidate', 'k8-prealloc'), default='candidate')
    a = p.parse_args()
    if a.tokens <= 0 or a.tokens % 128:
        p.error('tokens must be a positive multiple of 128')
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / 'operators/spec_v1'))
    sys.path.insert(0, str(root / 'operators/mhc/matched'))
    import torch
    import torch_npu
    from projection import DynamicHCProjection
    from mhc_bound import make_bound_candidate
    from cases import shifted_reference
    torch.manual_seed(42)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = True
    x0 = torch.randn(a.tokens, 4, 5120, dtype=torch.bfloat16)
    prev0 = torch.rand(a.tokens, 4, 1, dtype=torch.float32) + 1e-6
    if a.variant == 'k8-prealloc':
        phi0 = torch.empty(24, 20480).uniform_(-1, 1) / 20480**.5
        alpha0 = torch.tensor([.1, .1, .1])
        bias0 = torch.randn(24) * .1
    else:
        phi0 = torch.randn(24, 20480, dtype=torch.float32) * .01
        alpha0 = torch.ones(3, dtype=torch.float32)
        bias0 = torch.zeros(24, dtype=torch.float32)
    expected = shifted_reference(x0, prev0, phi0, alpha0, bias0)
    x, prev, phi, alpha, bias = [v.npu() for v in (x0, prev0, phi0, alpha0, bias0)]
    if a.variant == 'k8-prealloc':
        from shifted_pre_k8_preallocated import bind_preallocated_k8
        run, meta = bind_preallocated_k8(x, prev, phi, alpha, bias)
    else:
        projector = DynamicHCProjection(x, phi)
        options = dict(split_apply=True)
        if a.variant == 'candidate':
            options.update(apply_pipeline=(5120, 3), flat_pipeline=True, all_token_pipeline=True)
        run, meta = make_bound_candidate(projector, x, prev, alpha, bias, **options)
        meta['projection_config'] = projector.config
    a.output.mkdir(parents=True, exist_ok=True)
    for name, source in meta.pop('generated_sources', {}).items():
        (a.output / f'{name}.cpp').write_text(source)
    meta.update(tokens=a.tokens, variant=a.variant, target_calls=1)
    (a.output / 'metadata.json').write_text(json.dumps(meta, indent=2))
    torch.npu.synchronize()
    with torch.no_grad():
        actual = run()
    torch.npu.synchronize()
    checks = []
    for name, got, ref in zip(('layer_input', 'next_pre_mix', 'post_mix', 'comb_mix'), actual, expected):
        got = got.cpu()
        rtol, atol = (.02, .02) if ref.dtype == torch.bfloat16 else (1e-3, 1e-4)
        passed = bool(torch.isfinite(got).all() and torch.allclose(got, ref, rtol=rtol, atol=atol))
        checks.append(dict(output=name, passed=passed,
            max_abs=(got.float() - ref.float()).abs().max().item()))
    (a.output / 'correctness.json').write_text(json.dumps(checks, indent=2))
    print('ALL TOKEN CHECK:', checks, flush=True)
    if not all(v['passed'] for v in checks):
        raise AssertionError('Three-kernel N-pipeline CPU check failed')


if __name__ == '__main__':
    main()

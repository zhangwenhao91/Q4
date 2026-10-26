"""Q4: Sinkhorn, Pre, and Pre+Sinkhorn forward validation/API comparison.

Projection backend is explicit: PyTorch baseline or fused DeepGEMM-Ascend TileLang HC kernel.
"""
import argparse
import csv
import importlib
import json
from pathlib import Path
import statistics
import time


def load_bridge():
    from torch.utils.cpp_extension import load
    # Reuse the installed package's CANN-first include/link configuration.
    module = importlib.import_module('cann_ops_transformer.ops.mhc.mhc_post.mhc_post')
    builder = module.mhc_post_op_builder
    return load(name='q4_mhc_aux',
                sources=[str(Path(__file__).with_name('aclnn_bridge.cpp'))],
                extra_include_paths=builder.get_absolute_paths(builder.include_paths()),
                extra_cflags=builder.cxx_args(), extra_ldflags=builder.extra_ldflags(),
                verbose=True)


def sinkhorn_ref(x, repeat, eps):
    z = x.softmax(-1) + eps
    z = z / (z.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        z = z / (z.sum(-1, keepdim=True) + eps)
        z = z / (z.sum(-2, keepdim=True) + eps)
    return z


def make_case(a, tokens, hidden):
    import torch
    from tile_kernels.mhc import (mhc_sinkhorn_fwd, mhc_pre_split_mixes_fwd,
                                  mhc_pre_apply_mix_fwd, mhc_pre_big_fuse_fwd,
                                  mhc_reduce_partials_and_rmsnorm_fwd)
    bridge = load_bridge()
    if a.suite == 'sinkhorn':
        cpu_x = torch.randn(tokens, 4, 4)
        x = cpu_x.npu()
        def tl():
            y = torch.empty_like(x)
            mhc_sinkhorn_fwd(x, y, a.repeat, a.eps)
            return (y,)
        def ac():
            return (bridge.sinkhorn(x, a.eps, a.repeat),)
        return tl, ac, (sinkhorn_ref(cpu_x, a.repeat, a.eps),), ['h_res'], 'tilelang', []
    cpu_x = torch.randn(tokens, 4, hidden, dtype=torch.bfloat16)
    cpu_phi = torch.empty(24, 4 * hidden).uniform_(-1, 1) / (4 * hidden) ** 0.5
    cpu_alpha = torch.tensor([0.1, 0.1, 0.1])
    cpu_bias = torch.randn(24) * 0.1
    flat = cpu_x.float().flatten(1)
    normalized = (flat @ cpu_phi.T) * (flat.square().mean(-1, keepdim=True) + a.eps).rsqrt()
    pre = (normalized[:, :4] * cpu_alpha[0] + cpu_bias[:4]).sigmoid() + a.eps
    post = 2 * (normalized[:, 4:8] * cpu_alpha[1] + cpu_bias[4:8]).sigmoid()
    logits = (normalized[:, 8:] * cpu_alpha[2] + cpu_bias[8:]).reshape(tokens, 4, 4)
    hin = (cpu_x.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
    hres = sinkhorn_ref(logits, a.repeat, a.eps) if a.suite == 'pre_sinkhorn' else logits
    x, phi, alpha, bias = [v.npu() for v in (cpu_x, cpu_phi, cpu_alpha, cpu_bias)]
    projector = None
    projection_checks = []
    if a.projection == 'deepgemm':
        from fused_projection import HCProjection, check_projection
        projector = HCProjection(x, phi, deterministic=a.dg_deterministic)
        projection_checks = check_projection(projector, cpu_x, cpu_phi, tokens, hidden)
    if a.suite == 'pre_sinkhorn':
        from cann_ops_transformer.ops import mhc_pre_sinkhorn
        def tl():
            # All projection work, allocation and fused tail are timed for either backend.
            if projector is not None:
                mul, sq = projector()
            else:
                xf = x.float().flatten(1)
                mul = (xf @ phi.T).unsqueeze(0)
                sq = xf.square().sum(-1).unsqueeze(0)
            post_out = torch.empty(tokens, 4, device=x.device)
            res_out = torch.empty(tokens, 16, device=x.device)
            hin_out = torch.empty(tokens, hidden, device=x.device, dtype=x.dtype)
            mhc_pre_big_fuse_fwd(mul, sq, alpha, bias, x, post_out, res_out, hin_out,
                                rms_eps=a.eps, mhc_pre_eps=a.eps, mhc_sinkhorn_eps=a.eps,
                                mhc_post_mult_value=2.0, sinkhorn_repeat=a.repeat, n_splits=1)
            return hin_out, post_out, res_out.reshape(tokens, 4, 4)
        def ac():
            hi, hp, hr = mhc_pre_sinkhorn(x, phi, alpha, bias, 4, a.repeat, a.eps, a.eps)
            return hi, hp, hr.reshape(tokens, 4, 4)
    else:
        def tl():
            if projector is not None:
                mul, sq = projector()
                out_mul = torch.empty(tokens, 1, 24, device=x.device)
                out_sq = torch.empty(tokens, 1, device=x.device)
                mixes = torch.empty(tokens, 24, device=x.device)
                mhc_reduce_partials_and_rmsnorm_fwd(
                    mul.view(1, tokens, 1, 24), sq.view(1, tokens, 1),
                    out_mul, out_sq, mixes, rms_group_size=4*hidden, rms_eps=a.eps, n_splits=1)
            else:
                xf = x.float().flatten(1)
                mixes = (xf @ phi.T) * (xf.square().mean(-1, keepdim=True) + a.eps).rsqrt()
            pre_out = torch.empty(tokens, 4, device=x.device)
            post_out = torch.empty_like(pre_out)
            res_out = torch.empty(tokens, 16, device=x.device)
            hin_out = torch.empty(tokens, hidden, device=x.device, dtype=x.dtype)
            mhc_pre_split_mixes_fwd(mixes, alpha, bias, pre_out, post_out, res_out, 2.0, a.eps)
            mhc_pre_apply_mix_fwd(x, pre_out, hin_out)
            return hin_out, post_out, res_out.reshape(tokens, 4, 4)
        def ac():
            fn = bridge.pre_hf32 if a.pre_mode == 'hf32' else bridge.pre
            return fn(x, phi, alpha, bias, a.eps)
    label = 'deepgemm_tilekernels' if projector is not None else 'torch_projection_plus_tilekernels'
    return tl, ac, (hin, post, hres), ['h_in', 'h_post', 'h_res'], label, projection_checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=['sinkhorn', 'pre', 'pre_sinkhorn'], required=True)
    p.add_argument('--projection', choices=['torch', 'deepgemm'], default='torch')
    p.add_argument('--pre-mode', choices=['fp32', 'hf32'], default=None)
    p.add_argument('--dg-deterministic', action='store_true')
    p.add_argument('--tokens', nargs='+', type=int, default=[512])
    p.add_argument('--hidden', nargs='+', type=int, default=[4096])
    p.add_argument('--repeat', type=int, default=20)
    p.add_argument('--eps', type=float, default=1e-6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--profile', choices=['tilelang', 'ascendc'])
    p.add_argument('--output', required=True, help='New output directory; never overwrite prior measurements')
    a = p.parse_args()
    if min(a.tokens + a.hidden) <= 0 or a.iters < 1 or a.rounds < 1 or a.warmup < 0:
        p.error('Invalid shape or repeat count')
    if a.eps != 1e-6 or not 1 <= a.repeat <= 100:
        p.error('Use eps=1e-6 and repeat in 1..100')
    if a.suite == 'pre_sinkhorn' and (a.repeat != 20 or any(d not in [4096, 7168] for d in a.hidden)):
        p.error('AscendC pre_sinkhorn requires repeat=20, hidden=4096/7168')
    if a.suite != 'sinkhorn' and any(t % 16 for t in a.tokens):
        p.error('Use tokens divisible by 16 for the fused path')
    if a.projection == 'deepgemm' and a.suite == 'sinkhorn':
        p.error('Sinkhorn has no projection; use default --projection torch')
    if a.pre_mode is None:
        a.pre_mode = 'hf32' if a.projection == 'deepgemm' or a.suite == 'pre_sinkhorn' else 'fp32'
    if (a.projection == 'deepgemm' or a.suite == 'pre_sinkhorn') and a.pre_mode != 'hf32':
        p.error('DeepGEMM HC projection and PreSinkhorn require the HF32 comparison mode')
    import torch
    import torch_npu
    import tilelang
    torch.npu.set_device(0)
    # Match aclnnMhcPre FP32 and MhcPreSinkhorn HF32 paths respectively.
    torch.npu.matmul.allow_hf32 = a.pre_mode == 'hf32'
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(args=vars(a), torch=torch.__version__, torch_npu=torch_npu.__version__,
                tilelang=tilelang.__version__, tilelang_path=tilelang.__file__,
                device=torch.npu.get_device_name(0), allow_hf32=torch.npu.matmul.allow_hf32)
    if a.projection == 'deepgemm':
        import deep_gemm
        from fused_projection import DEEPGEMM_REVISION
        meta.update(deepgemm_version=deep_gemm.__version__, deepgemm_path=deep_gemm.__file__,
                    deepgemm_expected_revision=DEEPGEMM_REVISION,
                    deepgemm_kernel_hf32_rounding='nearest_even')
    (out/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta), flush=True)
    rows = []
    checks = []
    with torch.inference_mode():
        for t in a.tokens:
            for d in ([0] if a.suite == 'sinkhorn' else a.hidden):
                torch.manual_seed(a.seed)
                tl, ac, refs, names, label, projection_checks = make_case(a, t, d)
                checks.extend(projection_checks)
                (out/'correctness.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
                if not all(r['passed'] for r in projection_checks):
                    raise AssertionError('Raw fused projection/sqrsum validation failed; see correctness.json')
                actual = {}
                for impl, fn in [(label, tl), ('ascendc', ac)]:
                    actual[impl] = tuple(y.cpu() for y in fn())
                    for name, y, ref in zip(names, actual[impl], refs):
                        delta = y.float() - ref.float()
                        checks.append(dict(tokens=t, hidden=d, implementation=impl, output=name,
                                           max_abs=delta.abs().max().item(),
                                           rmse=delta.square().mean().sqrt().item()))
                        (out/'correctness.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
                        # Smoke thresholds only. Stop on failures; do not silently relax.
                        if y.dtype == torch.bfloat16:
                            torch.testing.assert_close(y, ref, rtol=0.02, atol=0.02)
                        else:
                            torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-4)
                # Also verify the two implementations agree at the same thresholds.
                for y, z in zip(actual[label], actual['ascendc']):
                    torch.testing.assert_close(y, z, rtol=0.02 if y.dtype == torch.bfloat16 else 1e-3,
                                               atol=0.02 if y.dtype == torch.bfloat16 else 1e-4)
                print('Correctness passed:', a.suite, t, d, flush=True)
                if a.check_only:
                    continue
                for fn in [tl, ac]:
                    for _ in range(a.warmup):
                        y = fn()
                    torch.npu.synchronize()
                if a.profile:
                    fn = tl if a.profile == 'tilelang' else ac
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                        schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/f'profile-{t}-{d}')),
                        record_shapes=False, profile_memory=False, with_stack=False,
                    ) as prof:
                        for _ in range(5):
                            for _ in range(10):
                                y = fn()
                            torch.npu.synchronize()
                            prof.step()
                    continue
                times = {label: [], 'ascendc': []}
                for k in range(a.rounds):
                    order = [(label, tl), ('ascendc', ac)]
                    if k % 2:
                        order.reverse()
                    for impl, fn in order:
                        start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
                        torch.npu.synchronize()
                        wall = time.perf_counter()
                        start.record()
                        for _ in range(a.iters):
                            y = fn()
                        end.record()
                        torch.npu.synchronize()
                        wall_us = (time.perf_counter()-wall)*1e6/a.iters
                        us = start.elapsed_time(end)*1000/a.iters
                        times[impl].append(us)
                        rows.append(dict(suite=a.suite, tokens=t, hidden=d, repeat=a.repeat,
                                         projection=a.projection, pre_mode=a.pre_mode,
                                         implementation=impl, round=k, event_us=us, wall_us=wall_us))
                with (out/'timings.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                med = {impl: statistics.median(v) for impl, v in times.items()}
                print(a.suite, t, d, med, 'AscendC/other=', med['ascendc']/med[label], flush=True)


if __name__ == '__main__':
    main()

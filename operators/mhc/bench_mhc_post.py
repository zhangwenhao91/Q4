"""Same-input mHC Post comparison; run on Ascend 950 Linux."""
import argparse
import csv
import json
import platform
import statistics
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--tokens', type=int, nargs='+', default=[512, 8192])
    p.add_argument('--hidden', type=int, nargs='+', default=[4096, 5120, 7168])
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--output', default='mhc_post.csv')
    a = p.parse_args()
    assert a.iters > 0 and a.rounds > 0 and a.warmup >= 0
    assert all(v > 0 for v in a.tokens + a.hidden)
    import torch
    import torch_npu
    import tilelang
    from cann_ops_transformer.ops import mhc_post
    from tile_kernels.mhc import mhc_post_fwd
    from tile_kernels.config import is_ascend
    torch.npu.set_device(a.device)
    assert is_ascend(), 'Ascend not detected; check /dev/davinci_manager'
    torch.manual_seed(20261026)
    print(json.dumps(dict(python=platform.python_version(), torch=torch.__version__,
                         torch_npu=torch_npu.__version__, tilelang=tilelang.__version__,
                         device=torch.npu.get_device_name(a.device), args=vars(a))))
    rows = []
    with torch.inference_mode():
        for t in a.tokens:
            for d in a.hidden:
                r = torch.randn(1, t, 4, d, dtype=torch.bfloat16).npu()
                x = torch.randn(1, t, d, dtype=torch.bfloat16).npu()
                c = torch.randn(1, t, 4, 4).softmax(-1).npu()
                h = torch.rand(1, t, 4).mul(2).npu()
                hp = h.unsqueeze(-1)
                tl = lambda: mhc_post_fwd(x, r, hp, c)
                ac = lambda: mhc_post(r, c, x, h)
                ref = (torch.einsum('bsij,bsid->bsjd', c.cpu(), r.cpu().float())
                       + x.cpu().float().unsqueeze(-2) * hp.cpu()).bfloat16()
                errors = {}
                for name, fn in [('tilelang', tl), ('ascendc', ac)]:
                    y = fn().cpu()
                    errors[name] = (y.float() - ref.float()).abs().max().item()
                    torch.testing.assert_close(y, ref, rtol=0.02, atol=0.02)
                    for _ in range(a.warmup):
                        fn()
                    torch.npu.synchronize()
                samples = {'tilelang': [], 'ascendc': []}
                for k in range(a.rounds):
                    order = [('tilelang', tl), ('ascendc', ac)]
                    if k % 2:
                        order.reverse()
                    for name, fn in order:
                        start = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        torch.npu.synchronize()
                        wall = time.perf_counter()
                        start.record()
                        for _ in range(a.iters):
                            y = fn()
                        end.record()
                        torch.npu.synchronize()
                        wall_us = (time.perf_counter() - wall) * 1e6 / a.iters
                        event_us = start.elapsed_time(end) * 1000 / a.iters
                        samples[name].append(event_us)
                        rows.append(dict(tokens=t, hidden=d, streams=4, dtype='bf16',
                                         implementation=name, round=k, event_us=event_us,
                                         wall_us=wall_us, max_abs_error=errors[name]))
                med = {k: statistics.median(v) for k, v in samples.items()}
                print(t, d, med, 'AscendC_us/TileLang_us=', med['ascendc']/med['tilelang'], flush=True)
                with open(a.output, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)


if __name__ == '__main__':
    main()

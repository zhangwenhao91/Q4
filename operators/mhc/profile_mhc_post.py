"""Profile one mHC Post implementation, separately from latency benchmarks."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--impl', choices=['tilelang', 'ascendc'], required=True)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import torch
    import torch_npu
    torch.npu.set_device(a.device)
    torch.manual_seed(20261026)
    r = torch.randn(1, a.tokens, 4, a.hidden, dtype=torch.bfloat16).npu()
    x = torch.randn(1, a.tokens, a.hidden, dtype=torch.bfloat16).npu()
    c = torch.randn(1, a.tokens, 4, 4).softmax(-1).npu()
    h = torch.rand(1, a.tokens, 4).mul(2).npu()
    hp = h.unsqueeze(-1)
    if a.impl == 'tilelang':
        from tile_kernels.mhc import mhc_post_fwd
        fn = lambda: mhc_post_fwd(x, r, hp, c)
    else:
        from cann_ops_transformer.ops import mhc_post
        fn = lambda: mhc_post(r, c, x, h)
    Path(a.output).mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        # Complete compilation, transfers and warmup before collecting.
        for _ in range(200):
            y = fn()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(a.output),
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as prof:
            for step in range(5):
                with torch.profiler.record_function('mhc_post_' + a.impl):
                    for _ in range(10):
                        y = fn()
                torch.npu.synchronize()
                prof.step()
    print('Profile collection finished:', a.output)
    print('Expected active region: 3 steps x 10 API calls; inspect all kernels per call.')


if __name__ == '__main__':
    main()

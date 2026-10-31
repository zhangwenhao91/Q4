"""Use the patched HC kernel's explicit padded square-sum storage for tail tokens."""
import hashlib
from pathlib import Path


class DynamicHCProjection:
    def __init__(self, x, phi):
        import torch
        import deep_gemm
        from deep_gemm.tilelang_ops import hc_prenorm
        if x.ndim != 3 or x.shape[1] != 4 or x.shape[2] not in (4096, 5120) or x.dtype != torch.bfloat16:
            raise ValueError('Expected BF16 residual[N,4,H], H=4096 or 5120')
        if phi.shape != (24, 4 * x.shape[2]) or phi.dtype != torch.float32:
            raise ValueError('Expected FP32 fn[24,4*H]')
        if not x.is_contiguous() or not phi.is_contiguous() or phi.device != x.device:
            raise ValueError('Expected contiguous tensors on the same NPU')
        source = Path(hc_prenorm.__file__).read_text()
        expected = 'sqrsum: T.Tensor[(T.ceildiv(num_tokens, block_num_tokens), num_aivs, half_block_num_tokens), T.float32]'
        if expected not in source:
            raise RuntimeError('Install the existing Q4 explicit-AIV HC layout patch before using tail storage')
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.t = x.shape[0]
        self.padded = (self.t + 127) // 128 * 128
        self.x, self.phi, self.torch = x.flatten(1), phi, torch
        deep_gemm.use_deterministic_algorithms(False)
        self.kernel = hc_prenorm.tf32_hc_prenorm_gemm_kernel
        self.config = hc_prenorm.get_config(4 * x.shape[2], 24, self.t, deterministic=False)

    def __call__(self):
        torch = self.torch
        mul = torch.empty(1, self.t, 24, dtype=torch.float32, device=self.x.device)
        sq = torch.empty(self.padded // 128, 2, 64, dtype=torch.float32, device=self.x.device)
        self.kernel(self.x, self.phi, mul[0], sq, deterministic=False, **self.config)
        # All valid rows are written by the kernel; the unused storage tail is never read.
        return mul, sq.view(1, self.padded)[:, :self.t]

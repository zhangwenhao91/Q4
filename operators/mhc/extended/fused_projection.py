"""Pinned DeepGEMM-Ascend HC projection adapter (no PyTorch numerical fallback)."""
DEEPGEMM_REVISION = '3a35326271b069b8b5bf1a754435acd6cc91575b'


class HCProjection:
    def __init__(self, x, phi, deterministic=False):
        import torch
        import deep_gemm
        assert x.dtype == torch.bfloat16 and phi.dtype == torch.float32
        assert x.device.type == 'npu' and phi.device == x.device
        assert x.ndim == 3 and x.shape[1] == 4 and x.is_contiguous()
        assert phi.shape == (24, x.shape[1] * x.shape[2]) and phi.is_contiguous()
        assert phi.shape[1] % 256 == 0
        self.x = x.flatten(1)  # View only; no cast/copy or projection precomputation.
        self.phi = phi
        self.torch = torch
        self.fn = deep_gemm.tf32_hc_prenorm_gemm
        deep_gemm.use_deterministic_algorithms(deterministic)

    def __call__(self):
        torch = self.torch
        t = self.x.shape[0]
        mul = torch.empty(1, t, 24, dtype=torch.float32, device=self.x.device)
        sq = torch.empty(1, t, dtype=torch.float32, device=self.x.device)
        self.fn(self.x, self.phi, mul, sq, num_splits=1)
        return mul, sq


def check_projection(projector, cpu_x, cpu_phi, tokens, hidden):
    """Check raw products/squared sums independently of normalized downstream outputs."""
    import torch
    flat = cpu_x.float().flatten(1)
    refs = (flat @ cpu_phi.T, flat.square().sum(-1))
    values = projector()
    records = []
    for name, value, ref in zip(['projection', 'sqrsum'], values, refs):
        value = value[0].cpu()
        delta = value.double() - ref.double()
        denom = (value.double().square() + ref.double().square()).sum()
        # Algebraically identical to upstream calc_diff, evaluated in FP64 to avoid cancellation.
        diff = (delta.square().sum() / denom).item() if denom.item() else 0.0
        record = dict(tokens=tokens, hidden=hidden, implementation='deepgemm_hc_projection',
                      output=name, max_abs=delta.abs().max().item(),
                      rmse=delta.square().mean().sqrt().item(), relative_energy_diff=diff,
                      passed=bool(torch.isfinite(value).all()) and diff < 1e-6)
        records.append(record)
    return records

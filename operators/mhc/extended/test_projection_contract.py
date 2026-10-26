"""CPU-only checks of the upstream adapter contract; no NPU execution is claimed."""
import sys
import types
import unittest
from unittest.mock import patch
from fused_projection import HCProjection


class Tensor:
    def __init__(self, shape, dtype, device=None):
        self.shape, self.dtype = shape, dtype
        self.device = device or types.SimpleNamespace(type='npu')
        self.ndim = len(shape)
    def is_contiguous(self):
        return True
    def flatten(self, start):
        assert start == 1
        return Tensor((self.shape[0], self.shape[1]*self.shape[2]), self.dtype, self.device)


class ProjectionContract(unittest.TestCase):
    def test_layout_and_fused_call(self):
        calls = []
        fake_torch = types.SimpleNamespace(bfloat16='bf16', float32='fp32',
                                           empty=lambda *shape, dtype, device: Tensor(shape, dtype, device))
        fake_dg = types.SimpleNamespace(
            use_deterministic_algorithms=lambda x: calls.append(('deterministic', x)),
            tf32_hc_prenorm_gemm=lambda a, b, d, q, **kw: calls.append((a, b, d, q, kw)))
        for tokens, hidden in [(512, 4096), (8192, 7168)]:
            with self.subTest(tokens=tokens, hidden=hidden), patch.dict(sys.modules, torch=fake_torch, deep_gemm=fake_dg):
                device = types.SimpleNamespace(type='npu')
                x = Tensor((tokens, 4, hidden), 'bf16', device)
                phi = Tensor((24, 4*hidden), 'fp32', device)
                projector = HCProjection(x, phi)
                mul, sq = projector()
                a, b, d, q, kw = calls[-1]
                self.assertEqual(a.shape, (tokens, 4*hidden))
                self.assertIs(b, phi)
                self.assertIs(d, mul)
                self.assertIs(q, sq)
                self.assertEqual(mul.shape, (1, tokens, 24))
                self.assertEqual(sq.shape, (1, tokens))
                self.assertEqual(kw, {'num_splits': 1})
    def test_missing_dependency_has_no_numerical_fallback(self):
        fake_torch = types.SimpleNamespace()
        with patch.dict(sys.modules, torch=fake_torch, deep_gemm=None):
            with self.assertRaises(ModuleNotFoundError):
                HCProjection(None, None)


if __name__ == '__main__':
    unittest.main()

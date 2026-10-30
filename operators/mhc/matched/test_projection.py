"""CPU adapter tests: check tail storage and calls, not device numerical correctness."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from projection import DynamicHCProjection


class Tensor:
    def __init__(self, shape, dtype='fp32', device=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = device or types.SimpleNamespace(type='npu')
        self.ndim = len(shape)

    def is_contiguous(self):
        return True

    def flatten(self, start):
        assert start == 1
        return Tensor((self.shape[0], self.shape[1] * self.shape[2]), self.dtype, self.device)

    def view(self, *shape):
        from math import prod
        assert prod(shape) == prod(self.shape)
        return Tensor(shape, self.dtype, self.device)

    def __getitem__(self, index):
        if isinstance(index, int):
            return Tensor(self.shape[1:], self.dtype, self.device)
        return Tensor((self.shape[0], index[1].stop), self.dtype, self.device)


class ProjectionTests(unittest.TestCase):
    def test_tail_storage_is_padded_without_padding_input_or_projection(self):
        for n in (1, 17, 127, 128, 129, 513, 8192):
            calls = []
            device = types.SimpleNamespace(type='npu')
            hc = types.SimpleNamespace(__file__=str(Path(__file__)),
                get_config=lambda k, c, t, **kw: dict(test_tokens=t),
                tf32_hc_prenorm_gemm_kernel=lambda *args, **kw: calls.append((args, kw)))
            dg = types.SimpleNamespace(use_deterministic_algorithms=lambda _: None)
            torch = types.SimpleNamespace(bfloat16='bf16', float32='fp32',
                empty=lambda *shape, dtype, device: Tensor(shape, dtype, device))
            source = 'sqrsum: T.Tensor[(T.ceildiv(num_tokens, block_num_tokens), num_aivs, half_block_num_tokens), T.float32]'
            with self.subTest(n=n), patch.dict(sys.modules, {
                'torch': torch, 'deep_gemm': dg,
                'deep_gemm.tilelang_ops': types.SimpleNamespace(hc_prenorm=hc)
            }), patch.object(Path, 'read_text', return_value=source):
                projector = DynamicHCProjection(Tensor((n, 4, 5120), 'bf16', device),
                                                Tensor((24, 20480), 'fp32', device))
                mul, sq = projector()
                args, kwargs = calls[0]
                self.assertEqual(args[0].shape, (n, 20480))
                self.assertEqual(args[2].shape, (n, 24))
                self.assertEqual(args[3].shape, ((n + 127) // 128, 2, 64))
                self.assertEqual(mul.shape, (1, n, 24))
                self.assertEqual(sq.shape, (1, n))
                self.assertEqual(kwargs, dict(deterministic=False, test_tokens=n))


if __name__ == '__main__':
    unittest.main()

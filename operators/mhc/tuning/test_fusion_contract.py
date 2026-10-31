"""CPU contracts for packed outputs and per-call launch boundaries; no NPU claims."""
import ast
from collections import Counter
import inspect
from math import prod
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'operators/spec_v1'))
from mhc_fusion import allocate_outputs, make_candidates
from bench import implementation_order


class Tensor:
    def __init__(self, shape, dtype, device, storage=None, offset=0):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.storage = storage if storage is not None else object()
        self.offset = offset

    def view(self, *shape):
        assert prod(shape) == prod(self.shape)
        return Tensor(shape, self.dtype, self.device, self.storage, self.offset)

    def split(self, lengths):
        assert sum(lengths) == prod(self.shape)
        result, offset = [], self.offset
        for length in lengths:
            result.append(Tensor((length,), self.dtype, self.device, self.storage, offset))
            offset += length
        return tuple(result)

    def unsqueeze(self, axis):
        assert axis == -1
        return Tensor((*self.shape, 1), self.dtype, self.device, self.storage, self.offset)

    def narrow(self, dim, start, length):
        assert dim == 0 and len(self.shape) == 1 and start + length <= self.shape[0]
        return Tensor((length,), self.dtype, self.device, self.storage, self.offset + start)


def fake_torch():
    return types.SimpleNamespace(float32='fp32',
        empty=lambda *shape, dtype, device: Tensor(shape, dtype, device))


def bound_kernel(filename, name, events, label):
    source = ast.parse((Path(__file__).parent / filename).read_text(encoding='utf-8'))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
    signature = inspect.Signature([inspect.Parameter(p.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                   for p in function.args.args])
    def call(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        events.append((label, values))
    return call


class FusionContracts(unittest.TestCase):
    def test_packed_outputs_are_contiguous_disjoint_and_have_required_dtypes(self):
        for t in (1, 129, 512, 8192):
            x = Tensor((t, 4, 5120), 'bf16', 'npu:0')
            hi, pre, post, comb = allocate_outputs(fake_torch(), t, x)
            self.assertEqual(hi.shape, (t, 5120))
            self.assertEqual(hi.dtype, 'bf16')
            self.assertEqual([z.shape for z in (pre, post, comb)], [(t, 4), (t, 4), (t, 16)])
            self.assertTrue(all(z.dtype == 'fp32' for z in (pre, post, comb)))
            self.assertIs(pre.storage, post.storage)
            self.assertIs(post.storage, comb.storage)
            self.assertIsNot(hi.storage, pre.storage)
            stride = ((4*t + 7)//8)*8
            self.assertEqual([z.offset for z in (pre, post, comb)], [0, stride, 2*stride])
            self.assertTrue(all(z.offset * 4 % 32 == 0 for z in (pre, post, comb)))
            self.assertLessEqual(pre.offset + prod(pre.shape), post.offset)
            self.assertLessEqual(post.offset + prod(post.shape), comb.offset)
            self.assertEqual(comb.offset + prod(comb.shape), 2*stride + 16*t)

    def test_candidates_recompute_projection_and_use_old_pre_each_call(self):
        t = 129
        events = []
        coeff = bound_kernel('coeff_tail.py', 'mhc_coeff_tail_fwd_asc', events, 'coeff')
        apply = bound_kernel('apply_tail.py', 'mhc_shifted_tail_apply_fwd_asc', events, 'fused_apply')
        def standalone(x, prev, hi):
            events.append(('standalone_apply', dict(x=x, prev=prev, hi=hi)))
        def project():
            events.append(('projection', {}))
            return Tensor((1, t, 24), 'fp32', 'npu:0'), Tensor((1, t), 'fp32', 'npu:0')
        modules = {
            'torch': fake_torch(),
            'tile_kernels.config': types.SimpleNamespace(get_num_vec_cores=lambda: 56),
            'tile_kernels.mhc': types.SimpleNamespace(mhc_pre_apply_mix_fwd=standalone),
            'coeff_tail': types.SimpleNamespace(mhc_coeff_tail_fwd_asc=coeff),
            'apply_tail': types.SimpleNamespace(mhc_shifted_tail_apply_fwd_asc=apply),
        }
        x = Tensor((t, 4, 5120), 'bf16', 'npu:0')
        prev = Tensor((t, 4, 1), 'fp32', 'npu:0')
        alpha = Tensor((3,), 'fp32', 'npu:0')
        bias = Tensor((24,), 'fp32', 'npu:0')
        with patch.dict(sys.modules, modules):
            functions, _ = make_candidates(project, x, prev, alpha, bias)
            for name, expected in [('tilelang_coeff_fused', ['standalone_apply', 'projection', 'coeff']),
                                   ('tilelang_apply_fused', ['projection', 'fused_apply'])]:
                previous = None
                for _ in range(2):
                    events.clear()
                    outputs = functions[name]()
                    self.assertEqual([e[0] for e in events], expected)
                    self.assertEqual([z.shape for z in outputs], [(t,5120),(t,4,1),(t,4,1),(t,4,4)])
                    if previous is not None:
                        self.assertIsNot(previous[0].storage, outputs[0].storage)
                        self.assertIsNot(previous[1].storage, outputs[1].storage)
                    previous = outputs
                    if name == 'tilelang_apply_fused':
                        values = events[-1][1]
                        self.assertIs(values['prev_pre_mix'].storage, prev.storage)
                        self.assertIsNot(values['prev_pre_mix'].storage, values['pre_mix'].storage)
                        self.assertEqual(values['sinkhorn_repeat'], 20)
                        self.assertEqual(values['rms_eps'], 1e-20)

    def test_sixteen_rounds_balance_every_implementation_position(self):
        labels = ['original', 'coeff', 'apply', 'ascendc']
        positions = Counter((label, position) for rnd in range(16)
                            for position, label in enumerate(implementation_order(labels, rnd)))
        self.assertEqual(set(positions.values()), {4})
        self.assertEqual(len(positions), 16)


if __name__ == '__main__':
    unittest.main()

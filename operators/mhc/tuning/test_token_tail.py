"""Execute the actual scalar tail-load AST on CPU to check bounds and zero fill."""
import ast
from contextlib import nullcontext
import hashlib
from itertools import product
import json
from pathlib import Path
import types
import unittest

SOURCE = Path(__file__).parent


class Grid:
    def __init__(self, shape, value=None):
        self.shape, self.value, self.writes = shape, value, {}

    def index(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if len(key) != len(self.shape) or any(i < 0 or i >= n for i, n in zip(key, self.shape)):
            raise IndexError((key, self.shape))
        return key

    def __getitem__(self, key):
        key = self.index(key)
        if key in self.writes:
            return self.writes[key]
        return self.value(key) if self.value is not None else -777.

    def __setitem__(self, key, value):
        self.writes[self.index(key)] = value


def parallel(*shape):
    return range(shape[0]) if len(shape) == 1 else product(*(range(n) for n in shape))


class TailLoads(unittest.TestCase):
    def test_scalar_tail_loads_never_read_outside_input_and_fill_every_lane(self):
        fake_t = types.SimpleNamespace(SimtVF=lambda **kw: nullcontext(), Parallel=parallel,
                                       float32=float, bfloat16=float)
        for filename in ('coeff_tail.py', 'apply_tail.py'):
            tree = ast.parse((SOURCE / filename).read_text(encoding='utf-8'))
            branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == 'token_base + tb <= num_tokens']
            self.assertEqual(len(branches), 2 if filename == 'apply_tail.py' else 1)
            for n in (1, 17, 127, 129, 513):
                with self.subTest(filename=filename, n=n):
                    base = (n - 1) // 16 * 16
                    env = dict(T=fake_t, token_base=base, tb=16, num_tokens=n,
                        mhc3=24, mhc=4, h_blk=64, i0_h=79,
                        gemm_out_mul=Grid((1,n,24), lambda i: float(i[1]*24+i[2]+1)),
                        gemm_out_sqrsum=Grid((1,n), lambda i: float(i[1]+1)),
                        prev_pre_mix=Grid((n,4), lambda i: float(i[0]*4+i[1]+1)),
                        residual=Grid((n,4,5120), lambda i: float(i[0]*20480+i[1]*5120+i[2]+1)),
                        mul_flat_ub=Grid((16*24,)), sq_ub=Grid((16,)),
                        prev_ub=Grid((16,4)), x_ub=Grid((16,4,64)))
                    for branch in branches:
                        code = ast.Module(body=branch.orelse, type_ignores=[])
                        exec(compile(code, filename, 'exec'), env)
                    for t in range(16):
                        valid = base + t < n
                        self.assertEqual(env['sq_ub'][t], float(base+t+1) if valid else 0.)
                        for c in range(24):
                            expected = float((base+t)*24+c+1) if valid else 0.
                            self.assertEqual(env['mul_flat_ub'][t*24+c], expected)
                        if filename == 'apply_tail.py':
                            for m in range(4):
                                expected = float((base+t)*4+m+1) if valid else 0.
                                self.assertEqual(env['prev_ub'][t,m], expected)
                                for h in range(64):
                                    expected = float((base+t)*20480+m*5120+79*64+h+1) if valid else 0.
                                    self.assertEqual(env['x_ub'][t,m,h], expected)

    def test_arithmetic_is_unchanged_and_dma_padding_option_is_absent(self):
        manifest = json.loads((SOURCE / 'source.json').read_text())
        for filename in ('coeff_tail.py', 'apply_tail.py'):
            tree = ast.parse((SOURCE / filename).read_text(encoding='utf-8'))
            phase = next(node for node in ast.walk(tree) if isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == 'SimdVF' for item in node.items))
            digest = hashlib.sha256(ast.dump(phase, include_attributes=False).encode()).hexdigest()
            self.assertEqual(digest, manifest['coefficient_arithmetic_ast_sha256'])
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'copy':
                    self.assertFalse(any(k.arg in ('pad_value', 'data_select') for k in node.keywords))


if __name__ == '__main__':
    unittest.main()

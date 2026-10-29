"""Offline cache-policy scope checks; no NPU/cache performance claims."""
import ast
from pathlib import Path
import unittest

from l2_variant import transform,CONFIGS,Q_LOAD,O_STORE

ROOT=Path(__file__).resolve().parents[3]


class L2Test(unittest.TestCase):
    def test_only_q_and_o_policies_change(self):
        original=(ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()
        best=transform(original,'tilelang_kv2')
        for label,changed in CONFIGS.items():
            with self.subTest(label=label):
                result=transform(original,label).decode()
                tree=ast.parse(result)
                calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call)
                       and any(k.arg=='l2_cache_ctrl' for k in x.keywords)]
                self.assertEqual(len(calls),(2 if 'Q' in changed else 0)+(1 if 'O' in changed else 0))
                for call in calls:
                    # No K/V copy can acquire a cache hint in this experiment.
                    source=ast.unparse(call.args[0])
                    destination=ast.unparse(call.args[1])
                    self.assertTrue(source.startswith('Q[') or destination.startswith('O['))
                    self.assertEqual(next(ast.literal_eval(k.value) for k in call.keywords
                                          if k.arg=='l2_cache_ctrl'),4)
                restored=result
                for old in (Q_LOAD,O_STORE):
                    restored=restored.replace(old[:-1]+', l2_cache_ctrl=4)',old)
                self.assertEqual(restored.encode(),best)

    def test_pin_and_configuration_guard(self):
        with self.assertRaises(RuntimeError):transform(b'wrong source','tilelang_q_l2')
        original=(ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()
        with self.assertRaises(ValueError):transform(original,'unknown')


if __name__=='__main__':unittest.main()

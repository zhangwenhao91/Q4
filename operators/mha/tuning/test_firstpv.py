"""Source scope and first-block initialization checks without an NPU."""
import ast
from pathlib import Path
import re
import unittest

import numpy as np

from firstpv_variant import transform, best_transform, PV, FIRST_PV, OLD_AUTO, NEW_AUTO

ROOT = Path(__file__).resolve().parents[3]


class FirstPVTest(unittest.TestCase):
    def test_only_initialization_and_first_accumulation_change(self):
        original = (ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()
        result = transform(original).decode()
        ast.parse(result)
        restored = result.replace(PV+FIRST_PV, PV).replace(NEW_AUTO, OLD_AUTO)
        restored, count = re.subn(
            r'(?m)^([ ]+)if manual_schedule:\n\1    clear_O_ub\(O_ub\)$',
            lambda m: m[1]+'clear_O_ub(O_ub)', restored)
        self.assertEqual(count,2)
        self.assertEqual(restored.encode(),best_transform(original))

    def test_first_copy_does_not_read_old_output(self):
        tree = ast.parse(transform((ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()))
        first = next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='pv_first')
        branch = next(x for x in first.body if isinstance(x,ast.If))
        self.assertEqual(ast.unparse(branch.test),'kv == 0')
        self.assertEqual(len(branch.body),1)
        self.assertIn('O_ub[0:ROWS, 0:D]',ast.unparse(branch.body[0]))
        self.assertNotIn('O_tmp_ub',ast.unparse(branch.body[0]))
        auto = next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='kv_auto')
        conditional = auto.body[-1]
        self.assertIsInstance(conditional,ast.If)
        self.assertEqual(ast.unparse(conditional.test),'kv > 0')
        self.assertIn('accumulate_output',ast.unparse(conditional))

    def test_online_output_matches_without_zero_initialization(self):
        rng = np.random.default_rng(42)
        for count in (1,2,32):
            blocks = rng.normal(size=(count,64,128)).astype(np.float32)
            alpha = rng.uniform(0,1,size=(count,64,1)).astype(np.float32)
            alpha[0] = 0  # exp(-inf - first finite max)
            reference = np.zeros((64,128),dtype=np.float32)
            candidate = np.full_like(reference,np.nan)
            for i in range(count):
                reference = reference*alpha[i]+blocks[i]
                candidate = blocks[i].copy() if i==0 else candidate*alpha[i]+blocks[i]
            np.testing.assert_array_equal(candidate,reference)

    def test_source_pin(self):
        with self.assertRaises(RuntimeError):transform(b'wrong source')


if __name__=='__main__':unittest.main()

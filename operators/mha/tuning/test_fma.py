"""Offline source-scope and FP32 FMA-rounding checks; no NPU required."""
import ast
from pathlib import Path
import unittest
import numpy as np
from fma_variant import transform,OLD,NEW
ROOT=Path(__file__).resolve().parents[3]


class FmaTest(unittest.TestCase):
    def test_exact_reversible_source_change(self):
        original=(ROOT/'operators'/'spec_v1'/'upstream_mha'/'core.py').read_bytes()
        result=transform(original).decode('utf-8')
        ast.parse(result)
        self.assertEqual(result.count(NEW),1)
        self.assertEqual(result.replace(NEW,OLD).encode('utf-8'),original)

    def test_pin_guard(self):
        with self.assertRaises(RuntimeError): transform(b'wrong source')

    def test_one_rounding_can_differ_from_separate_operations(self):
        # FP32-exact inputs make the separate product round to 1, losing a small term.
        alpha=np.float32(1+2**-23);previous=np.float32(1-2**-23);current=np.float32(-1)
        separate=np.float32(np.float32(alpha*previous)+current)
        fused=np.float32(np.float64(alpha)*np.float64(previous)+np.float64(current))
        self.assertEqual(float(separate),0.0)
        self.assertEqual(float(fused),-2**-46)
        self.assertNotEqual(separate,fused)  # Numerical validation cannot assume bitwise equality.


if __name__=='__main__': unittest.main()

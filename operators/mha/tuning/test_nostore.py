"""Offline scope and positive-scale/max equivalence; NPU validation remains required."""
import ast
from pathlib import Path
import unittest
import numpy as np
from nostore_variant import transform,CHANGES
ROOT=Path(__file__).resolve().parents[3]

class NoStoreTest(unittest.TestCase):
    def test_source_scope(self):
        original=(ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()
        changed=transform(original).decode('utf-8')
        ast.parse(changed)
        for old,new in reversed(CHANGES):
            self.assertEqual(changed.count(new),1)
            changed=changed.replace(new,old)
        self.assertEqual(changed.encode(),original)

    def test_pin(self):
        with self.assertRaises(RuntimeError): transform(b'wrong source')

    def test_score_and_max_equivalence(self):
        scale=np.float32(1/np.sqrt(128))
        rng=np.random.default_rng(42)
        for scores in [rng.normal(size=(1024,128)).astype(np.float32),
                       np.full((4,128),-1000,dtype=np.float32),
                       np.zeros((4,128),dtype=np.float32)]:
            scaled=scores*scale
            max_old=np.maximum(scaled[:,:64],scaled[:,64:]).max(axis=1)
            max_new=np.maximum(scores[:,:64],scores[:,64:]).max(axis=1)*scale
            np.testing.assert_array_equal(max_old,max_new)
            # Same deinterleaved even/odd score vectors reach exponentiation.
            for parity in (0,1):
                np.testing.assert_array_equal(scaled[:,parity::2],scores[:,parity::2]*scale)

if __name__=='__main__': unittest.main()

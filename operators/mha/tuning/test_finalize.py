import ast
from pathlib import Path
import sys
import unittest
import numpy as np
from finalize_variant import transform,OLD,NEW
from nostore_variant import transform as nostore_transform
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'operators/spec_v1'))
from bench import implementation_order

class FinalizeTest(unittest.TestCase):
    def test_only_finalization_changes_relative_to_best(self):
        original=(ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()
        result=transform(original).decode();ast.parse(result)
        self.assertEqual(result.replace(NEW,OLD).encode(),nostore_transform(original))
        self.assertNotIn('S.vsts(O_ub',NEW)
        self.assertEqual(NEW.count('T.SimdVF'),1)

    def test_pin(self):
        with self.assertRaises(RuntimeError):transform(b'wrong source')

    def test_inplace_forward_bf16_packing(self):
        rng=np.random.default_rng(44)
        data=rng.normal(size=(64,128)).astype(np.float32)
        denom=rng.uniform(.1,100,size=64).astype(np.float32)
        def bf16(x):
            bits=x.view(np.uint32)
            return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
        expected=bf16(data/denom[:,None]).ravel()
        storage=bytearray(data.tobytes())
        src=np.frombuffer(storage,dtype=np.float32)
        dst=np.frombuffer(storage,dtype=np.uint16)
        # Output is a BF16 view of FP32 O_ub. Ascending chunks must not clobber unread input.
        for row in range(64):
            for col in (0,64):
                offset=row*128+col
                value=src[offset:offset+64].copy()/denom[row]
                dst[offset:offset+64]=bf16(value)
        np.testing.assert_array_equal(dst[:64*128],expected)

    def test_eight_rounds_balance_four_paths(self):
        labels=['original','best','fused','ascendc']
        for label in labels:
            positions=[implementation_order(labels,i).index(label) for i in range(8)]
            self.assertEqual([positions.count(i) for i in range(4)],[2]*4)

if __name__=='__main__':unittest.main()

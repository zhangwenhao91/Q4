"""Offline checks for experiment configuration and five-way timing order."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from stages_variant import program

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'spec_v1'))
from bench import implementation_order


class StagesTest(unittest.TestCase):
    def test_only_stage_depth_differs(self):
        core=SimpleNamespace(FwdTiling=lambda **kw:kw,
                             flash_attention_fwd=lambda **kw:kw)
        two=program(core,2)
        three=program(core,3)
        self.assertEqual(two['tiling'].pop('num_stages'),2)
        self.assertEqual(three['tiling'].pop('num_stages'),3)
        self.assertEqual(two,three)
        self.assertEqual(two['out_dtype'],'bfloat16')
        self.assertEqual(two['tiling'],dict(block_q=128,block_kv=128))

    def test_unsupported_depth_rejected(self):
        with self.assertRaises(ValueError):program(None,4)

    def test_ten_rounds_balance_five_implementations(self):
        labels=['original_s3','nostore_s3','original_s2','nostore_s2','ascendc']
        counts=Counter()
        for i in range(10):
            order=implementation_order(labels,i)
            self.assertEqual(set(order),set(labels))
            for position,label in enumerate(order):counts[label,position]+=1
        self.assertTrue(all(counts[label,position]==2
                            for label in labels for position in range(5)))


if __name__=='__main__':unittest.main()

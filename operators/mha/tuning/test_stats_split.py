"""Offline change-scope and five-way timing-balance checks."""
import ast
from collections import Counter
from pathlib import Path
import sys
import unittest

from stats_split_variant import CONFIGS, transform, best_transform

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'operators/spec_v1'))
from bench import implementation_order


class StatsSplitTest(unittest.TestCase):
    def setUp(self):
        self.original = (ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()

    def test_only_requested_versions_change(self):
        best = best_transform(self.original)
        for label,changed in CONFIGS.items():
            with self.subTest(label=label):
                result = transform(self.original,label).decode()
                ast.parse(result)
                restored = result
                for name in changed:
                    self.assertEqual(result.count(name+': 2,'),2)
                    self.assertNotIn(name+': 3,',result)
                    restored = restored.replace(name+': 2,',name+': 3,')
                self.assertEqual(restored.encode(),best)

    def test_unknown_configuration_and_source_rejected(self):
        with self.assertRaises(ValueError):transform(self.original,'unknown')
        with self.assertRaises(RuntimeError):transform(b'wrong source','tilelang_s2buf')

    def test_ten_rounds_balance_five_paths(self):
        labels=['original',*CONFIGS,'ascendc']
        self.assertEqual(len(labels),5)
        counts=Counter()
        for i in range(10):
            for position,label in enumerate(implementation_order(labels,i)):
                counts[label,position]+=1
        self.assertTrue(all(counts[label,position]==2
                            for label in labels for position in range(5)))


if __name__=='__main__':unittest.main()

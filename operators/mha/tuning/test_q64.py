"""Offline checks for exact-source adaptation and balanced three-way timing."""
import ast
from collections import Counter
from pathlib import Path
import sys
import unittest
from q64_variant import transform,STAT_NAMES
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'operators'/'spec_v1'))
from bench import implementation_order


class UndoPadding(ast.NodeTransformer):
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='ROW_STATS' for t in node.targets): return None
        return self.generic_visit(node)
    def visit_Name(self,node):
        if node.id=='ROW_STATS': return ast.copy_location(ast.Name(id='ROWS',ctx=node.ctx),node)
        return node


class Q64Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=(ROOT/'operators'/'spec_v1'/'upstream_mha'/'core.py').read_bytes()

    def test_only_statistics_storage_changes(self):
        result,changed=transform(self.original)
        expected={name:2 for name in STAT_NAMES}
        self.assertEqual(changed,expected)
        normalized=UndoPadding().visit(ast.parse(result))
        self.assertEqual(ast.dump(normalized,include_attributes=False),
                         ast.dump(ast.parse(self.original),include_attributes=False))

    def test_exact_pin_guard(self):
        with self.assertRaisesRegex(RuntimeError,'exact pinned'):
            transform(self.original+b'\n# modified\n')

    def test_row_storage_covers_full_simd_load(self):
        for block_q,logical,allocated in [(64,32,64),(128,64,64)]:
            self.assertEqual(block_q//2,logical)
            self.assertEqual(max(logical,64),allocated)
            self.assertGreaterEqual(allocated,64)

    def test_three_way_order_positions_balanced(self):
        names=['tilelang_q128','tilelang_q64','ascendc']
        counts=Counter()
        for rnd in range(9):
            order=implementation_order(names,rnd)
            self.assertEqual(set(order),set(names))
            for pos,name in enumerate(order): counts[name,pos]+=1
        for name in names:
            for pos in range(3): self.assertEqual(counts[name,pos],3)

    def test_two_way_order_preserved(self):
        self.assertEqual(implementation_order(['tl','ac'],0),['tl','ac'])
        self.assertEqual(implementation_order(['tl','ac'],1),['ac','tl'])


if __name__=='__main__': unittest.main()

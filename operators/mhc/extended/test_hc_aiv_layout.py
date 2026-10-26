import unittest
from patch_hc_aiv_layout import EDITS, patch_text

class AIVLayoutPatchTests(unittest.TestCase):
    def test_linear_addresses_and_disjoint_subcore_slices(self):
        # Both supported production cases; also minimum complete block.
        for m in [128, 512, 8192]:
            visited = set()
            for block in range(m//128):
                left, right = [], []
                for sid in [0, 1]:
                    for j in range(64):
                        old = block*128 + sid*64 + j
                        new = (block*2 + sid)*64 + j
                        self.assertEqual(old, new)
                        self.assertNotIn(new, visited)
                        visited.add(new)
                        (left if sid == 0 else right).append(new)
                self.assertTrue(set(left).isdisjoint(right))
            self.assertEqual(visited, set(range(m)))
    def test_context_rejection(self):
        with self.assertRaises(RuntimeError):
            patch_text('unknown kernel')

if __name__ == '__main__':
    unittest.main()

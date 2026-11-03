import ast
from collections import Counter
from pathlib import Path
import unittest


class FlatPipelineContract(unittest.TestCase):
    def test_every_output_tile_owned_once_without_invalid_tokens(self):
        for tokens in (1,17,55,56,57,64,128,129,256,512,8192):
            for block in (1024,1280,5120):
                tiles = 5120 // block
                owned = Counter()
                for core in range(56):
                    jobs = (max(tokens-core,0)+55)//56
                    for step in range(jobs*tiles):
                        token = core + step//tiles*56
                        hidden_tile = step%tiles
                        self.assertLess(token,tokens)
                        owned[token,hidden_tile] += 1
                self.assertEqual(len(owned),tokens*tiles)
                self.assertTrue(all(v==1 for v in owned.values()))

    def test_vector_arithmetic_identical_to_nested_candidate(self):
        def vf(name):
            tree = ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
            return next(n for n in ast.walk(tree) if isinstance(n,ast.With)
                        and isinstance(n.items[0].context_expr,ast.Call)
                        and getattr(n.items[0].context_expr.func,'attr',None)=='SimdVF')
        self.assertEqual(ast.dump(vf('apply_pipeline.py')),ast.dump(vf('apply_flat_pipeline.py')))

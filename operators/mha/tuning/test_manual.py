import unittest
from types import SimpleNamespace
from manual_variant import program


class ManualContractTest(unittest.TestCase):
    def test_only_existing_manual_schedule_enabled(self):
        captured={}
        def tiling(**kw): captured['tiling']=kw;return 'tiling'
        def build(**kw): captured['builder']=kw;return 'program'
        self.assertEqual(program(SimpleNamespace(FwdTiling=tiling,flash_attention_fwd=build)),'program')
        self.assertEqual(captured['tiling'],dict(block_q=128,block_kv=128,num_stages=3))
        kw=captured['builder']
        self.assertTrue(kw['manual_schedule'])
        self.assertEqual((kw['q_len'],kw['kv_len'],kw['head_dim']),(4096,4096,128))
        self.assertEqual(kw['out_dtype'],'bfloat16')
        self.assertAlmostEqual(kw['softmax_scale'],128**-.5)


if __name__=='__main__': unittest.main()

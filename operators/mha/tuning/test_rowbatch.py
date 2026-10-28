import ast
import textwrap
from pathlib import Path
import unittest
import numpy as np
from rowbatch_variant import transform,NEW_SOFTMAX
ROOT=Path(__file__).resolve().parents[3]

class RowBatchTest(unittest.TestCase):
    def test_only_softmax_changes(self):
        old=(ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()
        text=old.decode();new=transform(old).decode();ast.parse(new)
        a=text.index('    @T.macro\n    def softmax(');b=text.index('    @T.macro\n    def pack_p(',a)
        self.assertEqual(new,text[:a]+NEW_SOFTMAX+text[b:])
        # Probability packing/scatter is copied exactly from the pinned source.
        start='                even_bf16_0 ='
        pack=text[text.index(start):text.index('\n            S.mem_bar',text.index(start))]
        self.assertIn(pack,new)

    def test_statistics_stay_batched(self):
        tree=ast.parse(textwrap.dedent(NEW_SOFTMAX))
        fn=tree.body[0];vf=fn.body[0]
        loops=[node for node in vf.body if isinstance(node,ast.For)]
        self.assertEqual(len(loops),1)
        self.assertNotIn('alpha_reg',ast.unparse(loops[0]))
        self.assertEqual(NEW_SOFTMAX.count('alpha_reg = S.vexpdif'),1)
        self.assertNotIn('S.vsts(S_ub',NEW_SOFTMAX)

    def test_pin(self):
        with self.assertRaises(RuntimeError):transform(b'wrong source')

    def test_online_row_stats(self):
        rng=np.random.default_rng(43);scale=np.float32(1/np.sqrt(128))
        m=np.full(64,-np.inf,dtype=np.float32);l=np.zeros(64,dtype=np.float32)
        m2=m.copy();l2=l.copy()
        for _ in range(32):
            scores=rng.normal(size=(64,128)).astype(np.float32)*scale
            new_m=np.maximum(m,np.maximum(scores[:,:64],scores[:,64:]).max(axis=1))
            e=np.exp(scores-new_m[:,None]);alpha=np.exp(m-new_m)
            sums=(e[:,::2]+e[:,1::2]).sum(axis=1,dtype=np.float32)
            l=l*alpha+sums;m=new_m
            # Per-row register path: even/odd max grouping; same sum grouping.
            for row in range(64):
                even=scores[row,::2];odd=scores[row,1::2]
                mr=np.maximum(m2[row],np.maximum(even,odd).max())
                ar=np.exp(m2[row]-mr)
                er0=np.exp(even-mr);er1=np.exp(odd-mr)
                lr=l2[row]*ar+(er0+er1).sum(dtype=np.float32)
                m2[row]=mr;l2[row]=lr
            np.testing.assert_array_equal(m,m2)
            np.testing.assert_array_equal(l,l2)

if __name__=='__main__': unittest.main()

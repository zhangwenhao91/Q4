"""Verify native timing boundaries without torch or an NPU."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'spec_v1'))
from mha_bf16 import native_runner


class Output:
    # Deliberately exposes only view: output cast/numerical fallback would fail.
    def view(self,*shape):
        assert shape==(4096,128)
        return self


class Ops:
    def __init__(self): self.calls=[];self.output=Output()
    def flash_attn(self,q,k,v,**kwargs):
        self.calls.append(((q,k,v),kwargs))
        return self.output,object()


class NativeBoundaryTest(unittest.TestCase):
    def test_cached_metadata_and_native_call_contract(self):
        ops=Ops();metadata=object()
        def forbidden(): raise AssertionError('metadata generation leaked into cached call')
        runner=native_runner(ops,'q','k','v',forbidden,metadata,'cached')
        self.assertIs(runner()[0],ops.output);runner()
        self.assertEqual(len(ops.calls),2)
        for args,kw in ops.calls:
            self.assertEqual(args,('q','k','v'));self.assertIs(kw['metadata'],metadata)
            self.assertAlmostEqual(kw['softmax_scale'],1/math.sqrt(128))
            self.assertEqual(kw['mask_mode'],0);self.assertFalse(kw['return_softmax_lse'])
            self.assertEqual((kw['win_left'],kw['win_right']),(-1,-1))
            self.assertEqual((kw['layout_q'],kw['layout_kv'],kw['layout_out']),('BSND',)*3)
            self.assertEqual((kw['max_seqlen_q'],kw['max_seqlen_kv']),(4096,4096))

    def test_per_call_generates_metadata_exactly_once(self):
        ops=Ops();generated=[]
        def prepare():
            value=object();generated.append(value);return value
        runner=native_runner(ops,'q','k','v',prepare,object(),'per_call')
        runner();runner()
        self.assertEqual(len(generated),2)
        for (_,kw),expected in zip(ops.calls,generated): self.assertIs(kw['metadata'],expected)

    def test_invalid_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            native_runner(Ops(),None,None,None,lambda:None,None,'unknown')


if __name__=='__main__': unittest.main()

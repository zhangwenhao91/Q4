"""CPU launch ABI checks; compiled NPU execution remains a server check."""
import sys
import types
import unittest
from unittest.mock import patch

from test_fusion_contract import Tensor, prod
from mhc_bound import make_bound_candidate


class ViewTensor(Tensor):
    def view(self, *shape):
        assert prod(shape) == prod(self.shape)
        return ViewTensor(shape, self.dtype, self.device, self.storage, self.offset)

    def __getitem__(self, key):
        if isinstance(key, int):
            return ViewTensor(self.shape[1:], self.dtype, self.device, self.storage,
                              self.offset + key * prod(self.shape[1:]))
        self_rows, cols = key
        assert self_rows == slice(None) and cols.start is None
        return ViewTensor((self.shape[0], cols.stop), self.dtype, self.device, self.storage, self.offset)


class Compiler:
    def __init__(self, name, count, events):
        self.name, self.count, self.events, self.compiles = name, count, events, []

    def __call__(self, *args, **kwargs):
        raise AssertionError('Bound path must not return to JIT dispatcher')

    def compile(self, *args, **kwargs):
        self.compiles.append((args, kwargs))
        def runtime(*tensors, **kw):
            assert not kw and len(tensors) == self.count
            self.events.append((self.name, tensors))
        runtime.get_kernel_source = lambda: 'device-source-' + self.name
        return runtime


class BoundLaunchContract(unittest.TestCase):
    def test_compiles_once_recomputes_twice_and_never_reuses_sample_outputs(self):
        for t in (128, 129, 512):
            with self.subTest(t=t):
                padded = (t+127)//128*128
                events = []
                projection, tail = Compiler('projection',4,events), Compiler('tail',10,events)
                torch = types.SimpleNamespace(float32='fp32',
                    empty=lambda *shape,dtype,device: ViewTensor(shape,dtype,device))
                x = ViewTensor((t,4,5120),'bf16','npu:0')
                prev = ViewTensor((t,4,1),'fp32','npu:0')
                phi = ViewTensor((24,20480),'fp32','npu:0')
                alpha = ViewTensor((3,),'fp32','npu:0')
                bias = ViewTensor((24,),'fp32','npu:0')
                projector = types.SimpleNamespace(t=t,padded=padded,x=x.view(t,20480),phi=phi,
                                                   kernel=projection,config={'config':1})
                with patch.dict(sys.modules, {
                    'torch': torch,
                    'tile_kernels.config': types.SimpleNamespace(get_num_vec_cores=lambda:56),
                    'apply_tail': types.SimpleNamespace(mhc_shifted_tail_apply_fwd_asc=tail)
                }):
                    run, meta = make_bound_candidate(projector,x,prev,alpha,bias)
                    self.assertEqual(events, [])  # Compilation performs no numerical calls.
                    self.assertEqual(len(projection.compiles),1)
                    self.assertEqual(len(tail.compiles),1)
                    previous = None
                    for _ in range(2):
                        events.clear()
                        outputs = run()
                        self.assertEqual([e[0] for e in events],['projection','tail'])
                        self.assertEqual([z.shape for z in outputs],[(t,5120),(t,4,1),(t,4,1),(t,4,4)])
                        project_args, tail_args = events[0][1], events[1][1]
                        self.assertIsNot(project_args[2].storage, projection.compiles[0][0][2].storage)
                        self.assertIsNot(tail_args[9].storage, tail.compiles[0][0][9].storage)
                        self.assertIs(tail_args[9].storage, outputs[0].storage)
                        self.assertEqual(tail_args[1].shape,(1,t))
                        self.assertIs(tail_args[2],alpha)
                        self.assertIs(tail_args[5].storage,prev.storage)
                        if previous is not None:
                            self.assertIsNot(outputs[0].storage,previous[0].storage)
                            self.assertIsNot(outputs[1].storage,previous[1].storage)
                        previous = outputs
                    self.assertFalse(meta['output_preallocation'])
                    self.assertEqual(len(projection.compiles),1)
                    self.assertEqual(len(tail.compiles),1)


if __name__ == '__main__':
    unittest.main()

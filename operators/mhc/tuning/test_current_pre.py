"""CPU launch dependency check: apply must consume this call's newly written pre."""
import sys
import types
import unittest
from unittest.mock import patch

from test_bound import Compiler
from test_fusion_contract import Tensor, prod
from mhc_current_pre import make_candidate


class View(Tensor):
    def view(self, *shape):
        assert prod(shape) == prod(self.shape)
        return View(shape, self.dtype, self.device, self.storage, self.offset)

    def __getitem__(self, key):
        if isinstance(key, int):
            return View(self.shape[1:], self.dtype, self.device, self.storage,
                        self.offset + key * prod(self.shape[1:]))
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            assert step == 1
            return View((stop-start,), self.dtype, self.device, self.storage, self.offset+start)
        return View((self.shape[0], key[1].stop), self.dtype, self.device, self.storage, self.offset)


class CurrentPreContract(unittest.TestCase):
    def test_new_pre_dependency_and_fresh_outputs(self):
        for t in (1, 64, 129, 8192):
            with self.subTest(t=t):
                events = []
                projection = Compiler('projection', 4, events)
                tail = Compiler('tail', 7, events)
                apply = Compiler('apply', 3, events)
                x = View((t,4,4096), 'bf16', 'npu:0')
                phi = View((24,16384), 'fp32', 'npu:0')
                alpha, bias = View((3,), 'fp32', 'npu:0'), View((24,), 'fp32', 'npu:0')
                projector = types.SimpleNamespace(padded=(t+127)//128*128, x=x.view(t,16384),
                    phi=phi, kernel=projection, config={}, source_sha256='test')
                modules = {
                    'torch':types.SimpleNamespace(float32='fp32', empty=lambda *shape,dtype,device:View(shape,dtype,device)),
                    'tile_kernels.config':types.SimpleNamespace(get_num_vec_cores=lambda:56),
                    'tile_kernels.mhc.pre_apply_mix_asc':types.SimpleNamespace(mhc_pre_apply_mix_fwd_asc=apply),
                    'coeff_tail':types.SimpleNamespace(mhc_coeff_tail_fwd_asc=tail),
                    'matched_case':types.SimpleNamespace(DynamicHCProjection=lambda *args:projector),
                    'fused_projection':types.SimpleNamespace(check_projection=lambda *args:[{'passed':True}])}
                with patch.dict(sys.modules, modules):
                    run, meta = make_candidate(x, phi, alpha, bias, None, None)
                    self.assertEqual(events, [])
                    self.assertEqual(apply.compiles[0][1]['h_blk'],4096)
                    self.assertEqual(tail.compiles[0][1]['hidden'],4096)
                    previous = None
                    for _ in range(2):
                        events.clear()
                        out = run()
                        self.assertEqual([e[0] for e in events], ['projection','tail','apply'])
                        coeff_pre, apply_pre = events[1][1][4], events[2][1][1]
                        self.assertIs(coeff_pre, apply_pre)
                        self.assertIsNot(apply_pre.storage, tail.compiles[0][0][4].storage)
                        self.assertEqual([v.shape for v in out],[(t,4096),(t,4,1),(t,4,4)])
                        self.assertLessEqual(apply_pre.offset+t*4, out[1].offset)
                        if previous:
                            self.assertIsNot(out[0].storage,previous[0].storage)
                            self.assertIsNot(out[1].storage,previous[1].storage)
                        previous = out
                    self.assertEqual(meta['expected_kernels_per_call'],3)

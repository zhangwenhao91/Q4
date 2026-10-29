"""Offline annotation-scope checks; actual slot allocation needs NPU codegen."""
import ast
from pathlib import Path
import unittest

from pbuffer_variant import transform, best_transform, OLD_VERSION, NEW_VERSION

ROOT = Path(__file__).resolve().parents[3]


class PBufferTest(unittest.TestCase):
    def test_only_p_l1_annotation_changes(self):
        original = (ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()
        result = transform(original).decode()
        self.assertEqual(result.replace(NEW_VERSION,OLD_VERSION).encode(),best_transform(original))
        tree = ast.parse(result)
        annotations = [node for node in ast.walk(tree) if isinstance(node,ast.Call)
                       and ast.unparse(node.func)=='T.annotate_buffer_versions']
        self.assertEqual(len(annotations),2)
        for call in annotations:
            versions = {ast.unparse(k):ast.literal_eval(v)
                        for k,v in zip(call.args[0].keys,call.args[0].values)}
            self.assertEqual(versions,dict(K_shared=2,V_shared=2,qk_acc_l0c=2,pv_acc_l0c=2,
                                          S_ub=3,P_shared=2,P_nz_ub=2,alpha_ub=3,O_tmp_ub=2))

    def test_wrong_source_rejected(self):
        with self.assertRaises(RuntimeError):transform(b'wrong source')


if __name__=='__main__':unittest.main()

"""Offline scope/lifetime checks; NPU codegen and accuracy still required."""
import ast
from pathlib import Path
import unittest

from qresident_variant import transform, best_transform, OLD_QK, NEW_QK, Q_LOAD, Q_COPY

ROOT = Path(__file__).resolve().parents[3]


class QResidentTest(unittest.TestCase):
    def setUp(self):
        self.original = (ROOT/'operators/spec_v1/upstream_mha/core.py').read_bytes()
        self.result = transform(self.original).decode()

    def test_only_copy_movement(self):
        lines = self.result.splitlines(keepends=True)
        restored = []
        for index, line in enumerate(lines):
            if line.strip() == Q_COPY and index and lines[index-1].strip() == Q_LOAD:
                continue
            restored.append(line)
        restored = ''.join(restored).replace(NEW_QK, OLD_QK)
        self.assertEqual(restored.encode(), best_transform(self.original))

    def test_copy_is_once_per_q_tile_before_pipeline_in_both_branches(self):
        tree = ast.parse(self.result)
        q_copy_text = ast.unparse(ast.parse(Q_COPY))
        q_load_text = ast.unparse(ast.parse(Q_LOAD))
        qk = next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='qk')
        self.assertNotIn(q_copy_text, ast.unparse(qk))
        loops = [x for x in ast.walk(tree) if isinstance(x,ast.For)
                 and isinstance(x.target,ast.Name) and x.target.id=='local_mt']
        self.assertEqual(len(loops),2)
        for loop in loops:
            statements = [ast.unparse(x) for x in loop.body]
            q_load = statements.index(q_load_text)
            self.assertEqual(statements[q_load+1],q_copy_text)
            kv_index = next(i for i,x in enumerate(loop.body)
                            if isinstance(x,ast.For) and ast.unparse(x.target)=='kv')
            self.assertLess(q_load+1,kv_index)
            self.assertNotIn(q_copy_text,statements[kv_index])

    def test_source_pin(self):
        with self.assertRaises(RuntimeError):transform(b'wrong source')


if __name__=='__main__':unittest.main()

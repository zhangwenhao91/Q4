"""Check target range isolation and raw CSV inventory without an NPU."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from timeline_target import capture_one,RANGE_NAME
from timeline_inventory import inspect_csv


class TimelineTest(unittest.TestCase):
    def test_exactly_one_call_inside_range(self):
        events=[]
        def start(name,stream):
            events.append(('start',name,stream));return 7
        api=SimpleNamespace(range_start=start,range_end=lambda rid:events.append(('end',rid)))
        def fn():
            events.append('kernel');return 'output'
        output=capture_one(fn,api,lambda:events.append('sync'),'stream')
        self.assertEqual(output,'output')
        self.assertEqual(events,[('start',RANGE_NAME,'stream'),'kernel','sync',('end',7)])

    def test_failed_mstx_does_not_run_kernel(self):
        api=SimpleNamespace(range_start=lambda *args:0)
        with self.assertRaisesRegex(RuntimeError,'MSTX'):
            capture_one(lambda:self.fail('must not run'),api,lambda:None,None)

    def test_range_closes_on_kernel_error(self):
        events=[]
        api=SimpleNamespace(range_start=lambda *args:9,range_end=lambda rid:events.append(rid))
        def fail():
            raise RuntimeError('kernel failed')
        with self.assertRaisesRegex(RuntimeError,'kernel failed'):
            capture_one(fail,api,lambda:None,None)
        self.assertEqual(events,[9])

    def test_core_rows_preserved_without_aggregation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'core.csv'
            path.write_text('Core ID,Block ID,Duration(us),Pipe\n0,0,20,Cube\n0,28,21,Cube\n1,1,20,Cube\n',encoding='utf-8')
            report=inspect_csv(path)
            self.assertTrue(report['has_explicit_core_id_column'])
            self.assertEqual(len(report['preview_rows']),3)
            self.assertEqual(report['sampled_rows'][1],['0','28','21','Cube'])

    def test_logical_block_id_not_claimed_as_physical_core(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'block.csv'
            path.write_text('Block ID,Duration(us)\n0,20\n28,20\n',encoding='utf-8')
            self.assertFalse(inspect_csv(path)['has_explicit_core_id_column'])


if __name__=='__main__':unittest.main()

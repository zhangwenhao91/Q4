"""Check target range isolation and raw CSV inventory without an NPU."""
import tempfile
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from timeline_target import capture_one,RANGE_NAME
from timeline_inventory import inspect_csv
from timeline_verify import verify_collection
from native_mstx import NativeMstx


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

    def test_native_range_preserves_64_bit_id_and_null_stream(self):
        calls=[]
        expected=(1<<40)+7
        def start(message,stream):
            calls.append((message,stream));return expected
        def end(range_id):
            calls.append(range_id)
        api=NativeMstx(SimpleNamespace(mstxRangeStartA=start,mstxRangeEnd=end),{})
        rid=api.range_start('Q4_TIMELINE',None)
        self.assertEqual(rid,expected)
        api.range_end(rid)
        self.assertEqual(calls,[(b'Q4_TIMELINE',None),expected])
        self.assertIs(start.restype,ctypes.c_uint64)
        with self.assertRaises(ValueError):api.range_start('test','not_a_native_pointer')

    def collection_fixture(self,root,completed=True,log='[INFO] finished\n',data=True):
        folder=Path(root)/'tilelang_s2buf'
        (folder/'app').mkdir(parents=True)
        (folder/'app/metadata.json').write_text(json.dumps(dict(
            mode='capture',implementation='tilelang_s2buf',capture_succeeded=completed)),encoding='utf-8')
        (folder/'collect.log').write_text(log,encoding='utf-8')
        if data:
            path=folder/'op/OPPROF_example/core.csv'
            path.parent.mkdir(parents=True)
            path.write_text('Core ID,Duration(us)\n0,20\n',encoding='utf-8')
        return folder

    def test_zero_tool_exit_cannot_override_failed_capture(self):
        with tempfile.TemporaryDirectory() as root:
            folder=self.collection_fixture(root,completed=False)
            with self.assertRaisesRegex(RuntimeError,'did not complete'):verify_collection(folder)

    def test_child_error_in_log_rejects_even_completed_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            folder=self.collection_fixture(root,log='[WARN] Child process exited with return value 1\n')
            with self.assertRaisesRegex(RuntimeError,'failure'):verify_collection(folder)

    def test_empty_collection_does_not_count_source_as_result(self):
        with tempfile.TemporaryDirectory() as root:
            folder=self.collection_fixture(root,data=False)
            (folder/'app/selected_kernel.cpp').write_text('kernel source',encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'No nonempty'):verify_collection(folder)

    def test_nonfatal_warning_and_nonempty_core_data_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            folder=self.collection_fixture(root,log='[WARN] root permission used\n[INFO] finished\n')
            result=verify_collection(folder)
            self.assertTrue(result['target_completed'])
            self.assertEqual(len(result['result_files']),1)


if __name__=='__main__':unittest.main()

"""Offline source integrity and preservation checks; no torch/NPU required."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from prepare_mha import prepare,FILES,REV
import json


class OfflinePrepareTest(unittest.TestCase):
    def run_quiet(self,target,bundle=None):
        with contextlib.redirect_stdout(io.StringIO()): prepare(target,bundle)

    def test_prepare_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst=Path(tmp)/'dst'
            self.run_quiet(dst);self.run_quiet(dst)
            self.assertEqual(json.loads((dst/'source.json').read_text())['revision'],REV)
            self.assertTrue(all((dst/f).exists() for f in FILES))

    def test_preserve_modified_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst=Path(tmp)/'dst';self.run_quiet(dst)
            (dst/'core.py').write_bytes(b'user local modification')
            with self.assertRaisesRegex(RuntimeError,'Local file differs'): self.run_quiet(dst)
            self.assertEqual((dst/'core.py').read_bytes(),b'user local modification')

    def test_reject_corrupt_bundle_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp)/'bundle';dst=Path(tmp)/'dst'
            shutil.copytree(Path(__file__).with_name('upstream_mha'),bundle)
            (bundle/'core.py').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'): self.run_quiet(dst,bundle)
            self.assertFalse(dst.exists())


if __name__=='__main__': unittest.main()

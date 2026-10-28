import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from clean_build_flags import clean_cache


class CleanFlagsTest(unittest.TestCase):
    def test_clear_global_flags_preserve_compiler_target_and_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'CMakeCache.txt'
            original=('CMAKE_CXX_FLAGS:STRING=-march=nocona -mtune=haswell\n'
                      'CMAKE_C_FLAGS:STRING=-march=nocona\n'
                      'CMAKE_SHARED_LINKER_FLAGS:STRING=-L/conda/lib\n'
                      'CMAKE_CXX_COMPILER:FILEPATH=/conda/bin/g++\n'
                      'CANN_CXX11_ABI:STRING=1\n'
                      'CANN_3RD_LIB_PATH:PATH=/cache\n'
                      'CUSTOM_TARGET_FLAGS:STRING=-march=armv8-a\n')
            p.write_text(original,encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()): clean_cache(p)
            text=p.read_text(encoding='utf-8')
            self.assertNotIn('nocona',text);self.assertNotIn('haswell',text)
            for line in original.splitlines()[3:]: self.assertIn(line,text)
            backups=list(p.parent.glob('*.q4-flags-backup-*'))
            self.assertEqual(len(backups),1)
            self.assertEqual(backups[0].read_text(encoding='utf-8'),original)
            with contextlib.redirect_stdout(io.StringIO()): clean_cache(p)
            self.assertEqual(len(list(p.parent.glob('*.q4-flags-backup-*'))),1)

    def test_no_cache_does_not_create_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'CMakeCache.txt'
            with contextlib.redirect_stdout(io.StringIO()): clean_cache(p)
            self.assertFalse(p.exists())


if __name__=='__main__': unittest.main()

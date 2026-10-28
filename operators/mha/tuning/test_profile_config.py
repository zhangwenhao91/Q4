import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'operators'/'spec_v1'))
from profiling_config import extra_profile_options


class ProfileConfigTest(unittest.TestCase):
    def test_default_does_not_require_experimental_api(self):
        self.assertEqual(extra_profile_options(object()),{})

    def test_pipe_metrics_use_level1(self):
        api=SimpleNamespace(ProfilerLevel=SimpleNamespace(Level1='L1'),
                            AiCMetrics=SimpleNamespace(PipeUtilization='PIPE'),
                            ExportType=SimpleNamespace(Text='TEXT'),
                            _ExperimentalConfig=lambda **kw:kw)
        result=extra_profile_options(api,True)
        self.assertEqual(result,{'experimental_config':dict(profiler_level='L1',aic_metrics='PIPE',export_type='TEXT')})

    def test_no_silent_downgrade(self):
        with self.assertRaisesRegex(RuntimeError,'lacks'):
            extra_profile_options(object(),True)


if __name__=='__main__': unittest.main()

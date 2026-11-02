import json
from pathlib import Path
import tempfile
import unittest

from analyze import analyze,matrix_flops,peak_for,throughput


MHA=dict(Q=[4096,128],K=[4096,128],V=[4096,128],batch=1,heads=1,
         input_dtype='bfloat16',output_dtype='bfloat16',causal=False,dropout=0)
DEVICE=dict(name='test-device',cube_core_num=28,vector_core_num=56)


class MFUTest(unittest.TestCase):
    def test_useful_flops_and_precision_mode(self):
        self.assertEqual(matrix_flops('mha_stats_split',MHA),(8589934592,'bf16'))
        contract=dict(N=512,hidden=5120,hc_mult=4,flattened=20480,channels=24,hc_shifted=True,repeat=20)
        self.assertEqual(matrix_flops('mhc_shifted_split',contract),(503316480,'hf32'))
        with self.assertRaises(ValueError):matrix_flops('mha_stats_split',{**MHA,'causal':True})

    def test_throughput_units_and_bad_latency(self):
        self.assertEqual(throughput(1_000_000_000,10),100)
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):throughput(10,value)

    def test_peak_missing_is_not_zero(self):
        self.assertIsNone(peak_for({},'hf32',DEVICE))

    def test_peak_requires_device_mode_scope_and_source(self):
        entry=dict(tflops=200,source='test fixture only',scope='visible_device_dense',
                   name='test-device',cube_core_num=28)
        self.assertEqual(peak_for({'bf16':entry},'bf16',DEVICE)['tflops'],200)
        for update in ({'source':''},{'scope':'sparse'},{'cube_core_num':32},{'name':'other'},
                       {'tflops':float('nan')},{'tflops':True}):
            with self.assertRaises(ValueError):peak_for({'bf16':{**entry,**update}},'bf16',DEVICE)

    def fixture(self,folder):
        plan={'cases':[]}
        for rep,latency in enumerate((10.,20.,30.),1):
            name=f'case{rep}';path=folder/name;path.mkdir()
            case=dict(directory=name,suite='mha_stats_split',tokens=4096,labels=['tilelang_s2buf'],
                      rounds=2,replica=rep)
            plan['cases'].append(case)
            meta=dict(arguments=dict(suite=case['suite'],tokens=4096,implementations=case['labels'],
                                     rounds=2,warmup=100,iters=100),hf32=True,
                      implementation_labels=case['labels'],contract=MHA,device_properties=DEVICE,q4_revision='test')
            for file,value in [('metadata.json',meta),('correctness.json',[dict(passed=True)]),
                               ('summary.json',dict(tilelang_s2buf=dict(event_us=latency,wall_us=latency+1)))]:
                (path/file).write_text(json.dumps(value),encoding='utf-8')
            (path/'timings.csv').write_text('round,implementation,event_us,wall_us\n'
                +''.join(f'{r},tilelang_s2buf,{latency},{latency+1}\n' for r in (0,1)),encoding='utf-8')
        (folder/'plan.json').write_text(json.dumps(plan),encoding='utf-8')
        return plan

    def test_aggregate_median_latency_not_best_replica(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixture(folder)
            result=analyze(folder)['results'][0]
            self.assertEqual(result['event_us'],20.)
            self.assertEqual(result['effective_tflops'],8589934592/20/1e6)
            self.assertIsNone(result['matrix_mfu_percent'])

    def test_incomplete_or_failed_run_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixture(folder)
            (folder/'case1/correctness.json').write_text('[{"passed": false}]')
            with self.assertRaisesRegex(ValueError,'precision'):analyze(folder)
            (folder/'case1/correctness.json').write_text('[{"passed": true}]')
            (folder/'case2/timings.csv').write_text('round,implementation,event_us,wall_us\n0,tilelang_s2buf,20,21\n')
            with self.assertRaisesRegex(ValueError,'Incomplete'):analyze(folder)

    def test_missing_replica_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);plan=self.fixture(folder)
            plan['cases'].pop();(folder/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'three replicas'):analyze(folder)


if __name__=='__main__':unittest.main()

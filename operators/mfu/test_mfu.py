import json
import shutil
from pathlib import Path
import tempfile
import unittest

from analyze import analyze,matrix_flops,peak_for,throughput,compare_sizes,attention_comparisons
from run import make_groups
from export import export_capture


MHA=dict(Q=[4096,128],K=[4096,128],V=[4096,128],batch=1,heads=1,
         input_dtype='bfloat16',output_dtype='bfloat16',causal=False,dropout=0)
DEVICE=dict(name='test-device',cube_core_num=28,vector_core_num=56)


class MFUTest(unittest.TestCase):
    def test_all_retained_variants_plan_is_fixed_s3584_and_balanced(self):
        groups=make_groups('mha','seven-eighths',all_mha_variants=True)
        self.assertEqual(len(groups),1)
        group=groups[0]
        self.assertEqual((group['suite'],group['tokens'],group['rounds']),('mha_scenarios',3584,16))
        self.assertEqual(len(group['labels']),8)
        self.assertEqual(group['labels'][-1],'ascendc')
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'spec_v1'))
        from bench import implementation_order
        from mha_scenarios import configurations
        self.assertEqual(set(group['labels'])-{'tilelang_q128','ascendc'},
                         {label for label,_ in configurations()})
        from collections import Counter
        positions=Counter((label,pos) for rnd in range(16)
            for pos,label in enumerate(implementation_order(group['labels'],rnd)))
        self.assertEqual(set(positions.values()),{2})

    def test_new_scenarios_suite_uses_scaled_flops(self):
        contract={**MHA,**{key:[3584,128] for key in ('Q','K','V')}}
        self.assertEqual(matrix_flops('mha_scenarios',contract),(6576668672,'bf16'))
        with self.assertRaisesRegex(ValueError,'requires S3584'):matrix_flops('mha_scenarios',MHA)

    def test_attention_ratio_uses_matching_shape_and_replica(self):
        base=dict(suite='mha_stats_split',tokens=3584,useful_matrix_flops=6576668672,
                  matrix_mfu_percent=None)
        tl={**base,'implementation':'tilelang_kv2','event_us':24.,'wall_us':25.,
            'replicas':[dict(replica=r,event_us=t) for r,t in ((1,24),(2,30),(3,24))]}
        ac={**base,'implementation':'ascendc','event_us':27.,'wall_us':28.,
            'replicas':[dict(replica=r,event_us=27.) for r in (3,1,2)]}
        # Do not pair with an AC at another sequence length.
        self.assertEqual(attention_comparisons([tl,{**ac,'tokens':4096}]),[])
        result=attention_comparisons([tl,ac])[0]
        self.assertEqual(result['ac_over_tl'],27/24)
        self.assertEqual(result['tl_wins'],2)
        self.assertEqual(result['paired_replicas'][1]['ac_over_tl'],.9)

    def test_scaled_plan_and_useful_flops(self):
        groups=make_groups('all','both')
        self.assertEqual(len(groups),12)
        scaled=[g for g in groups if g['size_mode']=='seven-eighths']
        self.assertEqual([g['tokens'] for g in scaled],[3584,56,112,224,448,7168])
        self.assertEqual(len({(g['suite'],g['tokens']) for g in groups}),12)
        contract={**MHA,**{key:[3584,128] for key in ('Q','K','V')}}
        flops,mode=matrix_flops('mha_stats_split',contract)
        self.assertEqual(flops,6576668672)
        self.assertEqual(flops/8589934592,49/64)
        self.assertEqual(mode,'bf16')

    def test_shape_efficiency_accounts_for_less_work(self):
        full=dict(suite='mha_stats_split',tokens=4096,implementation='tilelang_s2buf',
                  useful_matrix_flops=8589934592,event_us=50,effective_tflops=8589934592/50/1e6)
        scaled={**full,'tokens':3584,'useful_matrix_flops':6576668672,'event_us':40,
                'effective_tflops':6576668672/40/1e6}
        result=compare_sizes([full,scaled])[0]
        self.assertEqual(result['latency_ratio'],.8)
        self.assertLess(result['throughput_ratio'],1.)
        self.assertEqual(result['useful_flops_ratio'],49/64)

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

    def test_export_preserves_raw_timings_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';source.mkdir()
            self.fixture(source)
            target=Path(temp)/'exported'
            manifest=export_capture(source,target,{})
            self.assertEqual(len(manifest['copied_files']),13)
            self.assertEqual((target/'case1/timings.csv').read_bytes(),(source/'case1/timings.csv').read_bytes())
            self.assertEqual(analyze(target)['results'],analyze(source)['results'])
            with self.assertRaisesRegex(ValueError,'new output'):export_capture(source,target,{})

    def test_mha_only_export_does_not_require_or_copy_mhc_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';source.mkdir()
            plan=self.fixture(source)
            plan['cases'].append(dict(suite='mhc_shifted_split',directory='ignored-mhc'))
            (source/'plan.json').write_text(json.dumps(plan))
            target=Path(temp)/'mha-export'
            export_capture(source,target,{},operator='mha')
            self.assertEqual(len(json.loads((target/'plan.json').read_text())['cases']),3)
            self.assertFalse((target/'ignored-mhc').exists())
            self.assertEqual(len(analyze(target)['results']),1)

    def test_full_and_scaled_capture_analyzed_with_actual_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);plan=self.fixture(folder)
            for full in list(plan['cases']):
                scaled={**full,'directory':full['directory']+'-scaled','tokens':3584}
                plan['cases'].append(scaled)
                path=folder/scaled['directory']
                shutil.copytree(folder/full['directory'],path)
                meta=json.loads((path/'metadata.json').read_text())
                meta['arguments'].update(tokens=3584,mha_seqlen=3584)
                meta['contract']={**MHA,**{key:[3584,128] for key in ('Q','K','V')}}
                (path/'metadata.json').write_text(json.dumps(meta))
            (folder/'plan.json').write_text(json.dumps(plan))
            result=analyze(folder)
            self.assertEqual(len(result['results']),2)
            pair=result['size_comparisons'][0]
            self.assertEqual(pair['latency_ratio'],1.)
            self.assertAlmostEqual(pair['throughput_ratio'],49/64)
            bad=folder/'case1-scaled/metadata.json'
            meta=json.loads(bad.read_text());meta['arguments']['mha_seqlen']=4096
            bad.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'sequence argument mismatch'):analyze(folder)


if __name__=='__main__':unittest.main()

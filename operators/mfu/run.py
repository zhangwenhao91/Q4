"""Measure existing attention and bound three-kernel mHC paths, then analyze offline."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from analyze import analyze


def make_groups(operator,scale):
    groups=[]
    for size_mode in (('full','seven-eighths') if scale=='both' else (scale,)):
        def size(n):return n if size_mode=='full' else n*7//8
        if operator in ('all','mha'):
            groups.append(dict(suite='mha_stats_split',tokens=size(4096),base_tokens=4096,
                labels=['tilelang_q128','tilelang_kv2','tilelang_s2buf','ascendc'],rounds=8,
                package='q4fa-opp',vendor='q4fa_transformer',size_mode=size_mode))
        if operator in ('all','mhc'):
            for n in (64,128,256,512,8192):
                groups.append(dict(suite='mhc_shifted_split',tokens=size(n),base_tokens=n,
                    labels=['tilelang_coeff_bound'],rounds=9,package='q4mhcmatch-opp',
                    vendor='q4mhcmatch_transformer',size_mode=size_mode))
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',choices=('all','mha','mhc'),default='all')
    p.add_argument('--output',type=Path)
    p.add_argument('--scale',choices=('full','seven-eighths','both'),default='full')
    p.add_argument('--peak-json',type=Path,default=Path(__file__).with_name('peaks')/'ascend950pr_28aic.json')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    output=(a.output or root/'results/mfu'/f'{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}').resolve()
    prefix=os.environ.get('CONDA_PREFIX','')
    if Path(prefix).name!='zwh':p.error('Activate the existing zwh environment')
    if not os.environ.get('ASCEND_HOME_PATH'):p.error('Source CANN set_env.sh first')
    for compiler in ('gcc','g++'):
        if not (Path(prefix)/'bin'/f'x86_64-conda-linux-gnu-{compiler}').is_file():
            p.error('Missing Conda GCC13 toolchain')
    groups=make_groups(a.operator,a.scale)
    peaks=json.loads(a.peak_json.read_text(encoding='utf-8'))
    for group in groups:
        package,vendor=group['package'],group['vendor']
        if not (root/'third_party'/package/'vendors'/vendor/'op_api/lib/libcust_opapi.so').is_file():
            p.error(f'Missing {package}/{vendor}; restore the corresponding existing vendor first')
    output.mkdir(parents=True,exist_ok=False)
    print('MFU OUTPUT:',output,flush=True)
    plan={'scope':'Existing full API calls, matrix FLOP numerator, no new numeric kernels',
          'scale':a.scale,
          'q4_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'cases':[]}
    for rep in (1,2,3):
        # Reverse the shape order on alternate processes to reduce fixed shape-order bias.
        for group in (groups if rep%2 else list(reversed(groups))):
            suite,n,labels,rounds,package,vendor=(group[k] for k in
                ('suite','tokens','labels','rounds','package','vendor'))
            name=f'{suite}-N{n}-rep{rep}'
            env=os.environ.copy()
            env.update(ASCEND_RT_VISIBLE_DEVICES=env.get('ASCEND_RT_VISIBLE_DEVICES') or '4',
                ASCEND_NPU_ARCH='dav-3510',CC=str(Path(prefix)/'bin/x86_64-conda-linux-gnu-gcc'),
                CXX=str(Path(prefix)/'bin/x86_64-conda-linux-gnu-g++'),
                TORCH_EXTENSIONS_DIR=str(root/'third_party/torch-extensions-gcc13'))
            opp=root/'third_party'/package/'vendors'/vendor
            env['ASCEND_CUSTOM_OPP_PATH']=str(opp)
            # Each vendor loads in its own child process. Remove other Q4-local vendor paths.
            inherited=[x for x in env.get('LD_LIBRARY_PATH','').split(os.pathsep)
                       if x and not (x.startswith(str(root/'third_party')+os.sep)
                                     and '/vendors/' in x.replace('\\','/'))]
            env['LD_LIBRARY_PATH']=os.pathsep.join([str(opp/'op_api/lib'),*inherited])
            command=[sys.executable,str(root/'operators/spec_v1/bench.py'),'--suite',suite,
                '--tokens',str(n),'--warmup','100','--iters','100','--rounds',str(rounds),
                '--output',str(output/name),'--implementations',*labels]
            if suite=='mha_stats_split':command.extend(['--mha-seqlen',str(n)])
            case=dict(directory=name,suite=suite,tokens=n,labels=labels,rounds=rounds,replica=rep,
                      size_mode=group['size_mode'],base_tokens=group['base_tokens'])
            plan['cases'].append(case)
            (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
            print('RUN:',name,','.join(labels),flush=True)
            with (output/(name+'.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                print((output/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-14000:],flush=True)
                raise RuntimeError(f'Case failed ({result.returncode}); preserve logs at {output}')
            print('DONE:',name,flush=True)
    # Validate all expected files before the CLI writes reports.
    analyze(output,peaks)
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze.py')),
                    '--input',str(output),'--peak-json',str(a.peak_json.resolve())],check=True)
    print('Complete. Update error.txt with this output; retain the full MFU OUTPUT directory.',flush=True)


if __name__=='__main__':main()

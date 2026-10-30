"""Read-only inventory after recloning Q4. Does not import torch or execute NPU work."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[3]
    report={'root':str(root),'python':sys.executable,'conda_env':os.environ.get('CONDA_DEFAULT_ENV'),
            'paths':{},'packages':{},'tools':{},'environment':{}}
    paths=('results/mha-tuning/timeline-pipeline-20261030-091704-3140527',
           'reports/mha_timeline/captures/20261030_pipeline',
           'third_party/tilelang-mha-spec-v1/source.json',
           'third_party/ops-transformer-q4fa',
           'third_party/q4fa-opp/vendors/q4fa_transformer/op_api/lib',
           'third_party/torch-extensions-gcc13')
    for relative in paths:
        path=root/relative
        report['paths'][relative]={'exists':path.exists()}
        if path.is_dir() and path.name=='lib':
            report['paths'][relative]['shared_libraries']=[p.name for p in path.glob('*.so')]
    for name in ('torch','torch-npu','tilelang','cann-ops-transformer'):
        try:
            dist=importlib.metadata.distribution(name)
            entry={'version':dist.version,'location':str(dist.locate_file(''))}
            raw=dist.read_text('direct_url.json')
            if raw:
                direct=json.loads(raw)
                # Only report local paths; never print remote installation URLs or credentials.
                url=direct.get('url','')
                if url.startswith('file:'):
                    entry['local_install_source']=url
                    entry['editable']=direct.get('dir_info',{}).get('editable',False)
            report['packages'][name]=entry
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name]={'installed':False}
    for name in ('python','msprof','gcc','g++','cmake','ninja'):
        report['tools'][name]=shutil.which(name)
    for name in ('ASCEND_HOME_PATH','ASCEND_CUSTOM_OPP_PATH','TORCH_EXTENSIONS_DIR',
                 'ASCEND_RT_VISIBLE_DEVICES','ASCEND_NPU_ARCH','CC','CXX'):
        report['environment'][name]=os.environ.get(name)
    source=root/'third_party/ops-transformer-q4fa'
    if source.is_dir():
        result=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True)
        report['native_source_revision']=result.stdout.strip() if result.returncode==0 else 'unavailable'
    report['possible_capture_directories']=[str(p.relative_to(root)) for p in
        sorted((root/'results/mha-tuning').glob('timeline-pipeline-*')) if p.is_dir()]
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print('Read-only inventory complete. No installations, compilation or device execution.')


if __name__=='__main__':main()

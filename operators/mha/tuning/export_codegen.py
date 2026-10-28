"""Export three exact Q128 compiled device sources; no timing or kernel execution."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import tilelang
    from nostore_variant import transform as nostore,ORIGINAL_SHA256
    from rowfused_variant import transform as rowfused
    root=Path(__file__).resolve().parents[3]
    original=(root/'operators/spec_v1/upstream_mha/core.py').read_bytes()
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_SHA256:
        raise RuntimeError('Pinned core.py hash mismatch')
    manifest=dict(q4_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                  tilelang_version=getattr(tilelang,'__version__','unknown'),
                  scope='Compiled device C++ sources; not assembly or a register/spill report; no kernel run',kernels={})
    for name,data in [('q128',original),('nostore',nostore(original)),('rowfused',rowfused(original))]:
        pyfile=(a.output/f'core_{name}.py').resolve();pyfile.write_bytes(data)
        spec=importlib.util.spec_from_file_location('q4_export_'+name,pyfile)
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        program=mod.flash_attention_fwd(q_len=4096,kv_len=4096,head_dim=128,
            out_dtype='bfloat16',softmax_scale=1/math.sqrt(128),
            tiling=mod.FwdTiling(block_q=128,block_kv=128,num_stages=3))
        kernel=tilelang.compile(program,out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
        cpp=kernel.get_kernel_source()
        if not isinstance(cpp,str) or not cpp.strip():raise RuntimeError('Empty compiled source: '+name)
        (a.output/f'{name}.cpp').write_text(cpp,encoding='utf-8')
        manifest['kernels'][name]=dict(input_sha256=hashlib.sha256(data).hexdigest(),
            cpp_sha256=hashlib.sha256(cpp.encode()).hexdigest(),source_file=f'{name}.cpp')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with (a.output/'codegen.txt').open('w',encoding='utf-8') as f:
        f.write(json.dumps(manifest,indent=2)+'\n')
        for name in manifest['kernels']:
            f.write('\n===== '+name+'.cpp =====\n')
            f.write((a.output/f'{name}.cpp').read_text(encoding='utf-8')+'\n')
    print('Export complete:',(a.output/'codegen.txt').resolve())

if __name__=='__main__':main()

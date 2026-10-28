"""Native BF16 attention comparison. Both kernels write BF16 directly."""
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def make_case(a,extra_builder=None):
    import torch
    import tilelang
    import cann_ops_transformer
    from prepare_mha import REV,FILES
    native_source=ROOT/'third_party'/'ops-transformer-q4fa'
    native_rev=subprocess.check_output(['git','-C',str(native_source),'rev-parse','HEAD'],text=True).strip()
    if native_rev!='ba5162411c919e15fbabb28298e8b420c11ad905':
        raise RuntimeError('Native attention source revision mismatch')
    opp_value=os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(os.pathsep)[0]
    opp=Path(opp_value).resolve()
    if (ROOT/'third_party'/'q4fa-opp').resolve() not in opp.parents:
        raise RuntimeError('Select the new q4fa vendor in ASCEND_CUSTOM_OPP_PATH before testing')
    libraries=list((opp/'op_api'/'lib').glob('*.so'))
    if not libraries: raise RuntimeError('Missing q4fa op_api libraries; build/install native attention first')
    library_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries}
    source=ROOT/'third_party'/'tilelang-mha-spec-v1'
    manifest=json.loads((source/'source.json').read_text(encoding='utf-8'))
    if manifest['revision']!=REV or not set(FILES).issubset(manifest['sha256']):
        raise RuntimeError('Run prepare_mha.py: pinned source manifest is missing or mismatched')
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Modified pinned source: {name}')
    sys.path.insert(0,str(source))
    from core import flash_attention_fwd
    # The public core builder supports an in-kernel BF16 store: no torch cast.
    kernel=tilelang.compile(flash_attention_fwd(q_len=4096,kv_len=4096,head_dim=128,
                           out_dtype='bfloat16',softmax_scale=1/math.sqrt(128)),out_idx=-1,
                           pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
    cpu=[torch.randn(4096,128,dtype=torch.bfloat16) for _ in range(3)]
    q,k,v=[z.npu() for z in cpu]
    # One batch and one head: the views do not copy data, transpose, or flatten heads into tokens.
    aq,ak,av=[z.view(1,4096,1,128) for z in (q,k,v)]
    qf,kf,vf=[z.float() for z in cpu]
    ref=torch.empty(4096,128,dtype=torch.float32)
    for start in range(0,4096,128):
        ref[start:start+128]=((qf[start:start+128]@kf.T)/math.sqrt(128)).softmax(-1)@vf
    ref=ref.bfloat16()
    op=cann_ops_transformer.ops
    common=dict(mask_mode=0,win_left=-1,win_right=-1,
                layout_q='BSND',layout_kv='BSND',layout_out='BSND')
    def prepare_metadata():
        return op.flash_attn_metadata(1,1,128,batch_size=1,max_seqlen_q=4096,max_seqlen_kv=4096,
                                      is_grad_enabled=False,**common)
    # Cache setup is explicitly measured outside main calls, including first metadata JIT.
    torch.npu.synchronize();begin=time.perf_counter()
    metadata=prepare_metadata();torch.npu.synchronize()
    setup_ms=(time.perf_counter()-begin)*1000
    def tl(): return (kernel(q,k,v),)
    ac=native_runner(op,aq,ak,av,prepare_metadata,metadata,a.ac_metadata)
    functions={'tilelang':tl,'ascendc':ac}
    tuning_info=None
    if extra_builder is not None:
        extra,tuning_info=extra_builder(source,q,k,v)
        functions={'tilelang_q128':tl,**extra,'ascendc':ac}
    return functions,(ref,),['O'],{
        'contract':dict(Q=[4096,128],K=[4096,128],V=[4096,128],batch=1,heads=1,
                        input_dtype='bfloat16',output_dtype='bfloat16',scale=1/math.sqrt(128),
                        causal=False,dropout=0,attention_bias=False,return_lse=False),
        'source':manifest,'comparison_available':True,'tuning':tuning_info,
        'native_source_revision':native_rev,'native_vendor':str(opp),'native_library_sha256':library_hashes,
        'native_kernels':dict(tilelang='core.py::flash_attention_fwd(out_dtype=bfloat16)',
                              ascendc='ops-transformer FlashAttn + FlashAttnMetadata'),
        'reference':'blocked CPU FP32 attention rounded once to BF16',
        'dtype_change':'user-authorized BF16 output variant; separate from spec-v1 FP32 output baseline',
        'ascendc_layout':'BSND [1,4096,1,128]; same contiguous storage as TileLang [4096,128]',
        'metadata_mode':a.ac_metadata,'metadata_setup_wall_ms_including_first_jit':setup_ms,
        'metadata_numel':metadata.numel(),
        'timing_boundary':('reused fixed-shape native metadata, setup excluded; output/temp allocation included'
                           if a.ac_metadata=='cached' else
                           'native metadata regenerated on each AscendC call and included in timing'),
        'torch_role':'allocation, views, wrappers and timing only; no numerical attention fallback or output cast'}


def native_runner(op,q,k,v,prepare_metadata,metadata,mode):
    """Create the measured native call; cached mode must never regenerate metadata."""
    if mode not in ('cached','per_call'):
        raise ValueError('Unknown metadata timing boundary')
    def run():
        current=metadata if mode=='cached' else prepare_metadata()
        out,_=op.flash_attn(q,k,v,metadata=current,softmax_scale=1/math.sqrt(128),
                            mask_mode=0,win_left=-1,win_right=-1,
                            max_seqlen_q=4096,max_seqlen_kv=4096,
                            layout_q='BSND',layout_kv='BSND',layout_out='BSND',
                            return_softmax_lse=False)
        return (out.view(4096,128),)
    return run

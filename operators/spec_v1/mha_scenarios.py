"""All retained useful MHA stages at the fixed seven-eighths sequence length."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys


def configurations():
    # Variants with a measured regression were retired; buffer negative controls
    # remain in their original diagnostics rather than the useful-version sweep.
    return (('tilelang_s2',2),('tilelang_nostore',3),('tilelang_nostore_s2',2),
            ('tilelang_finalize',2),('tilelang_kv2',2),('tilelang_s2buf',2))


def build_candidate(source,q,k,v):
    import tilelang
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'operators/mha/tuning'))
    from nostore_variant import transform as nostore,ORIGINAL_SHA256
    from finalize_variant import transform as finalize
    from buffers_variant import transform as buffers
    from stats_split_variant import transform as stats
    if any(tuple(t.shape)!=(3584,128) for t in (q,k,v)):
        raise ValueError('All-scenarios sweep is fixed at Q/K/V [3584,128]')
    original=(Path(source)/'core.py').read_bytes()
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_SHA256:
        raise ValueError('Pinned original source mismatch')
    transforms=dict(tilelang_s2=lambda x:x,tilelang_nostore=nostore,tilelang_nostore_s2=nostore,
                    tilelang_finalize=finalize,tilelang_kv2=lambda x:buffers(x,'tilelang_kv2'),
                    tilelang_s2buf=lambda x:stats(x,'tilelang_s2buf'))
    functions={};details={}
    for label,stages in configurations():
        data=transforms[label](original)
        path=root/'third_party/mha-all-scenarios/s3584'/f'{label}.py'
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=data:
            raise ValueError('Preserve modified generated source first: '+str(path))
        path.write_bytes(data)
        spec=importlib.util.spec_from_file_location('q4_all_'+label,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module;spec.loader.exec_module(module)
        program=module.flash_attention_fwd(q_len=3584,kv_len=3584,head_dim=128,
            out_dtype='bfloat16',softmax_scale=1/math.sqrt(128),manual_schedule=False,
            tiling=module.FwdTiling(block_q=128,block_kv=128,num_stages=stages))
        kernel=tilelang.compile(program,out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
        cpp=kernel.get_kernel_source()
        if not isinstance(cpp,str) or not cpp.strip():raise ValueError('Empty generated source')
        path.with_suffix('.cpp').write_text(cpp,encoding='utf-8')
        functions[label]=lambda kernel=kernel:(kernel(q,k,v),)
        details[label]=dict(stages=stages,python_sha256=hashlib.sha256(data).hexdigest(),
                           cpp_sha256=hashlib.sha256(cpp.encode()).hexdigest(),source=str(path))
    return functions,dict(seq_len=3584,q_tiles=28,kv_blocks=28,sources=details,
        scope='Retained useful transforms and ablations, identical math contract; not a batch/head/dtype sweep.')


def make_case(a):
    if a.mha_seqlen!=3584 or a.ac_metadata!='cached':
        raise ValueError('All-scenarios sweep requires S3584 and cached metadata')
    from mha_bf16 import make_case as baseline
    functions,refs,names,meta=baseline(a,extra_builder=build_candidate)
    meta['sweep_scope']='All retained useful MHA versions/ablations at B1/H1/S3584/D128; cached native metadata'
    meta['native_kernels']={
        'tilelang_q128':'pinned original core.py, stages3',
        **{label:f'retained transform {label}, stages{stages}' for label,stages in configurations()},
        'ascendc':'native FlashAttn, cached FlashAttnMetadata'}
    return functions,refs,names,meta

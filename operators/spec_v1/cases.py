"""Exact declared tensor contracts; no hidden output-dtype or epsilon substitution."""
import importlib
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'operators'/'mhc'/'extended'))


def load_bridge():
    from torch.utils.cpp_extension import load
    m=importlib.import_module('cann_ops_transformer.ops.mhc.mhc_post.mhc_post')
    b=m.mhc_post_op_builder
    return load(name='q4_spec_v1_aux',sources=[str(Path(__file__).with_name('aclnn_bridge.cpp'))],
        extra_include_paths=b.get_absolute_paths(b.include_paths()),extra_cflags=b.cxx_args(),
        extra_ldflags=b.extra_ldflags(),verbose=True)


def shifted_reference(x,prev,phi,alpha,bias):
    """CPU torch or array-like module supplied by caller; norm/pre eps are distinct."""
    flat=x.float().flatten(1)
    mix=(flat@phi.T)*(flat.square().mean(-1,keepdim=True)+1e-20).rsqrt()
    new_pre=(mix[:,:4]*alpha[0]+bias[:4]).sigmoid()+1e-6
    post=2*(mix[:,4:8]*alpha[1]+bias[4:8]).sigmoid()
    logits=(mix[:,8:]*alpha[2]+bias[8:]).view(-1,4,4)
    from bench_forward import sinkhorn_ref
    hi=(x.float()*prev).sum(1).bfloat16()
    return hi,new_pre.unsqueeze(-1),post.unsqueeze(-1),sinkhorn_ref(logits,20,1e-6)


def mha_case(a):
    import torch
    import tilelang
    source=ROOT/'third_party'/'tilelang-mha-spec-v1'
    if not (source/'source.json').exists():
        raise RuntimeError('Run python operators/spec_v1/prepare_mha.py first')
    # Verify downloaded kernels still match their source manifest.
    import json,hashlib
    manifest=json.loads((source/'source.json').read_text())
    from prepare_mha import REV
    if manifest['revision']!=REV: raise RuntimeError('MHA revision mismatch')
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Modified source: {name}')
    sys.path.insert(0,str(source))
    from example_mha import mha
    kernel=tilelang.compile(mha(128,4096),out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
    cpu=[torch.randn(4096,128,dtype=torch.bfloat16) for _ in range(3)]
    q,k,v=[z.npu() for z in cpu]
    # Blocked CPU FP32 golden; avoids retaining a full 4096x4096 score matrix.
    qf,kf,vf=[z.float() for z in cpu]
    ref=torch.empty(4096,128,dtype=torch.float32)
    for i in range(0,4096,128):
        ref[i:i+128]=((qf[i:i+128]@kf.T)/math.sqrt(128)).softmax(-1)@vf
    return {'tilelang':lambda:(kernel(q,k,v),)},(ref,),['O'],{
        'contract':dict(Q=[4096,128],K=[4096,128],V=[4096,128],input_dtype='bfloat16',output_dtype='float32',
                        batch=1,heads=1,scale=1/math.sqrt(128),causal=False,dropout=0),
        'source':manifest,'ascendc_status':'blocked: stock FlashAttn BF16 input has BF16 output; no dtype substitution',
        'comparison_available':False}


def post_case(a):
    import torch
    from tile_kernels.mhc import mhc_post_fwd
    from cann_ops_transformer.ops import mhc_post
    x0=torch.randn(1,a.tokens,5120,dtype=torch.bfloat16)
    r0=torch.randn(1,a.tokens,4,5120,dtype=torch.bfloat16)
    hp0=2*torch.rand(1,a.tokens,4,1)
    hr0=torch.rand(1,a.tokens,4,4)
    from bench_forward import sinkhorn_ref
    hr0=sinkhorn_ref(hr0,20,1e-6)
    ref=(torch.einsum('btij,btid->btjd',hr0,r0.float())+hp0*x0.float().unsqueeze(-2)).bfloat16().squeeze(0)
    x,r,hp,hr=[z.npu() for z in [x0,r0,hp0,hr0]]
    return {'tilelang':lambda:(mhc_post_fwd(x,r,hp,hr).squeeze(0),),
            'ascendc':lambda:(mhc_post(r,hr,x,hp.squeeze(-1)).squeeze(0),)},(ref,),['new_residual'],{
                'contract':dict(N=a.tokens,hidden=5120,hc_mult=4,input_features='BF16',coefficients='FP32',output='BF16'),
                'comparison_available':True}


def shifted_case(a, native_shifted=False, fusion=False, bind=False):
    import torch
    from fused_projection import HCProjection
    from tile_kernels.mhc import (mhc_reduce_partials_and_rmsnorm_fwd,mhc_pre_split_mixes_fwd,
        mhc_pre_apply_mix_fwd,mhc_sinkhorn_fwd)
    t=a.tokens;d=5120
    x0=torch.randn(t,4,d,dtype=torch.bfloat16)
    prev0=torch.rand(t,4,1)+1e-6
    phi0=torch.empty(24,4*d).uniform_(-1,1)/(4*d)**.5
    alpha0=torch.tensor([.1,.1,.1]);bias0=torch.randn(24)*.1
    if native_shifted:
        if a.mhc_pattern == 'zero':
            x0.zero_()
        elif a.mhc_pattern == 'prev_zero':
            prev0.zero_()
        elif a.mhc_pattern == 'prev_basis':
            prev0.zero_();prev0[:,2,0]=1.
        if a.mhc_weights:
            weights=torch.load(a.mhc_weights,map_location='cpu',weights_only=True)
            phi0,alpha0,bias0=[weights[k] for k in ('fn','scale','base')]
            for value,shape in zip((phi0,alpha0,bias0),((24,20480),(3,),(24,))):
                if value.dtype!=torch.float32 or tuple(value.shape)!=shape or not torch.isfinite(value).all():
                    raise ValueError('Weights must be finite FP32 fn[24,20480], scale[3], base[24]')
            phi0,alpha0,bias0=[v.contiguous() for v in (phi0,alpha0,bias0)]
    refs=shifted_reference(x0,prev0,phi0,alpha0,bias0)
    x,prev,phi,alpha,bias=[z.npu() for z in [x0,prev0,phi0,alpha0,bias0]]
    if native_shifted:
        from matched_case import load_matched_bridge, DynamicHCProjection, native_metadata
        projector=DynamicHCProjection(x,phi)
        bridge=load_matched_bridge()
        from fused_projection import check_projection
        projection_checks=check_projection(projector,x0,phi0,t,d)
        if not all(item['passed'] for item in projection_checks):
            raise AssertionError('Raw projection check failed: '+str(projection_checks))
    else:
        projector=HCProjection(x,phi);bridge=load_bridge()
    def tl():
        hi=torch.empty(t,d,device=x.device,dtype=x.dtype)
        mhc_pre_apply_mix_fwd(x,prev.view(t,4),hi)
        raw,sq=projector()
        raw_out=torch.empty(t,1,24,device=x.device);sq_out=torch.empty(t,1,device=x.device)
        mix=torch.empty(t,24,device=x.device)
        mhc_reduce_partials_and_rmsnorm_fwd(raw.view(1,t,1,24),sq.view(1,t,1),raw_out,sq_out,mix,
                                         rms_group_size=4*d,rms_eps=1e-20,n_splits=1)
        pre=torch.empty(t,4,device=x.device);post=torch.empty_like(pre)
        logits=torch.empty(t,16,device=x.device)
        mhc_pre_split_mixes_fwd(mix,alpha,bias,pre,post,logits,2.,1e-6)
        comb=torch.empty(t,4,4,device=x.device)
        mhc_sinkhorn_fwd(logits.view(t,4,4),comb,20,1e-6)
        return hi,pre.unsqueeze(-1),post.unsqueeze(-1),comb
    def ac():
        if native_shifted:
            return bridge.shifted(x,prev,phi,alpha,bias)
        # All casts, elementwise/reduction kernels and redundant native work count.
        hi=(x.float()*prev).sum(1).bfloat16()
        pre,post,logits=bridge.pre_coefficients(x,phi,alpha,bias,1e-20,1e-6)
        comb=bridge.sinkhorn(logits,1e-6,20)
        return hi,pre.unsqueeze(-1),post.unsqueeze(-1),comb
    functions={'deepgemm_tilelang':tl,
               ('ascendc_shifted_native' if native_shifted else 'ascendc_pre_plus_torch_apply'):ac}
    fusion_meta=None
    bound_meta=None
    if fusion:
        from mhc_fusion import make_candidates
        candidates,fusion_meta=make_candidates(projector,x,prev,alpha,bias)
        functions={'deepgemm_tilelang':tl,**candidates,'ascendc_shifted_native':ac}
        if bind:
            from mhc_bound import make_bound_candidate
            bound_fn,bound_meta=make_bound_candidate(projector,x,prev,alpha,bias)
            functions={'tilelang_apply_fused':candidates['tilelang_apply_fused'],
                       'tilelang_apply_bound':bound_fn,'ascendc_shifted_native':ac}
            fusion_meta['expected_kernel_counts']={name:2 for name in functions}
    meta={
        'fusion':fusion_meta,
        'bound_launch':bound_meta,
        'native_adaptation':native_metadata() if native_shifted else None,
        'projection_checks':projection_checks if native_shifted else None,
        'pattern':getattr(a,'mhc_pattern','random'),
        'weight_file':str(a.mhc_weights) if getattr(a,'mhc_weights',None) else None,
    }
    return functions,refs,[
        'layer_input','next_pre_mix','post_mix','comb_mix'],{
        **meta,'contract':dict(N=t,hidden=5120,hc_mult=4,flattened=20480,channels=24,hc_shifted=True,
                        norm_eps=1e-20,pre_eps=1e-6,sinkhorn_eps=1e-6,repeat=20,post_mult=2.),
        'excluded':['trailing layer RMSNorm','FP8 cast','previous Post'],
        'weights':('provided FP32 effective folded weights' if getattr(a,'mhc_weights',None) else
                   'synthetic FP32 fn already in effective folded-weight layout; not checkpoint weights'),
        'ascendc_extra_work':('no unused new-pre h_in or backward cache stores; inherited internal workspace included'
                             if native_shifted else 'PreV2 also computes unused current-pre h_in and invRms/hMix caches; included'),
        'ascendc_label':('adapted native MhcShiftedPreV2 + native Sinkhorn; no PyTorch numerical apply'
                         if native_shifted else 'native PreV2+native Sinkhorn+PyTorch shifted apply; NOT pure fused AscendC'),
        'comparison_available':True}


def shifted_native_case(a):
    return shifted_case(a,native_shifted=True)


def shifted_fusion_case(a):
    return shifted_case(a,native_shifted=True,fusion=True)


def shifted_bound_case(a):
    return shifted_case(a,native_shifted=True,fusion=True,bind=True)

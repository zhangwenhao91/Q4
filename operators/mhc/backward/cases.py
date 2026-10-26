"""Backward runners and an independent CPU autograd reference; NPU imports are lazy."""
import importlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'extended'))
from bench_forward import sinkhorn_ref
from projection_grad import projection_gradients


def load_bridge():
    from torch.utils.cpp_extension import load
    mod=importlib.import_module('cann_ops_transformer.ops.mhc.mhc_post.mhc_post')
    b=mod.mhc_post_op_builder
    return load(name='q4_mhc_backward',sources=[str(Path(__file__).with_name('aclnn_backward.cpp'))],
                extra_include_paths=b.get_absolute_paths(b.include_paths()),
                extra_cflags=b.cxx_args(),extra_ldflags=b.extra_ldflags(),verbose=True)


def reference_pre(x,phi,alpha,bias,repeat,eps,with_sinkhorn):
    flat=x.float().flatten(1)
    mix=(flat@phi.T)*(flat.square().mean(-1,keepdim=True)+eps).rsqrt()
    pre=(mix[:,:4]*alpha[0]+bias[:4]).sigmoid()+eps
    post=2*(mix[:,4:8]*alpha[1]+bias[4:8]).sigmoid()
    res=(mix[:,8:]*alpha[2]+bias[8:]).reshape(-1,4,4)
    hin=(x.float()*pre.unsqueeze(-1)).sum(1).bfloat16()
    return hin,post,sinkhorn_ref(res,repeat,eps) if with_sinkhorn else res


def make_case(a,t,d):
    import torch
    from tile_kernels.mhc import (mhc_post_bwd,mhc_post_fwd,mhc_sinkhorn_bwd,mhc_sinkhorn_fwd,
        mhc_reduce_partials_and_rmsnorm_fwd,mhc_reduce_partials_and_rmsnorm_bwd,
        mhc_pre_split_mixes_fwd,mhc_pre_split_mixes_bwd,mhc_pre_apply_mix_fwd,mhc_pre_apply_mix_bwd)
    if a.suite=='post':
        from cann_ops_transformer.ops import mhc_post,mhc_post_backward
        x0=torch.randn(1,t,d,dtype=torch.bfloat16,requires_grad=True)
        r0=torch.randn(1,t,4,d,dtype=torch.bfloat16,requires_grad=True)
        h0=(torch.rand(1,t,4,1)*2).requires_grad_()
        c0=torch.randn(1,t,4,4).softmax(-1).requires_grad_()
        g0=torch.randn(1,t,4,d,dtype=torch.bfloat16)*0.1
        y=(torch.einsum('btij,btid->btjd',c0,r0.float())+x0.float().unsqueeze(-2)*h0).bfloat16()
        refs=torch.autograd.grad(y,(x0,r0,h0,c0),g0)
        x,r,h,c,g=[z.detach().npu() for z in (x0,r0,h0,c0,g0)]
        def tl_forward(): return mhc_post_fwd(x,r,h,c)
        def ac_forward(): return mhc_post(r,c,x,h.squeeze(-1))
        def tl_backward(_): return mhc_post_bwd(x,r,h,c,g)
        def ac_backward(_):
            gr,gc,gx,gh=mhc_post_backward(g,r,c,x,h.squeeze(-1))
            return gx,gr,gh.unsqueeze(-1),gc
        label='tilelang'
        names=['grad_h_out','grad_residual','grad_h_post','grad_h_res']
    elif a.suite=='sinkhorn':
        b=load_bridge()
        x0=torch.randn(t,4,4,requires_grad=True)
        g0=torch.randn(t,4,4)*0.1
        refs=torch.autograd.grad(sinkhorn_ref(x0,a.repeat,a.eps),x0,g0)
        x,g=[v.detach().npu() for v in (x0,g0)]
        def tl_forward():
            y=torch.empty_like(x);mhc_sinkhorn_fwd(x,y,a.repeat,a.eps);return y
        def ac_forward(): return b.sinkhorn_cache(x,a.eps,a.repeat)
        def tl_backward(_):
            out=torch.empty_like(x);mhc_sinkhorn_bwd(g,x,out,a.repeat,a.eps);return (out,)
        def ac_backward(cache): return (b.sinkhorn_bwd(g,cache[1],cache[2]),)
        label='tilelang_recompute'
        names=['grad_logits']
    else:
        from fused_projection import HCProjection
        b=load_bridge()
        use_sk=a.suite=='pre_sinkhorn'
        x0=torch.randn(t,4,d,dtype=torch.bfloat16,requires_grad=True)
        phi0=(torch.empty(24,4*d).uniform_(-1,1)/(4*d)**0.5).requires_grad_()
        alpha0=torch.tensor([.1,.1,.1],requires_grad=True)
        bias0=(torch.randn(24)*.1).requires_grad_()
        ghi0=torch.randn(t,d,dtype=torch.bfloat16)*.1
        ghp0=torch.randn(t,4)*.1
        ghr0=torch.randn(t,4,4)*.1
        outputs=reference_pre(x0,phi0,alpha0,bias0,a.repeat,a.eps,use_sk)
        refs=torch.autograd.grad(outputs,(x0,phi0,alpha0,bias0),(ghi0,ghp0,ghr0))
        del outputs
        x,phi,alpha,bias,ghi,ghp,ghr=[v.detach().npu() for v in (x0,phi0,alpha0,bias0,ghi0,ghp0,ghr0)]
        projector=HCProjection(x,phi)
        def tl_forward():
            raw,sq=projector()
            raw_out=torch.empty(t,1,24,device=x.device)
            sq_out=torch.empty(t,1,device=x.device)
            mix=torch.empty(t,24,device=x.device)
            mhc_reduce_partials_and_rmsnorm_fwd(raw.view(1,t,1,24),sq.view(1,t,1),
                raw_out,sq_out,mix,rms_group_size=4*d,rms_eps=a.eps,n_splits=1)
            pre=torch.empty(t,4,device=x.device);post=torch.empty_like(pre)
            logits=torch.empty(t,16,device=x.device)
            mhc_pre_split_mixes_fwd(mix,alpha,bias,pre,post,logits,2.,a.eps)
            hin=torch.empty(t,d,dtype=x.dtype,device=x.device)
            mhc_pre_apply_mix_fwd(x,pre,hin)
            res=logits.reshape(t,4,4)
            if use_sk:
                normalized=torch.empty_like(res);mhc_sinkhorn_fwd(res,normalized,a.repeat,a.eps)
            else: normalized=res
            return raw_out,sq_out,mix,pre,post,res,hin,normalized
        def tl_backward(cache):
            raw,sq,mix,pre,post,logits,_,_=cache
            # Upstream accumulates into x_grad: reset on EVERY invocation (timed).
            direct=torch.zeros_like(x)
            gpre=mhc_pre_apply_mix_bwd(ghi,x,pre,direct)
            if use_sk:
                gres=torch.empty_like(ghr);mhc_sinkhorn_bwd(ghr,logits,gres,a.repeat,a.eps)
            else: gres=ghr
            gmix=torch.empty(t,24,device=x.device)
            ga=torch.empty_like(alpha);gb=torch.empty_like(bias)
            mhc_pre_split_mixes_bwd(gpre,ghp,gres.view(t,16),mix,post,alpha,bias,gmix,ga,gb,2.)
            graw=torch.empty_like(raw);gsq=torch.empty_like(sq)
            mhc_reduce_partials_and_rmsnorm_bwd(gmix,raw,sq,graw,gsq,rms_group_size=4*d,rms_eps=a.eps)
            # No fused HC backward in pinned DG: count ALL cast/GEMM/reduction gradient work.
            xf=x.float().flatten(1);gm=graw.view(t,24)
            indirect,gp=projection_gradients(xf,phi,gm,gsq.view(t,1))
            gx=(direct.float().flatten(1)+indirect).view_as(x).bfloat16()
            return gx,gp,ga,gb
        if use_sk:
            fmod=importlib.import_module('cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn.mhc_pre_sinkhorn')
            bmod=importlib.import_module('cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn_backward.mhc_pre_sinkhorn_backward')
            forward=fmod.mhc_pre_sinkhorn_op_builder.load()
            backward=bmod.mhc_pre_sinkhorn_backward_op_builder.load()
            def ac_forward(): return forward.mhc_pre_sinkhorn(x,phi,alpha,bias,4,a.repeat,a.eps,a.eps,True)
            def ac_backward(cache):
                _,_,_,hp,mul,inv,sums,norms=cache
                return backward.mhc_pre_sinkhorn_backward(ghi,ghp,ghr,x,phi,alpha,bias,hp,mul,inv,sums,norms,a.eps)
        else:
            def ac_forward(): return b.pre_cache(x,phi,alpha,bias,a.eps)
            def ac_backward(cache):
                _,hp,_,inv,mul,hpre=cache
                return b.pre_bwd(x,phi,alpha,ghi,ghp,ghr,inv,mul,hpre,hp,a.eps)
        label='tilekernels_plus_torch_projection_grad'
        names=['grad_x','grad_phi','grad_alpha','grad_bias']
    # Prepare caches once outside isolated backward timing, using each native forward.
    with torch.no_grad():
        tl_cache,ac_cache=tl_forward(),ac_forward()
        torch.npu.synchronize()
    if a.mode=='backward':
        tl=lambda:tl_backward(tl_cache)
        ac=lambda:ac_backward(ac_cache)
    else:
        tl=lambda:tl_backward(tl_forward())
        ac=lambda:ac_backward(ac_forward())
    return tl,ac,tuple(r.detach() for r in refs),names,label

#include <torch/extension.h>
#include "aclnn_common.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor> sinkhorn_cache(
    const at::Tensor& x, double eps, int64_t repeat) {
    TORCH_CHECK(x.dim()==3 && x.size(1)==4 && x.size(2)==4);
    int64_t t=x.size(0);
    auto y=at::empty_like(x);
    // Native forward owns the opaque per-core/tile cache packing.
    auto norm=at::empty({2*repeat*t*4*8},x.options());
    auto sum=at::empty({2*repeat*t*8},x.options());
    float e=static_cast<float>(eps);
    ACLNN_CMD(aclnnMhcSinkhorn,x,e,repeat,y,norm,sum);
    return {y,norm,sum};
}
at::Tensor sinkhorn_bwd(const at::Tensor& grad,const at::Tensor& norm,const at::Tensor& sum) {
    auto y=at::empty_like(grad);
    ACLNN_CMD(aclnnMhcSinkhornBackward,grad,norm,sum,y);
    return y;
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> pre_cache(
    const at::Tensor& x,const at::Tensor& phi,const at::Tensor& alpha,const at::Tensor& bias,double eps) {
    TORCH_CHECK(x.dim()==3 && x.size(1)==4);
    int64_t t=x.size(0), d=x.size(2), mode=1;
    auto hi=at::empty({t,d},x.options());
    auto hp=at::empty({t,4},phi.options());
    auto hr=at::empty({t,4,4},phi.options());
    auto inv=at::empty({t},phi.options());
    auto mix=at::empty({t,24},phi.options());
    auto pre=at::empty({t,4},phi.options());
    at::Tensor absent;
    ACLNN_CMD(aclnnMhcPreV2,x,phi,alpha,bias,absent,eps,eps,mode,hi,hp,hr,inv,mix,pre);
    return {hi,hp,hr,inv,mix,pre};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> pre_bwd(
    const at::Tensor& x,const at::Tensor& phi,const at::Tensor& alpha,
    const at::Tensor& ghi,const at::Tensor& ghp,const at::Tensor& ghr,
    const at::Tensor& inv,const at::Tensor& mix,const at::Tensor& pre,const at::Tensor& post,double eps) {
    auto gx=at::empty_like(x), gp=at::empty_like(phi), ga=at::empty_like(alpha);
    auto gb=at::empty({24},phi.options());
    at::Tensor absent;
    float e=static_cast<float>(eps);
    int64_t mode=1;
    ACLNN_CMD(aclnnMhcPreBackwardV2,x,phi,alpha,ghi,ghp,ghr,inv,mix,pre,post,
              absent,absent,e,mode,gx,gp,ga,gb,absent);
    return {gx,gp,ga,gb};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("sinkhorn_cache",&sinkhorn_cache);
    m.def("sinkhorn_bwd",&sinkhorn_bwd);
    m.def("pre_cache",&pre_cache);
    m.def("pre_bwd",&pre_bwd);
}

#include <torch/extension.h>
#include "aclnn_common.h"

// No shifted native Pre entry exists in this pinned source. Export the actual
// next_pre cache from PreV2, then compose old-pre application outside this bridge.
std::tuple<at::Tensor,at::Tensor,at::Tensor> pre_coefficients(
    const at::Tensor& x,const at::Tensor& phi,const at::Tensor& alpha,
    const at::Tensor& bias,double norm_eps,double pre_eps) {
    auto t=x.size(0),d=x.size(2);
    auto hin=at::empty({t,d},x.options());
    auto hp=at::empty({t,4},phi.options());
    auto hr=at::empty({t,4,4},phi.options());
    auto inv=at::empty({t},phi.options());
    auto mix=at::empty({t,24},phi.options());
    auto pre=at::empty({t,4},phi.options());
    at::Tensor absent;int64_t mode=1;
    ACLNN_CMD(aclnnMhcPreV2,x,phi,alpha,bias,absent,norm_eps,pre_eps,mode,hin,hp,hr,inv,mix,pre);
    return {pre,hp,hr};
}
at::Tensor sinkhorn(const at::Tensor& x,double eps,int64_t repeat) {
    auto y=at::empty_like(x);at::Tensor absent;float e=static_cast<float>(eps);
    ACLNN_CMD(aclnnMhcSinkhorn,x,e,repeat,y,absent,absent);
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("pre_coefficients",&pre_coefficients);
    m.def("sinkhorn",&sinkhorn);
}

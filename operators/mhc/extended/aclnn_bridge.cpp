#include <torch/extension.h>
#include "aclnn_common.h"

// Q4 thin ACLNN bridge. Kernel implementations remain in ops-transformer.
at::Tensor sinkhorn(const at::Tensor& x, double eps, int64_t iters) {
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 4 && x.size(2) == 4);
    TORCH_CHECK(x.scalar_type() == at::kFloat && x.is_contiguous());
    auto out = at::empty_like(x);
    at::Tensor absent;
    float acl_eps = static_cast<float>(eps);
    ACLNN_CMD(aclnnMhcSinkhorn, x, acl_eps, iters, out, absent, absent);
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> pre(
    const at::Tensor& x, const at::Tensor& phi, const at::Tensor& alpha,
    const at::Tensor& bias, double eps) {
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 4);
    auto t = x.size(0), d = x.size(2);
    auto hin = at::empty({t, d}, x.options());
    auto post = at::empty({t, 4}, phi.options());
    auto res = at::empty({t, 4, 4}, phi.options());
    at::Tensor absent;
    // No gamma or backward caches. aclnnMhcPre selects FP32 Cube mode.
    ACLNN_CMD(aclnnMhcPre, x, phi, alpha, bias, absent, eps, eps,
              hin, post, res, absent, absent, absent);
    return {hin, post, res};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sinkhorn", &sinkhorn);
    m.def("pre", &pre);
}

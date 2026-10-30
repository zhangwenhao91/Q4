#include <torch/extension.h>
#include "aclnn_common.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> shifted(
    const at::Tensor& x, const at::Tensor& prev, const at::Tensor& phi,
    const at::Tensor& alpha, const at::Tensor& bias) {
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 4 && x.size(2) == 5120);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(prev.dim() == 3 && prev.size(0) == x.size(0) && prev.size(1) == 4 && prev.size(2) == 1);
    TORCH_CHECK(prev.scalar_type() == at::kFloat && prev.is_contiguous());
    auto t = x.size(0);
    auto hi = at::empty({t, 5120}, x.options());
    auto pre = at::empty({t, 4}, phi.options());
    auto post = at::empty_like(pre);
    auto logits = at::empty({t, 4, 4}, phi.options());
    at::Tensor absent;
    auto prev2d = prev.view({t, 4});
    int64_t mode = 1;
    ACLNN_CMD(aclnnMhcShiftedPreV2, x, phi, alpha, bias, prev2d, 1e-20, 1e-6, mode,
              hi, post, logits, absent, absent, pre);
    auto comb = at::empty_like(logits);
    float eps = 1e-6f;
    int64_t repeat = 20;
    ACLNN_CMD(aclnnMhcSinkhorn, logits, eps, repeat, comb, absent, absent);
    return {hi, pre.view({t, 4, 1}), post.view({t, 4, 1}), comb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("shifted", &shifted); }

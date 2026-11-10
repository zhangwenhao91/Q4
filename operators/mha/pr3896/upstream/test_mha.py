"""pytest test for example_mha.py — MHA Mix kernel (SimdVF softmax)."""

import torch
import tilelang

from example_mha import mha, ref_program


def _check_mha(kernel, D, SEQ_LEN):
    device = torch.device("npu")
    torch.manual_seed(42)
    Q = torch.randn(SEQ_LEN, D, dtype=torch.bfloat16, device=device)
    K = torch.randn(SEQ_LEN, D, dtype=torch.bfloat16, device=device)
    V = torch.randn(SEQ_LEN, D, dtype=torch.bfloat16, device=device)

    O = kernel(Q, K, V)
    torch.npu.synchronize()

    assert O.shape == Q.shape and O.dtype == torch.float32
    expected = ref_program(Q, K, V)
    max_diff = (O.cpu().float() - expected.cpu()).abs().max().item()
    rel = max_diff / expected.cpu().abs().max().item()
    assert rel < 0.05, f"SEQ_LEN={SEQ_LEN} rel={rel:.4f}"


def _compile_mha(seq_len):
    return tilelang.compile(
        mha(128, seq_len),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


def test_mha():
    _check_mha(_compile_mha(4096), 128, 4096)


def test_mha_dynamic():
    kernel = _compile_mha(None)
    for seq_len in (129, 1, 63, 64, 65, 127, 128, 193, 255, 256, 4096, 4097):
        _check_mha(kernel, 128, seq_len)


if __name__ == "__main__":
    test_mha()
    print("PASS: test_mha")
    test_mha_dynamic()
    print("PASS: test_mha_dynamic")

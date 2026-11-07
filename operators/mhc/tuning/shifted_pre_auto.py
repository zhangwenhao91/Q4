"""Preallocated H5120 shifted Pre with an N-based kernel selection.

N must be a positive multiple of 128. N < 512 selects the small-N two-kernel
implementation. N >= 512 selects the original three-kernel K8 implementation.
Bind once per input set and use the returned launcher on one stream.
Each invocation overwrites the outputs from the previous invocation.
"""


def bind_shifted_pre(x, prev_pre, phi, alpha, bias):
    """Return (launcher, metadata) for the selected preallocated implementation."""
    if len(x.shape) != 3 or tuple(x.shape[1:]) != (4, 5120):
        raise ValueError('Expected residual[N,4,5120]')
    n = x.shape[0]
    if n <= 0 or n % 128:
        raise ValueError('N must be a positive multiple of 128')

    if n < 512:
        if __package__:
            from .shifted_pre_k8_fused_preallocated import bind_shifted_pre as bind
        else:
            from shifted_pre_k8_fused_preallocated import bind_shifted_pre as bind
        implementation = 'two-kernel K8'
    else:
        if __package__:
            from .shifted_pre_k8_preallocated import bind_preallocated_k8 as bind
        else:
            from shifted_pre_k8_preallocated import bind_preallocated_k8 as bind
        implementation = 'three-kernel K8'

    launcher, metadata = bind(x, prev_pre, phi, alpha, bias)
    metadata['dispatch'] = dict(tokens=n, threshold=512, implementation=implementation)
    return launcher, metadata

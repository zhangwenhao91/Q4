"""Compare pinned PR3896 static/dynamic BF16 kernels with original TL and AC."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'spec_v1'))
import bench
import mha_bf16

original_make_case = mha_bf16.make_case
dynamic_kernel = None
tail_length = None


def load_pr():
    folder = HERE / 'upstream'
    manifest = json.loads((folder / 'source.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('PR source checksum mismatch: ' + name)
    name = 'q4_pr3896_core'
    spec = importlib.util.spec_from_file_location(name, folder / 'core.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, manifest


def make_tail_case(a):
    """Unpadded logical inputs; static TL does not support these lengths."""
    global dynamic_kernel
    import math
    import os
    import subprocess
    import torch
    import tilelang
    import tilelang.ascend.language as T
    import cann_ops_transformer
    n = tail_length
    module, manifest = load_pr()
    native = HERE.parents[2] / 'third_party/ops-transformer-q4fa'
    rev = subprocess.check_output(['git', '-C', str(native), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != 'ba5162411c919e15fbabb28298e8b420c11ad905':
        raise RuntimeError('Native source revision mismatch')
    opp = Path(os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(os.pathsep)[0]).resolve()
    if (HERE.parents[2] / 'third_party/q4fa-opp').resolve() not in opp.parents:
        raise RuntimeError('Select q4fa native vendor first')
    libraries = list((opp / 'op_api/lib').glob('*.so'))
    if not libraries:
        raise RuntimeError('Missing native libraries')
    if dynamic_kernel is None:
        length = T.dynamic('seq_len')
        dynamic_kernel = tilelang.compile(module.flash_attention_fwd(
            q_len=length, kv_len=length, head_dim=128, out_dtype='bfloat16'),
            out_idx=-1, pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    cpu = [torch.randn(n, 128, dtype=torch.bfloat16) for _ in range(3)]
    q, k, v = [x.npu() for x in cpu]
    qf, kf, vf = [x.float() for x in cpu]
    ref = torch.empty(n, 128, dtype=torch.float32)
    for start in range(0, n, 128):
        ref[start:start+128] = ((qf[start:start+128] @ kf.T) / math.sqrt(128)).softmax(-1) @ vf
    op = cann_ops_transformer.ops
    def prepare_metadata():
        return op.flash_attn_metadata(1, 1, 128, batch_size=1,
            max_seqlen_q=n, max_seqlen_kv=n, is_grad_enabled=False,
            mask_mode=0, win_left=-1, win_right=-1,
            layout_q='BSND', layout_kv='BSND', layout_out='BSND')
    metadata = prepare_metadata()
    ac = mha_bf16.native_runner(op, *[x.view(1, n, 1, 128) for x in (q, k, v)],
        prepare_metadata, metadata, a.ac_metadata, seq_len=n)
    kernel = dynamic_kernel
    return {'pr_dynamic': lambda: (kernel(q, k, v),), 'ascendc': ac}, (ref.bfloat16(),), ['O'], {
        'source': manifest, 'comparison_available': True,
        'contract': dict(Q=[n,128], K=[n,128], V=[n,128], input_dtype='bfloat16',
                         output_dtype='bfloat16', causal=False, dropout=0),
        'native_source_revision': rev, 'native_vendor': str(opp),
        'native_library_sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in libraries},
        'metadata_mode': a.ac_metadata, 'static_path': 'unsupported: length not divisible by 128',
        'reference': 'blocked CPU FP32 attention rounded once to BF16',
        'timing_boundary': 'metadata ' + a.ac_metadata + '; no host padding or output cast',
        'dynamic_reuse': 'one compiled kernel reused across every length in this process'}


def make_case(a):
    if tail_length is not None:
        return make_tail_case(a)
    def build(source, q, k, v):
        global dynamic_kernel
        import tilelang
        import tilelang.ascend.language as T
        folder = HERE / 'upstream'
        manifest = json.loads((folder / 'source.json').read_text())
        for name, digest in manifest['sha256'].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('PR source checksum mismatch: ' + name)
        name = 'q4_pr3896_core'
        spec = importlib.util.spec_from_file_location(name, folder / 'core.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        def compile_kernel(length):
            return tilelang.compile(module.flash_attention_fwd(
                q_len=length, kv_len=length, head_dim=128, out_dtype='bfloat16'),
                out_idx=-1, pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
        static = compile_kernel(a.mha_seqlen)
        if dynamic_kernel is None:
            dynamic_kernel = compile_kernel(T.dynamic('seq_len'))
        dynamic = dynamic_kernel
        return {'pr_static': lambda: (static(q, k, v),),
                'pr_dynamic': lambda: (dynamic(q, k, v),)}, manifest
    return original_make_case(a, extra_builder=build)


if __name__ == '__main__':
    tails = '--tails' in sys.argv
    if tails:
        sys.argv.remove('--tails')
    # Two lengths in one process deliberately reuse the same dynamic kernel.
    prefix = Path(sys.argv[sys.argv.index('--output') + 1])
    base_args = list(sys.argv)
    sys.modules['mha_bf16'] = types.SimpleNamespace(make_case=make_case)
    lengths = (1, 63, 64, 65, 127, 129, 193, 255, 3583, 3585, 4095, 4097) if tails else (4096, 3584)
    for length in lengths:
        tail_length = length if tails else None
        sys.argv = list(base_args)
        sys.argv[sys.argv.index('--output') + 1] = str(prefix / str(length))
        sys.argv += ['--mha-seqlen', str(4096 if tails else length)]
        bench.main()

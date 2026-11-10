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


def make_case(a):
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
    # Two lengths in one process deliberately reuse the same dynamic kernel.
    prefix = Path(sys.argv[sys.argv.index('--output') + 1])
    base_args = list(sys.argv)
    sys.modules['mha_bf16'] = types.SimpleNamespace(make_case=make_case)
    for length in (4096, 3584):
        sys.argv = list(base_args)
        sys.argv[sys.argv.index('--output') + 1] = str(prefix / str(length))
        sys.argv += ['--mha-seqlen', str(length)]
        bench.main()

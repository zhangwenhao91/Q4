"""Identity checks and thin bridge for the isolated matched AscendC experiment."""
import hashlib
import ctypes
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'operators/mhc/matched'))
from projection import DynamicHCProjection


def native_metadata():
    expected = (ROOT / 'third_party/q4mhcmatch-opp/vendors/q4mhcmatch_transformer').resolve()
    actual = Path(os.environ.get('ASCEND_CUSTOM_OPP_PATH', '/missing')).resolve()
    if actual != expected:
        raise RuntimeError('Select the isolated q4mhcmatch vendor before testing')
    lib = actual / 'op_api/lib/libcust_opapi.so'
    if not lib.is_file():
        raise RuntimeError('Missing matched native library: ' + str(lib))
    handle = ctypes.CDLL(str(lib))
    for symbol in ('aclnnMhcShiftedPreV2', 'aclnnMhcShiftedPreV2GetWorkspaceSize', 'aclnnMhcSinkhorn'):
        getattr(handle, symbol)
    manifest = ROOT / 'third_party/mhc-matched-source/source.json'
    source = json.loads(manifest.read_text())
    generated = ROOT / 'third_party/ops-transformer-q4mhcmatch/mhc/mhc_shifted_pre'
    for relative, digest in source['files'].items():
        if hashlib.sha256((generated / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Generated native source changed: ' + relative)
    return dict(source=source, library=str(lib), library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
                note='source/build provenance plus library hash; device kernel names checked by profiler')


def load_matched_bridge():
    native_metadata()
    from torch.utils.cpp_extension import load
    m = importlib.import_module('cann_ops_transformer.ops.mhc.mhc_post.mhc_post')
    b = m.mhc_post_op_builder
    return load(name='q4_mhc_shifted_native', sources=[str(ROOT / 'operators/mhc/matched/bridge.cpp')],
                extra_include_paths=b.get_absolute_paths(b.include_paths()), extra_cflags=b.cxx_args(),
                extra_ldflags=b.extra_ldflags(), verbose=True)

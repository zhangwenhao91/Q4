"""Verify pinned upstream files and extract the original v3 binding verbatim."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REV = 'e838a32888113cd20e1b5bad2053d1b4a06bf935'
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'third_party/vllm-ascend-q4-v3'
RECORD = ROOT / 'third_party/vllm-v3-record'
VENDOR = ROOT / 'third_party/q4vllmv3-opp/vendors/q4vllmv3_transformer'
PREFIXES = ('csrc/moe/hc_pre/', 'csrc/aclnn_torch_adapter/')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(source=SOURCE):
    rev = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != REV:
        raise RuntimeError('Unexpected vLLM-Ascend revision: ' + rev)
    names = subprocess.check_output(['git', '-C', str(source), 'ls-tree', '-r', '--name-only', 'HEAD'], text=True).splitlines()
    files = [v for v in names if v.startswith(PREFIXES) or v in ('csrc/torch_binding.cpp', 'csrc/build.sh')]
    hashes = {}
    for name in files:
        original = subprocess.check_output(['git', '-C', str(source), 'show', 'HEAD:' + name])
        if (source / name).read_bytes().replace(b'\r\n', b'\n') != original.replace(b'\r\n', b'\n'):
            raise RuntimeError('Upstream file changed: ' + name)
        hashes[name] = hashlib.sha256(original).hexdigest()
    return hashes


def extract(text):
    start = text.index('constexpr int64_t HC_PRE_HC_LIMIT = 4;')
    end = text.index('void inplace_partial_rotary_mul_npu(', start)
    functions = text[start:end]
    marker = '    ops.def(\n        "npu_hc_pre_v3("'
    start = text.index(marker)
    final = '    ops.impl("npu_hc_pre_v3", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_v3_npu);'
    end = text.index(final, start) + len(final)
    registration = text[start:end]
    license_header = text[:text.index('#include')]
    header = '''#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <array>
#include <string>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include "aclnn_torch_adapter/op_api_common.h"
namespace vllm_ascend {
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
'''
    footer = '''
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
'''
    address = '''
}
TORCH_LIBRARY(q4_vllm_v3, ops) {
    ops.def("api_address() -> int", []() -> int64_t {
        return reinterpret_cast<int64_t>(GetOpApiFuncAddr("aclnnHcPre"));
    });
}
'''
    return license_header + header + functions + footer + registration + address


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--record', type=Path, default=RECORD)
    p.add_argument('--installer', type=Path, help='record completed isolated build')
    a = p.parse_args()
    hashes = verify(a.source)
    text = (a.source / 'csrc/torch_binding.cpp').read_text(encoding='utf-8')
    bridge = extract(text)
    a.record.mkdir(parents=True, exist_ok=True)
    path = a.record / 'v3_binding.cpp'
    path.write_text(bridge, encoding='utf-8')
    meta = dict(upstream='https://github.com/vllm-project/vllm-ascend', revision=REV,
        files=hashes, binding_sha256=sha(path),
        binding_note='Original HcPre helper functions, v3 function and torch registration copied verbatim; original EXEC_NPU_CMD header; only includes/namespace and diagnostic registration added')
    (a.record / 'source.json').write_text(json.dumps(meta, indent=2))
    if a.installer:
        library = VENDOR / 'op_api/lib/libcust_opapi.so'
        build = dict(revision=REV, library=str(library), library_sha256=sha(library),
            installer=str(a.installer.resolve()), installer_sha256=sha(a.installer),
            source_files=hashes, build_command='bash build.sh --pkg --soc=ascend950 --ops=hc_pre --vendor_name=q4vllmv3')
        (a.record / 'build.json').write_text(json.dumps(build, indent=2))
    print('Pinned unmodified kernel and adapter verified; original v3 binding extracted.')


if __name__ == '__main__':
    main()

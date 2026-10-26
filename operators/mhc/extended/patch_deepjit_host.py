"""Repair DeepJIT toolkit discovery for the pinned DeepGEMM-Ascend dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import os
import subprocess

PIN = '65f513952b8771408a19202f65e36398522db6db'
OLD = '''        const auto asc_devkit_include = home_path / "aarch64-linux/asc/include/adv_api";
        DJ_HOST_ASSERT(std::filesystem::is_directory(asc_devkit_include),
                       "Ascend adv_api include directory was not found: {}", asc_devkit_include.string());'''
NEW = '''        // Q4: discover headers for the host architecture, without changing device kernels.
#if defined(__x86_64__)
        const std::string host_arch = "x86_64-linux";
#elif defined(__aarch64__)
        const std::string host_arch = "aarch64-linux";
#else
        DJ_PANIC("Unsupported Ascend host architecture for toolkit discovery");
#endif
        const std::vector<std::filesystem::path> include_candidates = {
            home_path / host_arch / "asc/include/adv_api",
            home_path / "asc/include/adv_api",
        };
        std::filesystem::path asc_devkit_include;
        for (const auto& candidate : include_candidates) {
            if (std::filesystem::is_directory(candidate)) {
                asc_devkit_include = candidate;
                break;
            }
        }
        DJ_HOST_ASSERT(not asc_devkit_include.empty(),
                       "Ascend adv_api include directory was not found for host {} under {}",
                       host_arch, home_path.string());'''


def patch_text(source):
    if NEW in source:
        return source
    if source.count(OLD) != 1:
        raise RuntimeError('Source differs from the pinned patch context; refusing to guess')
    return source.replace(OLD, NEW)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    a = p.parse_args()
    deepjit = a.source.resolve()/'third-party/deep-jit'
    revision = subprocess.check_output(['git', '-C', str(deepjit), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise RuntimeError(f'Unexpected DeepJIT revision {revision}; expected {PIN}')
    home = Path(os.environ['ASCEND_HOME_PATH'])
    arch = {'x86_64': 'x86_64-linux', 'amd64': 'x86_64-linux', 'aarch64': 'aarch64-linux'}[platform.machine().lower()]
    candidates = [home/arch/'asc/include/adv_api', home/'asc/include/adv_api']
    for candidate in candidates:
        print('candidate:', candidate, 'exists:', candidate.is_dir())
    if not any(c.is_dir() for c in candidates):
        raise RuntimeError('No host/common adv_api directory found. No source was modified; inspect the CANN installation.')
    header = deepjit/'include/deep_jit/backend/ascend/backend.hpp'
    original = header.read_text(encoding='utf-8')
    patched = patch_text(original)
    a.record.mkdir(parents=True, exist_ok=True)
    base = subprocess.check_output(['git', '-C', str(deepjit), 'show', 'HEAD:include/deep_jit/backend/ascend/backend.hpp'])
    (a.record/'backend.hpp.original').write_bytes(base)
    header.write_text(patched, encoding='utf-8', newline='\n')
    diff = subprocess.check_output(['git', '-C', str(deepjit), 'diff', '--', 'include/deep_jit/backend/ascend/backend.hpp'])
    (a.record/'deepjit-host-discovery.patch').write_bytes(diff)
    meta = dict(deepjit_revision=revision, host=arch, candidates=[str(c) for c in candidates],
                original_sha256=hashlib.sha256(base).hexdigest(),
                patched_sha256=hashlib.sha256(header.read_bytes()).hexdigest())
    (a.record/'patch.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('Header patched. Rebuild the deep_gemm native extension; restarting Python alone is insufficient.')


if __name__ == '__main__':
    main()

"""Expose per-AIV squared-sum GM regions without changing their linear addresses."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import difflib

PIN = '3a35326271b069b8b5bf1a754435acd6cc91575b'
REL = 'deep_gemm/tilelang_ops/hc_prenorm.py'
EDITS = [
    ('    sqrsum: T.Tensor[(num_tokens,), T.float32]',
     '    sqrsum: T.Tensor[(T.ceildiv(num_tokens, block_num_tokens), num_aivs, half_block_num_tokens), T.float32]'),
    ('                                    sqrsum[token_begin : token_begin + actual_aiv_num_tokens],',
     '                                    sqrsum[token_block, aiv_id, 0:actual_aiv_num_tokens],'),
    ('''    if m == 0:
        return
    deterministic = _C.get_deterministic_algorithms()''',
     '''    if m == 0:
        return
    # Q4 explicit AIV axis: identical linear storage, limited to complete token blocks.
    assert m % 128 == 0, 'Q4 HC layout adapter currently requires M divisible by 128'
    out_sqr_sum = out_sqr_sum.view(m // 128, 2, 64)
    deterministic = _C.get_deterministic_algorithms()'''),
]
MARKER = '# Q4 explicit AIV axis:'


def patch_text(source):
    if MARKER in source:
        if not all(new in source for _, new in EDITS):
            raise RuntimeError('Incomplete or modified Q4 HC patch')
        return source
    for old, new in EDITS:
        if source.count(old) != 1:
            raise RuntimeError('Pinned HC source context mismatch; refusing to guess')
        source = source.replace(old, new)
    ast.parse(source)
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    a = p.parse_args()
    revision = subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise RuntimeError('Unexpected DeepGEMM-Ascend revision: ' + revision)
    file = a.source/REL
    base = subprocess.check_output(['git', '-C', str(a.source), 'show', 'HEAD:'+REL]).decode('utf-8')
    source = file.read_text(encoding='utf-8')
    expected = patch_text(base)
    if source not in (base, expected):
        raise RuntimeError('HC source contains other modifications; preserve them before patching')
    a.record.mkdir(parents=True, exist_ok=True)
    (a.record/'hc_prenorm.py.original').write_text(base, encoding='utf-8')
    file.write_text(expected, encoding='utf-8', newline='\n')
    diff = ''.join(difflib.unified_diff(base.splitlines(True), expected.splitlines(True),
                                      fromfile='a/'+REL, tofile='b/'+REL))
    (a.record/'hc-explicit-aiv-layout.patch').write_text(diff, encoding='utf-8')
    meta = dict(revision=revision, supported_tokens='positive multiple of 128',
                original_sha256=hashlib.sha256(base.encode()).hexdigest(),
                patched_sha256=hashlib.sha256(expected.encode()).hexdigest())
    (a.record/'patch.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('Explicit AIV squared-sum layout patched; synchronization checks remain enabled.')


if __name__ == '__main__':
    main()

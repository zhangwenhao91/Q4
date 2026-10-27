"""Prepare the pinned, bundled Ascend MHA source offline; no compiler reinstall."""
import hashlib
import json
from pathlib import Path

REV='80364b3d9758854a9692cc0f14089025f54d3f3f'
FILES=['core.py','example_mha.py','test_mha.py','README.md','LICENSE','THIRDPARTYNOTICES']


def prepare(dst, bundle=None):
    bundle=Path(bundle) if bundle is not None else Path(__file__).with_name('upstream_mha')
    dst=Path(dst)
    manifest=json.loads((bundle/'source.json').read_text(encoding='utf-8'))
    if manifest['revision']!=REV or set(manifest['sha256'])!=set(FILES):
        raise RuntimeError('Bundled source manifest does not match the pinned contract')
    data={}
    # Validate every source and existing target before copying anything.
    for name in FILES:
        data[name]=(bundle/name).read_bytes()
        digest=hashlib.sha256(data[name]).hexdigest()
        if digest!=manifest['sha256'][name]:
            raise RuntimeError(f'Bundled source checksum mismatch: {name}')
        target=dst/name
        if target.exists() and target.read_bytes()!=data[name]:
            raise RuntimeError(f'Local file differs; preserve it before proceeding: {target}')
    dst.mkdir(parents=True,exist_ok=True)
    for name in FILES:
        (dst/name).write_bytes(data[name])
        print(name,manifest['sha256'][name],flush=True)
    (dst/'source.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Offline source ready:',dst,'; installed TileLang was not changed.')


def main():
    prepare(Path(__file__).resolve().parents[2]/'third_party'/'tilelang-mha-spec-v1')


if __name__=='__main__': main()

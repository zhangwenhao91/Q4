"""Download ONLY the pinned Ascend MHA example; preserve installed TileLang."""
import hashlib
import json
from pathlib import Path
import urllib.request

REV='80364b3d9758854a9692cc0f14089025f54d3f3f'
FILES=['core.py','example_mha.py','test_mha.py','README.md']


def main():
    root=Path(__file__).resolve().parents[2]
    dst=root/'third_party'/'tilelang-mha-spec-v1'
    dst.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name in FILES:
        url=f'https://raw.githubusercontent.com/tile-ai/tilelang/{REV}/examples/ascend/flash_attention/{name}'
        with urllib.request.urlopen(url,timeout=60) as response: data=response.read()
        path=dst/name
        if path.exists() and path.read_bytes()!=data:
            raise RuntimeError(f'Local file differs; preserve it before proceeding: {path}')
        path.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
        print(name,hashes[name],flush=True)
    (dst/'source.json').write_text(json.dumps(dict(repository='https://github.com/tile-ai/tilelang',revision=REV,sha256=hashes),indent=2))
    print('Source ready:',dst,'; installed TileLang was not changed.')


if __name__=='__main__': main()

"""Seed exact dependency archives from an old local build, verified against CANN master-059."""
import hashlib
from pathlib import Path
import shutil

# Upstream: cann/cmake tag master-059, third_party/{json,eigen,abseil-cpp,protobuf}.cmake.
# Copy archives into locations explicitly checked BEFORE each upstream HTTPS URL.
DEPENDENCIES=[
    ('json-3.12.0.tar.gz','json-3.12.0.tar.gz','4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187'),
    ('eigen-5.0.0.tar.gz','eigen-5.0.0.tar.gz','93f7f0462988b934e632a9fba58af55192ffceae38e8f46233f4f62cb1e79370'),
    ('abseil-cpp-20230802.1.tar.gz','abseil-cpp-20230802.1.tar.gz','987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed'),
    ('protobuf-25.1.tar.gz','protobuf-25.1.tar.gz','9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42'),
    ('backport-CVE-2025-0838.patch','patch/backport-CVE-2025-0838.patch','a4e4ef6fc4f4c1ed871a3472c5e650068aaadaad2adfef2c8ccb00918bb36d8c'),
]


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def seed(source,target,dependencies=DEPENDENCIES,bundle=None):
    source=Path(source).resolve();target=Path(target).resolve()
    bundle=Path(bundle) if bundle is not None else Path(__file__).with_name('dependency_archives')
    plans=[];missing=[]
    for name,relative,digest in dependencies:
        dst=(target/relative).resolve()
        if not dst.is_relative_to(target): raise RuntimeError('Cache target escapes its directory')
        if dst.is_file() and sha256(dst)==digest:
            print('Verified existing cache:',dst,flush=True);continue
        candidates=list(source.rglob(name))
        if (bundle/name).is_file(): candidates.append(bundle/name)
        # Some previous CANN releases saved the protobuf archive under this alias.
        if name=='protobuf-25.1.tar.gz': candidates+=list(source.rglob('protobuf-all-25.1.tar.gz'))
        match=next((p for p in candidates if p.is_file() and sha256(p)==digest),None)
        if match is None:
            missing.append(name)
            print('Missing verified cache:',name,'candidates:',[str(p) for p in candidates],flush=True)
        else: plans.append((match,dst,digest))
    if missing:
        raise RuntimeError('Offline dependencies incomplete: '+', '.join(missing)+
                           '. Stop before rebuilding; provide cache output for diagnosis.')
    for src,dst,digest in plans:
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        if sha256(dst)!=digest: raise RuntimeError('Copied cache checksum mismatch: '+str(dst))
        print('Reused verified dependency:',src,'->',dst,'SHA256='+digest,flush=True)
    print('All dependency caches ready; no TLS bypass or upstream source changes.',flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    seed(root/'third_party'/'ops-transformer'/'third_party',
         root/'third_party'/'ops-transformer-q4fa'/'third_party')

"""Export validated MFU timing evidence for offline review; no NPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from analyze import analyze


def export_capture(source,destination,peaks):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists() or destination==source or source in destination.parents:
        raise ValueError('Use a new output directory outside the original capture')
    result=analyze(source,peaks)
    plan=json.loads((source/'plan.json').read_text(encoding='utf-8'))
    files=[source/'plan.json']
    for case in plan['cases']:
        folder=(source/case['directory']).resolve()
        if source not in folder.parents:raise ValueError('Case path escapes input directory')
        files.extend(folder/name for name in ('metadata.json','correctness.json','timings.csv','summary.json'))
    files=sorted(set(files),key=str)
    if sum(p.stat().st_size for p in files)>20*1024**2:raise ValueError('Export exceeds 20 MiB')
    destination.mkdir(parents=True,exist_ok=False)
    manifest={'source':str(source),'note':'Validated existing timing evidence; no new NPU work.',
              'copied_files':[]}
    for file in files:
        target=destination/file.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file,target)
        manifest['copied_files'].append(dict(path=file.relative_to(source).as_posix(),
            bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (destination/'mfu.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (destination/'peak_config.json').write_text(json.dumps(peaks,indent=2)+'\n',encoding='utf-8')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--peak-json',type=Path,default=Path(__file__).with_name('peaks')/'ascend950pr_28aic.json')
    a=p.parse_args()
    peaks=json.loads(a.peak_json.read_text(encoding='utf-8'))
    manifest=export_capture(a.input,a.output,peaks)
    print(f"Exported {len(manifest['copied_files'])} evidence files plus analysis/config/manifest to {a.output}")
    print('No new timing. Commit only this exported directory for offline review.')


if __name__=='__main__':main()

"""Copy bounded analysis artifacts from an existing capture, without running NPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def export_capture(source, destination):
    source=Path(source).resolve()
    destination=Path(destination).resolve()
    if not source.is_dir():raise ValueError('Capture directory does not exist')
    if destination.exists():raise ValueError('Output already exists; preserve it or choose a new output')
    if source==destination or source in destination.parents:
        raise ValueError('Output must be outside capture directory')
    selected=[]
    for impl in ('tilelang_s2buf','tilelang_kv2'):
        base=source/impl
        traces=list((base/'op').glob('OPPROF_*/trace.json'))
        if len(traces)!=1:raise ValueError(f'{impl}: expected exactly one trace.json')
        # Confirm it is readable JSON, but do not assume an event schema or time unit.
        json.loads(traces[0].read_text(encoding='utf-8-sig'))
        op=traces[0].parent
        required=[traces[0],base/'app/metadata.json',base/'validate/correctness.json',
                  base/'command.txt',base/'collect.log',base/'verified.json']
        for path in required:
            if not path.is_file():raise ValueError(f'Missing {path}')
        selected.extend(required)
        selected.extend(op.glob('*.csv'))
        for path in (op/'visualize_data.bin',base/'app/selected_core.py',base/'app/selected_kernel.cpp'):
            if path.is_file():selected.append(path)
    for name in ('tool-version.txt','tool-help.txt'):
        path=source/name
        if path.is_file():selected.append(path)
    selected=sorted(set(selected),key=str)
    if any(p.stat().st_size>10*1024**2 for p in selected):
        raise ValueError('An analysis artifact exceeds 10 MiB; inspect before exporting')
    if sum(p.stat().st_size for p in selected)>20*1024**2:
        raise ValueError('Export exceeds 20 MiB; inspect before exporting')
    destination.mkdir(parents=True,exist_ok=False)
    manifest={'source_directory':str(source),'files':[],
              'note':'Existing capture only; dump/timeline.bin.* excluded. No new measurement.'}
    for path in selected:
        relative=path.relative_to(source)
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        manifest['files'].append({'path':relative.as_posix(),'bytes':target.stat().st_size,
                                  'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=export_capture(a.input,a.output)
    print(f"Exported {len(result['files'])} files, {sum(f['bytes'] for f in result['files'])} bytes to {a.output}")
    print('No kernel execution or new profiling. Commit this output directory for local analysis.')


if __name__=='__main__':main()

"""Reject false-success collector exits using target state, logs and result files."""
import argparse
import csv
import json
from pathlib import Path
import re

FATAL_PATTERNS=(
    r'Traceback \(most recent call last\):',
    r'Child process exited with return value [1-9]\d*',
    r'Profiling results are empty',
    r'No kernel profiling data was generated',
    r'No profiling data matched',
    r'Get profiling data failed',
    r'Profiling data parse failed',
    r'\[ERROR\]',
)


def verify_collection(folder):
    folder=Path(folder)
    metadata=folder/'app/metadata.json'
    if not metadata.is_file():
        raise RuntimeError('Missing capture-app metadata; target execution not confirmed')
    try:
        state=json.loads(metadata.read_text(encoding='utf-8'))
    except (ValueError,OSError) as error:
        raise RuntimeError('Cannot read capture-app metadata') from error
    if state.get('mode')!='capture' or state.get('capture_succeeded') is not True:
        raise RuntimeError('Target MSTX range did not complete; collector exit code is insufficient')
    if state.get('implementation')!=folder.name:
        raise RuntimeError('Capture implementation does not match result directory')
    log=folder/'collect.log'
    if not log.is_file():
        raise RuntimeError('Missing collector log')
    text=log.read_text(encoding='utf-8',errors='replace')
    for pattern in FATAL_PATTERNS:
        if re.search(pattern,text,re.I):
            raise RuntimeError('Collector log contains a failure: '+pattern)
    data=[]
    for path in sorted((folder/'op').rglob('*'),key=str):
        if not path.is_file() or path.stat().st_size==0:
            continue
        if path.suffix.lower()=='.bin':
            data.append(str(path.relative_to(folder)))
        elif path.suffix.lower()=='.csv':
            try:
                with path.open(encoding='utf-8-sig',errors='replace',newline='') as f:
                    reader=csv.reader(f)
                    header=next(reader,[])
                    if len(header)>1 and any(len(row)==len(header) for row in reader):
                        data.append(str(path.relative_to(folder)))
            except (OSError,csv.Error):
                pass
    if not data:
        raise RuntimeError('No nonempty profiler CSV/bin results; source files and logs do not count')
    return dict(implementation=folder.name,target_completed=True,result_files=data,
                note='Collection artifacts exist. Metric coverage/core identifiers still require review.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    a=p.parse_args()
    report=verify_collection(a.input)
    (a.input/'verified.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('COLLECTION VERIFIED:',json.dumps(report),flush=True)


if __name__=='__main__':main()

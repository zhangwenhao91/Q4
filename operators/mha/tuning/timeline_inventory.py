"""Export bounded raw CSV previews and file inventory; do not infer core tail time."""
import argparse
import csv
import io
import json
from pathlib import Path
import re


def inspect_csv(path):
    try:
        with path.open(encoding='utf-8-sig',errors='replace',newline='') as f:
            reader=csv.reader(f)
            headers=next(reader,[])
            rows=[]
            for row in reader:
                if len(row)==len(headers):
                    rows.append(row)
                if len(rows)>256:
                    break
        normalize=lambda s:re.sub(r'[^a-z0-9]','',s.lower())
        identifiers=[i for i,h in enumerate(headers) if normalize(h) in (
            'coreid','physicalcoreid','aicid','aivid','coretype','blockid','subblockid','taskid','name')]
        times=[i for i,h in enumerate(headers) if any(x in h.lower() for x in ('time','cycle','duration'))]
        selected=list(dict.fromkeys(identifiers+times))[:14]
        if not selected:
            selected=list(range(min(10,len(headers))))
        has_core_id=any(normalize(h) in ('coreid','physicalcoreid','aicid','aivid') for h in headers)
        limit=128 if has_core_id else 3
        return dict(headers=headers,sampled_rows=rows[:256],more_than_256=len(rows)>256,
                    has_explicit_core_id_column=has_core_id,
                    preview_headers=[headers[i] for i in selected],
                    preview_rows=[[row[i] for i in selected] for row in rows[:limit]],
                    preview_truncated=len(rows)>limit)
    except (OSError,csv.Error) as error:
        return dict(error=f'{type(error).__name__}: {error}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    a=p.parse_args()
    if not a.input.is_dir():
        p.error('--input must be an existing collection directory')
    files=sorted((path for path in a.input.rglob('*') if path.is_file()),key=str)
    report=dict(root=str(a.input.resolve()),files=[],csv={},
        note='Raw previews only. No timing aggregation. A core ID may be logical; Block ID is not a physical-core count. Missing core/timeline data must not be reconstructed from aggregate metrics.')
    for file in files:
        if file.name in ('inventory.json','inventory.txt'):
            continue
        rel=str(file.relative_to(a.input))
        report['files'].append(dict(path=rel,bytes=file.stat().st_size))
        if file.suffix.lower()=='.csv' and len(report['csv'])<32:
            report['csv'][rel]=inspect_csv(file)
    lines=[report['note'],f"Files: {len(report['files'])}"]
    for entry in report['files'][:100]:
        lines.append(f"FILE {entry['bytes']} {entry['path']}")
    if len(report['files'])>100:
        lines.append('File list truncated in text; full list in inventory.json.')
    for name,data in report['csv'].items():
        lines.append('\n===== CSV '+name+' =====')
        if 'error' in data:
            lines.append(data['error'])
            continue
        lines.append('ALL HEADERS: '+json.dumps(data['headers'],ensure_ascii=False))
        buffer=io.StringIO()
        writer=csv.writer(buffer)
        writer.writerow(data['preview_headers'])
        writer.writerows(data['preview_rows'])
        lines.append(buffer.getvalue())
        if data['preview_truncated']:
            lines.append('Preview truncated; retain original CSV for subsequent analysis.')
    (a.input/'inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    text='\n'.join(lines)+'\n'
    (a.input/'inventory.txt').write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':
    main()

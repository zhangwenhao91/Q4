"""Clear inherited global flags in the root CMake cache before mixed x86/AICPU builds."""
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

KEY=re.compile(r'^CMAKE_(?:(?:C|CXX|ASM)_FLAGS|(?:EXE|SHARED|MODULE)_LINKER_FLAGS):STRING=')


def clean_cache(path):
    path=Path(path)
    if not path.exists():
        print('No existing root CMake cache; sanitized environment will initialize it.');return
    original=path.read_bytes()
    text=original.decode('utf-8')
    changed=[];lines=[]
    for line in text.splitlines(keepends=True):
        if KEY.match(line) and line.rstrip('\r\n').split('=',1)[1]:
            key=line.split('=',1)[0]
            changed.append(key)
            line=key+'=\n'
        lines.append(line)
    if not changed:
        print('Root CMake global flags already empty.');return
    backup=path.with_name(path.name+'.q4-flags-backup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    with backup.open('xb') as f: f.write(original)
    path.write_text(''.join(lines),encoding='utf-8',newline='\n')
    print('Preserved cache:',backup)
    print('Reset inherited global flag entries:',', '.join(changed))
    print('Compiler paths, dependency caches and target-specific options retained.')


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    clean_cache(root/'third_party'/'ops-transformer-q4fa'/'build'/'CMakeCache.txt')

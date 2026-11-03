"""Run one collector, abort its process group on DBI transport failure."""
import argparse
from collections import deque
from pathlib import Path
import os
import signal
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('missing collector command')
    tail = deque(maxlen=40)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors='replace', start_new_session=True)
    failed = False
    try:
        with a.log.open('w') as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                tail.append(line)
                if 'Send one DBI data failed' in line or 'NotifyStopTransLog failed' in line:
                    failed = True
                    log.write('Q4 collector guard: DBI transport failed; stopping this collector process group.\n')
                    break
    finally:
        if failed:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        else:
            proc.wait()
    if failed or proc.returncode:
        print(''.join(tail), end='')
        print(f'Collector failed; full log: {a.log}')
        sys.exit(1)


if __name__ == '__main__':
    main()

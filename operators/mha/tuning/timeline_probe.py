"""Inspect installed single-operator profiling tools; do not launch a kernel."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

METRIC_NAMES = (
    'Default', 'Occupancy', 'TimelineDetail', 'PipeTimeline', 'instrTimeLine',
    'PcSampling', 'PipeUtilization', 'ResourceConflictRatio', 'MemoryDetail',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tool-timeout', type=int, default=30)
    a = parser.parse_args()
    if a.tool_timeout <= 0:
        parser.error('--tool-timeout must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    report = dict(
        q4_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        python=sys.version, python_executable=sys.executable, platform=platform.platform(),
        environment={k: os.environ.get(k) for k in (
            'CONDA_DEFAULT_ENV', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_HOME_PATH', 'ASCEND_NPU_ARCH')},
        packages={}, tools={}, commands=[],
        scope='Read tool help/version and device properties only; no compilation, kernel execution or profiling.')
    for name in ('torch', 'torch-npu', 'tilelang'):
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    try:
        import torch_npu
        properties = torch_npu.npu.get_device_properties(0)
        report['device'] = {name: getattr(properties, name, None) for name in (
            'name', 'cube_core_num', 'vector_core_num', 'total_memory')}
    except Exception as error:
        report['device_query_error'] = f'{type(error).__name__}: {error}'
    for name in ('msprof', 'msopprof'):
        report['tools'][name] = shutil.which(name)
    tool = report['tools']['msprof']
    # Missing tools are a diagnostic outcome, not permission to install or change CANN.
    if tool:
        commands = [
            ('msprof-version', [tool, '--version']),
            ('msprof-help', [tool, '--help']),
            ('msprof-op-help', [tool, 'op', '--help']),
        ]
        for name, command in commands:
            print('PROBE:', subprocess.list2cmdline(command), flush=True)
            try:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, errors='replace', timeout=a.tool_timeout, check=False)
                entry = dict(name=name, argv=command, exit_code=result.returncode)
                output = result.stdout
            except subprocess.TimeoutExpired as error:
                partial = error.stdout or b''
                output = partial.decode(errors='replace') if isinstance(partial, bytes) else partial
                entry = dict(name=name, argv=command, error='timeout', timeout_seconds=a.tool_timeout)
            except OSError as error:
                output = str(error)
                entry = dict(name=name, argv=command, error=f'{type(error).__name__}: {error}')
            filename = name+'.txt'
            (a.output/filename).write_text(output, encoding='utf-8')
            entry['output_file'] = filename
            if name == 'msprof-op-help':
                entry['metric_names_mentioned_in_help'] = [m for m in METRIC_NAMES if m in output]
                entry['note'] = 'Mentioned in help does not establish support on this device.'
            report['commands'].append(entry)
            print('PROBE RESULT:', json.dumps(entry, ensure_ascii=False), flush=True)
    (a.output/'probe.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    combined = json.dumps(report, indent=2, ensure_ascii=False)+'\n'
    for entry in report['commands']:
        combined += '\n===== '+entry['output_file']+' =====\n'
        combined += (a.output/entry['output_file']).read_text(encoding='utf-8')+'\n'
    (a.output/'probe.txt').write_text(combined, encoding='utf-8')
    print(combined, flush=True)
    print('Probe saved:', (a.output/'probe.txt').resolve(), flush=True)


if __name__ == '__main__':
    main()

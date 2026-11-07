"""Summarize the small-N split-K A/B and device profiles."""
import json
import hashlib
from pathlib import Path
from statistics import median
import sys

from summarize_projection import profile_lines
from summarize_small_split_timeline import timeline_lines


def fused_report(root):
    v3 = 'ascendc_vllm_v3'
    k8 = 'tilelang_deferred_k8_batch_prealloc'
    fused = 'tilelang_fused_small'
    labels = (v3, k8, fused, f'{fused}_prealloc')
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'H=5120 shifted Pre; CPU and pairwise checks passed. Preallocated outputs alias across calls.',
             'N | implementation | event us (3 processes) | median us | paired v3/TL | paired preallocated K8/TL']
    for n in (128, 256):
        cases = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        timings = [json.loads((case / 'summary.json').read_text()) for case in cases]
        for label in labels:
            times = [run[label]['event_us'] for run in timings]
            ratios = [median(run[ref]['event_us'] / value for run, value in zip(timings, times))
                      for ref in (v3, k8)]
            lines.append(f'{n} | {label} | {", ".join(f"{value:.3f}" for value in times)} | '
                         f'{median(times):.3f} | {ratios[0]:.4f} | {ratios[1]:.4f}')
        meta = json.loads((cases[0] / 'metadata.json').read_text())['variants'][fused]
        lines.append(f'N={n} fused tiling: {meta["projection_settings"]}; tail: {meta["tail_settings"]}')
    return '\n'.join(lines) + '\n'


def fused_apply_report(root, pack128=False):
    base = 'tilelang_fused_small_prealloc'
    variants = [('U1', base)]
    variants.extend([('Pack128', f'{base}_pack128')] if pack128 else [('U2', f'{base}_u2'), ('U4', f'{base}_u4')])
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'H=5120 shifted Pre; CPU and pairwise checks passed. All variants preallocate outputs and workspace.',
             'N | Apply variant | Pre event us (3 processes) | median us | paired U1/candidate']
    for n in (128, 256):
        timings = [json.loads((root / f'N{n}-rep{i}' / 'summary.json').read_text()) for i in (1, 2, 3)]
        for variant, label in variants:
            times = [run[label]['event_us'] for run in timings]
            ratio = median(run[base]['event_us'] / value for run, value in zip(timings, times))
            lines.append(f'{n} | {variant} | {", ".join(f"{value:.3f}" for value in times)} | '
                         f'{median(times):.3f} | {ratio:.4f}')
            lines.extend(profile_lines(root / f'N{n}-profile-{label}', 'mhc_projection_token_pipeline'))
    return '\n'.join(lines) + '\n'


def allocation_report(root, mode='alloc'):
    labels = ('tilelang_deferred_k8', 'tilelang_deferred_k8_batch')
    pairs = [(name, f'{name}_prealloc') for name in labels]
    tags = ('k8', 'k8-prealloc', 'batch', 'batch-prealloc')
    if mode == 'workspace':
        batch = labels[1]
        pairs = [(batch, f'{batch}_workspace'), (batch, f'{batch}_prealloc'),
                 (f'{batch}_workspace', f'{batch}_prealloc')]
        tags = ('batch', 'batch-workspace', 'batch-prealloc')
    elif mode == 'refactor':
        batch = labels[1]
        pairs = [(f'{batch}_prealloc_reference', f'{batch}_prealloc')]
        tags = ()
    hashes = ('standalone_apply_source_sha256', 'projection_device_source_sha256', 'tail_device_source_sha256')
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             ('CPU and pairwise checks passed. Both preallocated versions reuse their outputs.'
              if mode == 'refactor' else
              'CPU and pairwise checks passed. Workspace-only returns fresh outputs; full preallocation aliases.'),
             'N | reference | candidate | reference event us | candidate event us | paired reference/candidate | '
             'reference wall us | candidate wall us | source match']
    for n in (128, 256):
        cases = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        timings = [json.loads((case / 'summary.json').read_text()) for case in cases]
        metas = [json.loads((case / 'metadata.json').read_text())['variants'] for case in cases]
        for name, reused in pairs:
            fresh, reuse = ([run[label]['event_us'] for run in timings] for label in (name, reused))
            ratios = [a / b for a, b in zip(fresh, reuse)]
            wall = [median(run[label]['wall_us'] for run in timings) for label in (name, reused)]
            same = all(meta[name][key] == meta[reused][key] for meta in metas for key in hashes)
            lines.append(f'{n} | {name} | {reused} | {median(fresh):.3f} | {median(reuse):.3f} | {median(ratios):.4f} | '
                         f'{wall[0]:.3f} | {wall[1]:.3f} | {same}')
            lines.append(f'  Pair ratios: {", ".join(f"{ratio:.4f}" for ratio in ratios)}')
        if mode == 'alloc':
            batch_ratios = [run[labels[0] + '_prealloc']['event_us'] / run[labels[1] + '_prealloc']['event_us']
                            for run in timings]
            lines.append(f'{n} | preallocated sequential/batch K8 event ratio: {median(batch_ratios):.4f}')
    timeline, errors = timeline_lines(root, tags) if tags else ([], [])
    return '\n'.join([*lines, '', *timeline, *errors]) + '\n'


def v3_fused_k8_report(root, hidden=5120, combined=False, large=False, automatic=False, post_only=False):
    ac = 'ascendc_vllm_v3'
    tl = 'tilelang_shifted_pre_auto' if automatic else 'tilelang_projection_coeff_k8_batch_prealloc'
    scope = 'shifted Pre + Post' if combined else 'shifted Pre'
    allocation_note = ('AscendC uses upstream vLLM APIs with per-call allocation; TL Pre preallocates and TL Post uses its allocating API.'
                       if combined else
                       'AscendC uses the upstream vLLM v3 API with per-call allocation; TL reuses outputs and workspace.')
    if post_only:
        ac, tl = 'ascendc_vllm_post', 'tilelang_post'
        scope = 'Post'
        allocation_note = 'Both Post APIs include per-call output allocation and use identical precomputed coefficients and BF16 features.'
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             f'H={hidden} {scope}; CPU and pairwise checks passed.',
             allocation_note,
             'N | run | AC event us | TL event us | AC/TL speedup | AC wall us | TL wall us']
    shapes = (512, 1024, 2048, 4096, 8192) if large else ((128, 256, 384) if combined else (128, 256))
    if automatic or post_only:
        shapes = (128, 256, 384, 512, 1024, 2048, 4096, 8192)
    for n in shapes:
        timings = [json.loads((root / f'N{n}-rep{rep}' / 'summary.json').read_text())
                   for rep in (1, 2, 3)]
        if automatic:
            meta = json.loads((root / f'N{n}-rep1' / 'metadata.json').read_text())['variants'][tl]
            kernel_scope = 'Pre kernels + Post' if combined else 'kernels'
            lines.append(f'{n} | TL dispatch | {meta["dispatch"]["implementation"]} | '
                         f'{meta["expected_kernels_per_call"]} {kernel_scope}')
        for rep, run in enumerate(timings, 1):
            lines.append(f'{n} | rep{rep} | {run[ac]["event_us"]:.3f} | {run[tl]["event_us"]:.3f} | '
                         f'{run[ac]["event_us"] / run[tl]["event_us"]:.4f} | '
                         f'{run[ac]["wall_us"]:.3f} | {run[tl]["wall_us"]:.3f}')
        ac_time = median(run[ac]['event_us'] for run in timings)
        tl_time = median(run[tl]['event_us'] for run in timings)
        paired_ratio = median(run[ac]['event_us'] / run[tl]['event_us'] for run in timings)
        lines.append(f'{n} | median | {ac_time:.3f} | {tl_time:.3f} | {paired_ratio:.4f} | '
                     f'{median(run[ac]["wall_us"] for run in timings):.3f} | '
                     f'{median(run[tl]["wall_us"] for run in timings):.3f}')
    return '\n'.join(lines) + '\n'


def stages_report(root):
    scopes = [('Pre', 'ascendc_vllm_v3', 'tilelang_shifted_pre_auto'),
              ('Post', 'ascendc_vllm_post', 'tilelang_post'),
              ('Pre+Post', 'ascendc_vllm_combined', 'tilelang_auto_combined')]
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'H=5120 N=8192; CPU and same-stage pairwise checks passed.',
             'Each process times all six callbacks on shared inputs, then profiles both combined callbacks.',
             'Standalone Post uses shared reference coefficients; combined Post consumes its own Pre coefficients.',
             'TL Pre preallocates; Post and upstream AscendC use their allocating APIs.',
             'run | stage | AC event us | TL event us | AC/TL | AC wall us | TL wall us']
    timings = []
    for rep in (1, 2, 3):
        case = root / f'N8192-rep{rep}'
        run = json.loads((case / 'summary.json').read_text())
        timings.append(run)
        for stage, ac, tl in scopes:
            lines.append(f'rep{rep} | {stage} | {run[ac]["event_us"]:.3f} | {run[tl]["event_us"]:.3f} | '
                         f'{run[ac]["event_us"] / run[tl]["event_us"]:.4f} | '
                         f'{run[ac]["wall_us"]:.3f} | {run[tl]["wall_us"]:.3f}')
        for side, pre, post, combined in (
                ('AC', scopes[0][1], scopes[1][1], scopes[2][1]),
                ('TL', scopes[0][2], scopes[1][2], scopes[2][2])):
            summed = run[pre]['event_us'] + run[post]['event_us']
            chained = run[combined]['event_us']
            lines.append(f'rep{rep} | {side} isolated sum {summed:.3f} us | '
                         f'combined {chained:.3f} us | reduction {summed - chained:.3f} us')
            lines.extend(profile_lines(case / 'profile-after-timing' / combined))
    for stage, ac, tl in scopes:
        ac_time = median(run[ac]['event_us'] for run in timings)
        tl_time = median(run[tl]['event_us'] for run in timings)
        ratio = median(run[ac]['event_us'] / run[tl]['event_us'] for run in timings)
        lines.append(f'median | {stage} | {ac_time:.3f} | {tl_time:.3f} | {ratio:.4f} | '
                     f'{median(run[ac]["wall_us"] for run in timings):.3f} | '
                     f'{median(run[tl]["wall_us"] for run in timings):.3f}')
    return '\n'.join(lines) + '\n'


def buffer_report(root, cast_reuse=False):
    ac = 'ascendc_vllm_v3'
    base = 'tilelang_projection_coeff_k8_batch_prealloc'
    variants = [('current', base), ('I1C1', base + '_i1c1'), ('I2C1', base + '_i2c1')]
    if cast_reuse:
        base += '_i2c1'
        variants = [('I2C1', base), ('I2C1-reuse', base + '_reuse'),
                    ('I2C2-reuse', 'tilelang_projection_coeff_k8_batch_prealloc_i2c2_reuse')]
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'H=5120 shifted Pre; CPU and pairwise checks passed.' if cast_reuse else
             'H=5120 shifted Pre + Post; CPU and pairwise checks passed.',
             'I=input slots; C=FP32 NZ cast slots. Reuse candidates share dead input storage with the partial staging buffers.'
             if cast_reuse else 'I=input slots; C=FP32 NZ cast slots. I1C1/I2C1 isolates input buffering at C=1.',
             'TL Pre fully preallocates; AscendC allocates through its upstream API.' if cast_reuse else
             'TL Pre fully preallocates; TL Post and upstream AscendC allocate through their APIs.',
             'N | variant | input slots | cast slots | event us (3 processes) | median us | paired AC/TL | paired current/TL']
    for n in (512, 1024, 2048, 4096, 8192):
        cases = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        timings = [json.loads((case / 'summary.json').read_text()) for case in cases]
        metadata = json.loads((cases[0] / 'metadata.json').read_text())['variants']
        ac_times = [run[ac]['event_us'] for run in timings]
        lines.append(f'{n} | AC | - | - | {", ".join(f"{v:.3f}" for v in ac_times)} | {median(ac_times):.3f} | 1.0000 | -')
        for variant, label in variants:
            times = [run[label]['event_us'] for run in timings]
            config = metadata[label]['projection_config']
            ratios = [median(run[reference]['event_us'] / run[label]['event_us'] for run in timings)
                      for reference in (ac, base)]
            lines.append(f'{n} | {variant} | {config["input_stages"]} | {config["cast_stages"]} | '
                         f'{", ".join(f"{v:.3f}" for v in times)} | {median(times):.3f} | '
                         f'{ratios[0]:.4f} | {ratios[1]:.4f}')
            lines.extend(profile_lines(root / f'N{n}-profile-{label}', 'mhc_projection_coeff_k8_kernel'))
        ratio = median(run[variants[1][1]]['event_us'] / run[variants[2][1]]['event_us'] for run in timings)
        comparison = 'I2C1-reuse/I2C2-reuse' if cast_reuse else 'I1C1/I2C1'
        lines.append(f'{n} | paired {comparison} speedup | {ratio:.4f}')
    return '\n'.join(lines) + '\n'


def cube_report(root, full28=False):
    ac = 'ascendc_vllm_v3'
    base = 'tilelang_projection_coeff_k8_batch_prealloc_i2c1'
    labels = (base, base + '_c24')
    shapes = (512, 1024, 2048, 4096, 8192)
    if full28:
        base += '_c24'
        labels = (base, 'tilelang_projection_coeff_k4_batch_prealloc_i2c1_c28')
        shapes = (1024, 2048, 4096, 8192)
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'H=5120 shifted Pre; CPU and pairwise checks passed.',
             'Both TL variants use two input slots, one cast slot and preallocated outputs/workspace.',
             'The extra tile count is per K split. Tail groups repeat only their own last tile before the barrier.',
             'K4 changes the K reduction grouping; the original CPU and pairwise precision gates apply.' if full28 else
             'Both variants use K8.',
             'N | variant | K splits | Cube cores | rounds | extra tiles per split | event us (3 processes) | median us | paired AC/TL | paired baseline/TL']
    for n in shapes:
        cases = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        timings = [json.loads((case / 'summary.json').read_text()) for case in cases]
        metadata = json.loads((cases[0] / 'metadata.json').read_text())['variants']
        ac_times = [run[ac]['event_us'] for run in timings]
        lines.append(f'{n} | AC | - | - | - | - | {", ".join(f"{v:.3f}" for v in ac_times)} | {median(ac_times):.3f} | 1.0000 | -')
        for label in labels:
            config = metadata[label]['projection_config']
            groups = config['cores_per_split']
            rounds = (n // 128 + groups - 1) // groups
            extra = rounds * groups - n // 128
            times = [run[label]['event_us'] for run in timings]
            ratios = [median(run[reference]['event_us'] / run[label]['event_us'] for run in timings)
                      for reference in (ac, base)]
            variant = 'baseline' if label == base else 'more-Cube'
            lines.append(f'{n} | {variant} | {config["num_k_splits"]} | {config["num_aic_cores"]} | {rounds} | {extra} | '
                         f'{", ".join(f"{v:.3f}" for v in times)} | {median(times):.3f} | '
                         f'{ratios[0]:.4f} | {ratios[1]:.4f}')
            lines.extend(profile_lines(root / f'N{n}-profile-{label}', 'mhc_projection_coeff_k8_kernel'))
    return '\n'.join(lines) + '\n'


def three_vs_two_report(root, small=False):
    base = 'tilelang_deferred_k8_batch_prealloc'
    variants = [('three-kernel K8', base),
                ('two-kernel K8/24', 'tilelang_projection_coeff_k8_batch_prealloc_i2c1_c24'),
                ('two-kernel K4/28', 'tilelang_projection_coeff_k4_batch_prealloc_i2c1_c28')]
    if small:
        variants = [('three-kernel K8', base),
                    ('two-kernel K8 small', 'tilelang_projection_coeff_k8_batch_prealloc'),
                    ('two-kernel K8 balanced', 'tilelang_projection_coeff_k8_batch_prealloc_i2c1_c24'),
                    ('two-kernel K4 balanced', 'tilelang_projection_coeff_k4_batch_prealloc_i2c1_c28')]
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'H=5120 shifted Pre; CPU and pairwise checks passed. All variants preallocate outputs/workspace.',
             'N | variant | Cube cores | K splits | event us (3 processes) | median us | paired three-kernel/TL']
    for n in ((128, 256, 384, 512) if small else (1024, 2048, 4096, 8192)):
        cases = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        timings = [json.loads((case / 'summary.json').read_text()) for case in cases]
        metadata = json.loads((cases[0] / 'metadata.json').read_text())['variants']
        for variant, label in variants:
            times = [run[label]['event_us'] for run in timings]
            config = metadata[label]['projection_config']
            ratio = median(run[base]['event_us'] / run[label]['event_us'] for run in timings)
            lines.append(f'{n} | {variant} | {config["num_aic_cores"]} | {config["num_k_splits"]} | '
                         f'{", ".join(f"{v:.3f}" for v in times)} | {median(times):.3f} | {ratio:.4f}')
            lines.extend(profile_lines(root / f'N{n}-profile-{label}', 'mhc_projection'))
    return '\n'.join(lines) + '\n'


def auto_dispatch_report(root):
    candidate = 'tilelang_shifted_pre_auto'
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'H=5120 shifted Pre; CPU and pairwise checks passed. All outputs/workspace are preallocated.',
             'N | selected implementation | kernel count | reference us | auto us | paired reference/auto | all device sources equal']
    for n in (128, 256, 384, 512, 1024):
        reference = 'tilelang_projection_coeff_k8_batch_prealloc' if n < 512 else 'tilelang_deferred_k8_batch_prealloc'
        kernels = ('apply', 'projection_coeff') if n < 512 else ('apply', 'projection', 'tail')
        cases = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        timings = [json.loads((case / 'summary.json').read_text()) for case in cases]
        meta = json.loads((cases[0] / 'metadata.json').read_text())['variants'][candidate]
        matches = all((case / f'{reference}_{kernel}.cpp').read_bytes() ==
                      (case / f'{candidate}_{kernel}.cpp').read_bytes()
                      for case in cases for kernel in kernels)
        old = median(run[reference]['event_us'] for run in timings)
        new = median(run[candidate]['event_us'] for run in timings)
        ratio = median(run[reference]['event_us'] / run[candidate]['event_us'] for run in timings)
        lines.append(f'{n} | {meta["dispatch"]["implementation"]} | {meta["expected_kernels_per_call"]} | '
                     f'{old:.3f} | {new:.3f} | {ratio:.4f} | {matches}')
    return '\n'.join(lines) + '\n'


def fused_k8_refactor_report(root):
    candidate = 'tilelang_projection_coeff_k8_batch_prealloc'
    reference = candidate + '_reference'
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'H=5120 shifted Pre; CPU and pairwise checks passed. Both variants preallocate outputs and workspace.',
             'N | run | original us | standalone us | original/standalone | device source match (apply, fused)']
    for n in (128, 256):
        ratios = []
        for rep in (1, 2, 3):
            case = root / f'N{n}-rep{rep}'
            timings = json.loads((case / 'summary.json').read_text())
            original = timings[reference]['event_us']
            standalone = timings[candidate]['event_us']
            ratio = original / standalone
            ratios.append(ratio)
            matches = []
            for kernel in ('apply', 'projection_coeff'):
                hashes = [hashlib.sha256((case / f'{label}_{kernel}.cpp').read_bytes()).hexdigest()
                          for label in (reference, candidate)]
                matches.append(hashes[0] == hashes[1])
            lines.append(f'{n} | rep{rep} | {original:.3f} | {standalone:.3f} | {ratio:.4f} | {matches}')
        lines.append(f'{n} | median paired speedup | {median(ratios):.4f}')
    return '\n'.join(lines) + '\n'


def projection_coeff_report(root, k8=False):
    base = 'tilelang_deferred_k8_batch_prealloc'
    variants = ((base, 'mhc_projection_split_partials'),
                ('tilelang_projection_coeff_prealloc', 'mhc_projection_coeff_kernel'),
                ('tilelang_projection_coeff_m32_prealloc', 'mhc_projection_coeff_kernel'))
    if k8:
        variants = ((base, 'mhc_projection_split_partials'),
                    ('tilelang_projection_coeff_k8_prealloc', 'mhc_projection_coeff_k8_kernel'),
                    ('tilelang_projection_coeff_k8_batch_prealloc', 'mhc_projection_coeff_k8_kernel'))
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'H=5120 shifted Pre; CPU and pairwise checks passed. Both variants preallocate outputs.',
             'N | implementation | Cube cores | K splits | Pre event us (3 processes) | median us | paired K8/TL']
    for n in (128, 256):
        cases = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        timings = [json.loads((case / 'summary.json').read_text()) for case in cases]
        metadata = json.loads((cases[0] / 'metadata.json').read_text())['variants']
        for label, kernel in variants:
            times = [run[label]['event_us'] for run in timings]
            speedup = median(run[base]['event_us'] / value for run, value in zip(timings, times))
            config = metadata[label]['projection_config']
            lines.append(f'{n} | {label} | {config["num_aic_cores"]} | {config["num_k_splits"]} | '
                         f'{", ".join(f"{value:.3f}" for value in times)} | {median(times):.3f} | {speedup:.4f}')
            lines.extend(profile_lines(root / f'N{n}-profile-{label}', kernel))
    return '\n'.join(lines) + '\n'


def report(root, batch=False):
    base = 'tilelang_all_token_l2_default'
    lines = [f'Run: {root}', f'Q4 revision: {(root / "revision.txt").read_text().strip()}',
             'N | implementation | Cube cores | Pre event us (3 processes) | median us | paired baseline/TL']
    if batch:
        lines[-1] += ' | paired sequential K8/TL'
    for n in ((128, 256) if batch else (128, 256, 512)):
        runs = [root / f'N{n}-rep{rep}' for rep in (1, 2, 3)]
        metadata = json.loads((runs[0] / 'metadata.json').read_text())['variants']
        timings = [json.loads((run / 'summary.json').read_text()) for run in runs]
        for label in metadata:
            times = [run[label]['event_us'] for run in timings]
            ratio = median(run[base]['event_us'] / value for run, value in zip(timings, times))
            cores = metadata[label]['projection_config']['num_aic_cores']
            line = (f'{n} | {label} | {cores} | {", ".join(f"{v:.3f}" for v in times)} | '
                    f'{median(times):.3f} | {ratio:.4f}')
            if batch:
                k8_ratio = median(run['tilelang_deferred_k8']['event_us'] / value
                                  for run, value in zip(timings, times))
                line += f' | {k8_ratio:.4f}'
            lines.append(line)
        profiles = [('base', 'mhc_projection_token_pipeline'), ('k8', 'mhc_projection_split_partials')]
        if batch:
            profiles.append(('batch', 'mhc_projection_split_partials'))
        for tag, kernel in profiles:
            lines.extend(profile_lines(root / f'N{n}-profile-{tag}', kernel))
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    root = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else 'scan'
    if mode == 'fused':
        summary = fused_report(root)
    elif mode == 'v3-stages':
        summary = stages_report(root)
    elif mode == 'v3-post':
        summary = v3_fused_k8_report(root, post_only=True)
    elif mode == 'auto-dispatch':
        summary = auto_dispatch_report(root)
    elif mode in ('v3-auto-pre', 'v3-auto-combined'):
        summary = v3_fused_k8_report(root, automatic=True, combined=mode.endswith('-combined'))
    elif mode in ('three-vs-two-large', 'three-vs-two-small'):
        summary = three_vs_two_report(root, small=mode.endswith('-small'))
    elif mode in ('v3-fused-k8-combined-buffers', 'v3-fused-k8-cast-reuse'):
        summary = buffer_report(root, cast_reuse=mode.endswith('-cast-reuse'))
    elif mode in ('v3-fused-k8-cube24', 'v3-fused-k4-cube28'):
        summary = cube_report(root, full28=mode == 'v3-fused-k4-cube28')
    elif mode in ('v3-fused-k8', 'v3-fused-k8-h4096', 'v3-fused-k8-combined', 'v3-fused-k8-combined-large'):
        summary = v3_fused_k8_report(root, hidden=4096 if mode.endswith('-h4096') else 5120,
                                   combined='-combined' in mode, large=mode.endswith('-large'))
    elif mode == 'fused-k8-refactor':
        summary = fused_k8_refactor_report(root)
    elif mode in ('fused-apply', 'fused-pack128'):
        summary = fused_apply_report(root, pack128=mode == 'fused-pack128')
    elif mode in ('fused-projection-coeff', 'fused-projection-coeff-k8'):
        summary = projection_coeff_report(root, k8=mode.endswith('-k8'))
    elif mode in ('alloc', 'workspace', 'refactor'):
        summary = allocation_report(root, mode=mode)
    else:
        summary = report(root, batch=mode == 'batch')
    (root / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary, end='')

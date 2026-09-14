"""Machine state capture, so every measurement self-describes.

The reviewer asked runtime comparisons to identify hardware settings. The reason this is
a module rather than a checklist is `rate_live_on_final.json`: it was labelled at capture
as blend+ROI ON, actually measured a stale OFF launch, and the mistake was only caught
later by reading back the node's parameters. A number that carries its own provenance
cannot drift from its label.

The specific trap this exists to close: `nvpmodel` persists across reboots but
`jetson_clocks` does NOT. A board can sit in MAXN with clocks free-running under DVFS,
which reads as "maximum performance mode" in a paper and is a different measurement.
jetson_clocks pins min == max rather than renaming the governor, so min == max is the
test -- reading the governor name would report `schedutil` either way and miss it.

Usage:
    from envinfo import capture
    report = {'label': ..., 'env': capture(engines=[backbone, residual]), ...}

or standalone:
    python3 tools/diagnostics/envinfo.py --out docs/demo/benchmarks/env_r31.json
"""
import hashlib
import json
import os
import re
import subprocess


def _read(path, default=''):
    # Broad except on purpose: some tegra sysfs nodes exist but fault on read (a thermal
    # zone with no backing sensor raises TypeError out of the codec layer, not OSError).
    # A capture helper must never be the thing that kills a measurement run.
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return default


def _run(cmd, default=''):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return default


def clocks():
    """DVFS state. `jetson_clocks_applied` is the one that resets every boot."""
    gpu = '/sys/class/devfreq/17000000.gpu'
    cpu = '/sys/devices/system/cpu/cpu0/cpufreq'
    st = {
        'cpu_governor': _read(f'{cpu}/scaling_governor'),
        'cpu_min_khz': _read(f'{cpu}/scaling_min_freq'),
        'cpu_max_khz': _read(f'{cpu}/scaling_max_freq'),
        'gpu_governor': _read(f'{gpu}/governor'),
        'gpu_min_hz': _read(f'{gpu}/min_freq'),
        'gpu_max_hz': _read(f'{gpu}/max_freq'),
        'gpu_cur_hz': _read(f'{gpu}/cur_freq'),
        'nvpmodel': _run('nvpmodel -q 2>/dev/null | head -2').replace('\n', ' '),
    }
    st['cpu_locked'] = bool(st['cpu_min_khz'] and st['cpu_min_khz'] == st['cpu_max_khz'])
    st['gpu_locked'] = bool(st['gpu_min_hz'] and st['gpu_min_hz'] == st['gpu_max_hz'])
    st['jetson_clocks_applied'] = st['cpu_locked'] and st['gpu_locked']
    return st


def versions():
    trt = _run("dpkg -l 2>/dev/null | awk '/libnvinfer-bin/{print $3; exit}'")
    cud = _run("dpkg -l 2>/dev/null | awk '/libcudnn9-cuda/{print $3; exit}'")
    nvcc = _run('nvcc --version 2>/dev/null | tail -2 | head -1')
    m = re.search(r'release ([\d.]+), V([\d.]+)', nvcc)
    return {
        'l4t': _read('/etc/nv_tegra_release').splitlines()[:1],
        'tensorrt': trt,
        'cuda': m.group(2) if m else '',
        'cudnn': cud,
        'jetpack_metapackage': _run("dpkg -l 2>/dev/null | awk '/nvidia-jetpack /{print $3}'"),
    }


def thermal():
    """Long captures heat the board; a throttled tail is a real effect on a 15 min run."""
    out = {}
    base = '/sys/class/thermal'
    try:
        for z in sorted(os.listdir(base)):
            if not z.startswith('thermal_zone'):
                continue
            t = _read(f'{base}/{z}/type')
            v = _read(f'{base}/{z}/temp')
            if t and v and v.isdigit() and int(v) > 0:
                out[t] = round(int(v) / 1000.0, 1)
    except Exception:
        pass
    return out


def _sha(path, nbytes=1 << 20):
    """Hash the first MB -- enough to tell two engines apart, cheap on a 9 MB file."""
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read(nbytes)).hexdigest()[:16]
    except Exception:
        return ''


def engines(paths):
    out = {}
    for p in paths or []:
        if not p:
            continue
        out[os.path.basename(p)] = {
            'path': os.path.abspath(p),
            'exists': os.path.exists(p),
            'bytes': os.path.getsize(p) if os.path.exists(p) else 0,
            'sha256_1mb': _sha(p),
        }
    return out


def git():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {
        'commit': _run(f'git -C {repo} rev-parse HEAD'),
        'dirty': bool(_run(f'git -C {repo} status --porcelain')),
        'branch': _run(f'git -C {repo} rev-parse --abbrev-ref HEAD'),
    }


def capture(engines_list=None, note=''):
    return {
        'note': note,
        'captured_utc': _run('date -u +%Y-%m-%dT%H:%M:%SZ'),
        'uptime': _run('uptime -p'),
        'versions': versions(),
        'clocks': clocks(),
        'thermal_c': thermal(),
        'engines': engines(engines_list),
        'git': git(),
    }


def warn_if_unlocked(env):
    """Print a loud warning rather than silently recording a non-comparable number."""
    if not env['clocks']['jetson_clocks_applied']:
        print('\n!! jetson_clocks is NOT applied -- clocks are free-running under DVFS.')
        print('!! Run `sudo jetson_clocks` before any timing capture, or this number')
        print('!! will not be comparable with the others. Recorded as such either way.\n')
        return False
    return True


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    ap.add_argument('--engine', action='append', default=[])
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    env = capture(a.engine, a.note)
    print(json.dumps(env, indent=1))
    warn_if_unlocked(env)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(env, f, indent=1)
        print(f'wrote {a.out}')

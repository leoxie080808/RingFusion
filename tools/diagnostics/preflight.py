#!/usr/bin/env python3
"""Check the Orin is quiet before a timing measurement. Run this FIRST, every time.

A timing run is only as trustworthy as the machine under it, and this project has already
been burned once: a background GPU job inflated a DEPTHOR-Small measurement to 166 ms against
the clean 79.4 ms -- 2x -- and it was only caught because a network-only figure came out
larger than an end-to-end one, which is impossible. Nothing in the numbers themselves said
"contaminated".

Checks, in the order they are worth caring about:
  1. stray compute processes (training, benchmarks, other ROS graphs)
  2. GPU load right now
  3. CPU load average vs core count
  4. thermal throttling
  5. clocks pinned (jetson_clocks) and power mode -- these need sudo, so they are REPORTED
     for you to eyeball rather than auto-checked

Exit code 0 = clear to measure, 1 = something is using the machine.
"""
import glob
import os
import re
import subprocess
import sys
import time

# Substrings that mean "this process would corrupt a timing run". Deliberately narrow: this
# script's own name and the perception stack under test must NOT match.
BUSY_PAT = re.compile(
    r'(train|distill|zjul5|depthor|baselines|time_pipeline|sigma_|ceiling_|teacher_vs|'
    r'colcon|pytest|ffmpeg|export_onnx|build_engine|trtexec)', re.I)
# The perception stack itself is expected to be running during rate_live -- never flag it.
ALLOW_PAT = re.compile(r'(rate_live|preflight|perception|tof_driver|camera|ros2 launch|'
                       r'tape_capture|moving_ab|stream_test)', re.I)

GPU_LOAD = '/sys/devices/platform/gpu.0/load'          # per-mille, 0-1000. No sudo needed.


def ok(msg):
    print(f'  \033[32mOK\033[0m   {msg}')


def bad(msg):
    print(f'  \033[31mBUSY\033[0m {msg}')


def info(msg):
    print(f'  --   {msg}')


def check_procs():
    print('\n1. stray compute processes')
    try:
        out = subprocess.run(['ps', '-eo', 'pid,pcpu,pmem,etimes,args'],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception as e:                                    # noqa: BLE001
        info(f'could not list processes ({e})')
        return True
    hits = []
    me = os.getpid()
    for line in out.splitlines()[1:]:
        f = line.split(None, 4)
        if len(f) < 5:
            continue
        pid, pcpu, _, _, args = f
        if int(pid) == me or ALLOW_PAT.search(args):
            continue
        if BUSY_PAT.search(args) and float(pcpu) > 5.0:
            hits.append((pid, pcpu, args[:90]))
    if hits:
        for pid, pcpu, args in hits:
            bad(f'pid {pid} at {pcpu}% CPU: {args}')
        print('     -> stop these before measuring:  kill <pid>')
        return False
    ok('no training/benchmark/build processes above 5% CPU')
    return True


def check_gpu():
    print('\n2. GPU load')
    if not os.path.exists(GPU_LOAD):
        info(f'{GPU_LOAD} not present; check manually with: sudo tegrastats')
        return True
    # Sample over a second -- a single read can land in a gap between kernels.
    s = []
    for _ in range(10):
        try:
            s.append(int(open(GPU_LOAD).read().strip()) / 10.0)
        except Exception:                                     # noqa: BLE001
            break
        time.sleep(0.1)
    if not s:
        info('could not read GPU load')
        return True
    peak, mean = max(s), sum(s) / len(s)
    if peak > 10.0:
        bad(f'GPU busy: mean {mean:.1f}%, peak {peak:.1f}% over 1 s')
        print('     -> find it with:  sudo fuser -v /dev/nvhost-gpu')
        return False
    ok(f'GPU idle (mean {mean:.1f}%, peak {peak:.1f}%)')
    return True


def check_cpu():
    print('\n3. CPU load')
    n = os.cpu_count() or 1
    l1, l5, _ = os.getloadavg()
    if l1 > 0.5 * n:
        bad(f'1-min load {l1:.2f} on {n} cores ({100*l1/n:.0f}% busy)')
        print('     -> check with:  top -b -n1 | head -15')
        return False
    ok(f'load {l1:.2f} (1 min) / {l5:.2f} (5 min) on {n} cores')
    return True


def check_thermal():
    print('\n4. thermal')
    hot = []
    for z in sorted(glob.glob('/sys/devices/virtual/thermal/thermal_zone*')):
        try:
            t = int(open(os.path.join(z, 'temp')).read().strip()) / 1000.0
            name = open(os.path.join(z, 'type')).read().strip()
        except Exception:                                     # noqa: BLE001
            continue
        if t > 80.0:
            hot.append((name, t))
    if hot:
        for name, t in hot:
            bad(f'{name} at {t:.1f} C -- may be throttling, which fakes a slow pipeline')
        return False
    ok('all thermal zones below 80 C')
    return True


def report_clocks():
    """Needs sudo, so REPORT rather than gate -- a preflight that demands a password is a
    preflight people skip."""
    print('\n5. clocks + power mode (needs sudo -- verify by eye)')
    print('     sudo nvpmodel -q          # want: MAXN')
    print('     sudo jetson_clocks --show # want: every clock at its max')
    print('     If unsure, just run:  sudo jetson_clocks')


def main():
    print('=' * 62)
    print('PREFLIGHT -- is this machine quiet enough to time something?')
    print('=' * 62)
    results = [check_procs(), check_gpu(), check_cpu(), check_thermal()]
    report_clocks()
    print('\n' + '=' * 62)
    if all(results):
        print('\033[32mCLEAR TO MEASURE\033[0m -- nothing else is using the machine.')
        print('=' * 62)
        return 0
    print('\033[31mDO NOT MEASURE YET\033[0m -- something above is using the machine.')
    print('A contaminated timing run looks exactly like a slow pipeline.')
    print('=' * 62)
    return 1


if __name__ == '__main__':
    sys.exit(main())

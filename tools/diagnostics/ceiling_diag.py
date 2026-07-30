#!/usr/bin/env python3
"""What is the deepest depth our affine form can even EXPRESS? No ground truth needed.

D = 1/(a*disp + b). As disparity goes to zero (far away), D asymptotes to 1/b. So b sets a
HARD CEILING on the deepest value the pipeline can output, whatever the image shows -- and b
is fitted entirely from ToF anchors, which only cover near range.

Found on ZJU-L5 (dense independent GT, 2026-07-29): median ceiling 2.04 m against ground
truth reaching 20 m. That single mechanism produces BOTH failures measured there:
  * depth under-reads by ~12 m at 10-20 m true depth (56% of all squared error)
  * sigma is 150x too small there and DECREASES with range, because
    Var[D] = D^4 * j^T Cov j and D is itself capped at 1/b -- most confident where most wrong
Their ToF only reaches ~1.59 m, so the ceiling may be pathologically low on that rig.

This script asks the same question of OUR sensor, which reads to 6.5 m. It needs no GT: the
ceiling falls out of the fit. Compare `ceiling` against the ToF's own furthest reading and
against the p99 of emitted depth:
  * ceiling >> anchor_max  -> the far field is merely extrapolated, not structurally capped
  * ceiling ~= anchor_max  -> we have the same cap on the robot and never measured it
  * emitted p99 pinned at the ceiling -> pixels are actively saturating against it
"""
import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anchoring_bridge import calib_from_yaml                        # noqa: E402
from ringfusion_perception import anchoring as anc                  # noqa: E402
from baselines import project, split_zones, MIN_RANGE, MAX_RANGE    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-dir', required=True)
    ap.add_argument('--tof-dir', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--backbone-engine', required=True)
    ap.add_argument('--protocol', default='insample',
                    choices=['insample', 'random', 'center'],
                    help='insample = all valid zones anchor the fit, as deployed')
    ap.add_argument('--limit', type=int, default=300)
    ap.add_argument('--out', default='')
    # Ridge on b (anchoring.solve_scale_shift). Chosen as 0.003 on the ZJU-L5 TRAIN split,
    # but that sensor's anchors run to a median 1.59 m against our 0.49 m, so it must be
    # re-validated here before any deployed default changes. Error is binned by TRUE anchor
    # depth because that is the axis the ceiling acts on.
    ap.add_argument('--sweep-bprior', type=float, nargs='*', default=[])
    a = ap.parse_args()

    import cv2
    from ringfusion_perception.backbone import TensorRTBackbone

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a.tof_dir)
                   if f.endswith('.npz'))[:a.limit]
    first = cv2.imread(os.path.join(a.rgb_dir, stems[0] + '.png'))
    h, w = first.shape[:2]
    calib = calib_from_yaml(a.calib, train_size=(h, w))
    bb = TensorRTBackbone(a.backbone_engine)

    ceil, amax, p99, pmax, satur, bs = [], [], [], [], [], []
    sweep = {bp: {'e': [], 'z': [], 'c': []} for bp in a.sweep_bprior}
    for n, stem in enumerate(stems):
        img = cv2.imread(os.path.join(a.rgb_dir, stem + '.png'))
        if img is None:
            continue
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        d = np.load(os.path.join(a.tof_dir, stem + '.npz'))['dist_m'].astype(np.float32)
        valid = np.isfinite(d) & (d >= MIN_RANGE) & (d <= MAX_RANGE)
        am_z, _ = split_zones(valid, a.protocol, np.random.default_rng(n))
        if am_z.sum() < 32:
            continue
        disp = bb.infer(rgb)
        u, v, z, _ = project(d, am_z, calib, h, w)
        if u.size < 32:
            continue
        fit = anc.solve_robust(disp[v, u].astype(np.float64), 1.0 / z,
                              np.ones_like(z), iters=1)
        if fit is None:
            continue
        a_, b_ = fit
        D = anc.to_metric_depth(disp, a_, b_)
        c = 1.0 / max(b_, 1e-9) if b_ > 0 else np.inf
        ceil.append(c); bs.append(b_); amax.append(float(z.max()))
        p99.append(float(np.percentile(D, 99))); pmax.append(float(D.max()))
        # "saturating" = within 5% of the ceiling, i.e. actively pinned against it
        satur.append(float(np.mean(D > 0.95 * c)) if np.isfinite(c) else 0.0)
        for bp in a.sweep_bprior:
            fp = anc.solve_robust(disp[v, u].astype(np.float64), 1.0 / z,
                                  np.ones_like(z), iters=1, b_prior=bp)
            if fp is None:
                continue
            Dp = anc.to_metric_depth(disp, *fp)
            sweep[bp]['e'].append(Dp[v, u] - z)
            sweep[bp]['z'].append(z)
            sweep[bp]['c'].append(1.0 / max(fp[1], 1e-9) if fp[1] > 0 else np.inf)
        # baseline (b_prior = 0) arm, for the same frames
        if a.sweep_bprior:
            sweep.setdefault(0.0, {'e': [], 'z': [], 'c': []})
            sweep[0.0]['e'].append(D[v, u] - z)
            sweep[0.0]['z'].append(z)
            sweep[0.0]['c'].append(c)
        if (n + 1) % 100 == 0:
            print(f'  {n+1}/{len(stems)}', flush=True)

    ceil = np.array(ceil); amax = np.array(amax)
    p99 = np.array(p99); pmax = np.array(pmax); satur = np.array(satur)
    bneg = int(np.sum(np.array(bs) <= 0))

    def q(x, lbl, unit='m'):
        print(f'{lbl:<34} p10 {np.percentile(x,10):7.2f}  median {np.median(x):7.2f}'
              f'  p90 {np.percentile(x,90):7.2f} {unit}')

    print(f'\n{ceil.size} frames, protocol={a.protocol}, ToF gate {MIN_RANGE}-{MAX_RANGE} m')
    print(f'frames with b <= 0 (no finite ceiling): {bneg}')
    q(ceil, 'ceiling 1/b')
    q(amax, 'furthest ToF anchor')
    q(ceil / np.maximum(amax, 1e-9), 'ceiling / anchor_max', 'x')
    q(p99, 'emitted depth p99')
    q(pmax, 'emitted depth max')
    q(100 * satur, 'pixels within 5% of ceiling', '%')

    print(f'\nZJU-L5 for reference: ceiling median 2.04 m, furthest anchor 1.59 m '
          f'(ratio 1.28x), GT to 20 m')
    r = float(np.median(ceil / np.maximum(amax, 1e-9)))
    print(f'ours: ratio {r:.2f}x')
    if r < 2.0:
        print('  ** ceiling sits close to the ToF range -> same structural cap as ZJU-L5,\n'
              '     and it has never been measured on the robot **')
    else:
        print('  -> ceiling is well beyond the ToF range; the far field is extrapolated\n'
              '     rather than structurally capped. Still unvalidated (no far GT).')

    if a.sweep_bprior:
        BINS = ((0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 6.5))
        print(f'\n b_prior sweep -- median |error| by TRUE anchor depth')
        hdr = f'{"b_prior":>9}{"ceiling":>10}' + ''.join(
            f'{f"{lo}-{hi}m":>11}' for lo, hi in BINS) + f'{"MAE":>10}{"medAE":>9}{"bias":>9}'
        print(hdr); print('-' * len(hdr))
        for bp in sorted(sweep, key=lambda x: (x != 0.0, x)):
            d = sweep[bp]
            if not d['e']:
                continue
            E = np.concatenate(d['e']); Z = np.concatenate(d['z'])
            C = np.array([x for x in d['c'] if np.isfinite(x)])
            row = f'{bp:>9g}{np.median(C) if C.size else float("inf"):>9.2f}m'
            for lo, hi in BINS:
                sel = (Z >= lo) & (Z < hi)
                row += (f'{np.median(np.abs(E[sel])):>10.3f}m' if sel.sum() > 30
                        else f'{"-":>11}')
            print(row + f'{np.abs(E).mean():>9.3f}m{np.median(np.abs(E)):>8.3f}m'
                        f'{E.mean():>+8.3f}m')

    if a.out:
        import json
        json.dump({'n': int(ceil.size), 'protocol': a.protocol, 'b_nonpos': bneg,
                   'ceiling_median': float(np.median(ceil)),
                   'ceiling_p10': float(np.percentile(ceil, 10)),
                   'ceiling_p90': float(np.percentile(ceil, 90)),
                   'anchor_max_median': float(np.median(amax)),
                   'ratio_median': r,
                   'emitted_p99_median': float(np.median(p99)),
                   'saturating_pct_median': float(np.median(100 * satur))},
                  open(a.out, 'w'), indent=1)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()

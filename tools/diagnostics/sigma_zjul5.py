#!/usr/bin/env python3
"""Does sigma already flag the far-field under-read, or do we have a real hole?

On ZJU-L5 with dense independent GT, 56% of the squared error comes from 885 pixels (0.3%
of the data) at true depth 10-20 m, where the closed-form path under-reads by ~12 m. The
mechanism: D = 1/(a*disp + b) asymptotes to 1/b as disparity goes to zero, and b is fitted
entirely from anchors whose furthest reading is a median of 1.75 m. So the model has a hard
ceiling it cannot express past.

Before trying to change the depth, check whether the UNCERTAINTY already says so. The
analytic variance carries a D^4 factor (Var[D] = D^4 * j^T Cov(a,b) j, j = (disp, 1)), so
far pixels should already be flagged. If sigma tracks that 12 m error, the honest answer is
to emit the depth, mark it low-confidence, and let consumers discount it -- which converts a
failure into a correctly-reported limitation. It is also a direct test of the
corr(sigma,|error|) = 0.943 figure, measured on our own sensor in a regime this one never
covered.

ONLY the ANALYTIC variance is used. Network B's learned variance head is out of domain on an
8x8 grid (its depth output scores Rel 1.819 here), so its sigma would be meaningless. The
analytic term has no learned parameters and transfers with the fit.

The decisive number is TAIL CAPTURE: rank pixels by sigma, and ask what share of the total
squared error sits in the top few percent. If sigma is informative, a consumer can discard a
tiny fraction of pixels and remove most of the error. If it is flat, sigma cannot be used as
a filter and the far field is a genuine hole.
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_perception'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ringfusion_perception import anchoring as anc                  # noqa: E402
from ringfusion_perception.residual import MAX_DEPTH_M              # noqa: E402
from zjul5_eval import zone_anchors, coverage_mask, _Teacher, GT_MIN, GT_MAX   # noqa: E402

IDEAL_1S, IDEAL_2S = 0.6827, 0.9545
BANDS = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--teacher', default='depth-anything/Depth-Anything-V2-Large-hf')
    ap.add_argument('--split', default='train', choices=['train', 'test', 'all'])
    ap.add_argument('--limit', type=int, default=80)
    ap.add_argument('--stride', type=int, default=17, help='pixel subsample for memory')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import h5py
    bb = _Teacher(a.teacher)
    meta = json.load(open(os.path.join(a.root, 'ZJUL5', 'data.json')))
    names = ([m['filename'] for m in meta[a.split]] if a.split != 'all'
             else [m['filename'] for k in meta for m in meta[k]])[:a.limit]

    E, S, G, CEIL, AMAX = [], [], [], [], []
    for rel in names:
        with h5py.File(os.path.join(a.root, 'ZJUL5', rel), 'r') as f:
            rgb = f['rgb'][:]
            gt = f['depth'][:].astype(np.float64)
            fr, hist, mask = f['fr'][:], f['hist_data'][:], f['mask'][:]
        h, w = gt.shape
        u, v, z = zone_anchors(fr, hist, mask, h, w)
        if u.size < 8:
            continue
        disp = bb.infer(np.ascontiguousarray(rgb))
        disp_at = disp[v, u].astype(np.float64)
        inv = 1.0 / z
        wts = np.ones_like(z)
        fit = anc.solve_robust(disp_at, inv, wts, iters=1)
        if fit is None:
            continue
        a_, b_ = fit
        cov = anc.covariance(disp_at, inv, wts, a_, b_)
        if cov is None:
            continue
        D = np.clip(anc.to_metric_depth(disp, a_, b_), None, MAX_DEPTH_M)
        # Var[D] = D^4 * j^T Cov j, j = (disp, 1) -- the delta method. D^4 is why far
        # pixels should carry far larger variance.
        j0, j1 = disp.astype(np.float64), np.ones_like(disp, np.float64)
        quad = (j0 * (cov[0, 0] * j0 + cov[0, 1] * j1) +
                j1 * (cov[1, 0] * j0 + cov[1, 1] * j1))
        sig = np.sqrt(np.clip(D.astype(np.float64) ** 4 * quad, 1e-12, None))

        gv = (gt > GT_MIN) & (gt < GT_MAX)
        sl = (slice(None, None, a.stride), slice(None, None, a.stride))
        m = gv[sl]
        E.append((D - gt)[sl][m]); S.append(sig[sl][m]); G.append(gt[sl][m])
        CEIL.append(1.0 / max(b_, 1e-9)); AMAX.append(float(z.max()))

    E = np.concatenate(E); S = np.concatenate(S); G = np.concatenate(G)
    A = np.abs(E)
    print(f'\n{len(CEIL)} frames, {E.size} pixels sampled (stride {a.stride})')
    print(f'furthest anchor / frame : median {np.median(AMAX):.2f} m')
    print(f'model ceiling 1/b       : median {np.median(CEIL):.2f} m   '
          f'(p10 {np.percentile(CEIL,10):.2f}, p90 {np.percentile(CEIL,90):.2f})')
    print('  -- the deepest value the affine form can express at disp -> 0')

    print(f'\n{"true depth":>12}{"n":>9}{"mean err":>11}{"med |e|":>9}{"med sigma":>11}'
          f'{"sigma/|e|":>11}{"share RMSE^2":>13}')
    tot = float((E ** 2).sum())
    for lo, hi in BANDS:
        s = (G >= lo) & (G < hi)
        if s.sum() < 50:
            continue
        me, ms = float(np.median(A[s])), float(np.median(S[s]))
        print(f'{lo:5.0f}-{hi:<6.0f}{s.sum():9d}{E[s].mean():>+10.3f}m{me:>8.3f}m'
              f'{ms:>10.3f}m{ms/max(me,1e-9):>11.2f}{100*(E[s]**2).sum()/tot:>12.1f}%')

    fin = np.isfinite(S) & np.isfinite(A)
    print(f'\ncorr(sigma, |error|)      : {np.corrcoef(S[fin], A[fin])[0,1]:+.3f}   '
          f'(claimed 0.943 on our own sensor)')
    print(f'spearman (rank) corr      : '
          f'{np.corrcoef(np.argsort(np.argsort(S[fin])), np.argsort(np.argsort(A[fin])))[0,1]:+.3f}')
    print(f'coverage |e| <= 1 sigma   : {np.mean(A <= S):.3f}   (ideal {IDEAL_1S:.3f})')
    print(f'coverage |e| <= 2 sigma   : {np.mean(A <= 2*S):.3f}   (ideal {IDEAL_2S:.3f})')

    # THE decisive test: can a consumer drop a small high-sigma slice and lose the tail?
    print(f'\ntail capture -- drop the top X% of pixels BY SIGMA:')
    print(f'{"drop":>8}{"RMSE^2 removed":>17}{"RMSE after":>13}{"medAE after":>13}')
    order = np.argsort(-S)
    for frac in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        cut = int(frac * S.size)
        keep = np.ones(S.size, bool); keep[order[:cut]] = False
        removed = 1.0 - float((E[keep] ** 2).sum()) / tot
        print(f'{100*frac:>7.1f}%{100*removed:>16.1f}%'
              f'{np.sqrt((E[keep]**2).mean()):>12.3f}m{np.median(A[keep]):>12.3f}m')
    print(f'{"none":>8}{0.0:>16.1f}%{np.sqrt((E**2).mean()):>12.3f}m{np.median(A):>12.3f}m')
    print('\nIf a ~1% drop removes most of RMSE^2, sigma is a usable filter and the far\n'
          'field is a reported limitation rather than a hole. If not, sigma is flat here.')

    if a.out:
        json.dump({'corr': float(np.corrcoef(S[fin], A[fin])[0, 1]),
                   'cov1': float(np.mean(A <= S)), 'cov2': float(np.mean(A <= 2 * S)),
                   'ceiling_median': float(np.median(CEIL)),
                   'anchor_max_median': float(np.median(AMAX)),
                   'n_px': int(E.size), 'n_frames': len(CEIL)}, open(a.out, 'w'), indent=1)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()

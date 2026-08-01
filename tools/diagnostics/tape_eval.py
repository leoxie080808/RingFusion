#!/usr/bin/env python3
"""Score the pipeline against tape/laser ground truth, split INSIDE vs OUTSIDE the ToF cone.

This is the only open-loop evaluation on our own rig. Everything else in this repo is scored
against the ToF the pipeline anchors to, which cannot detect a ToF bias and says nothing
about the 92.5% of the frame the ToF never covers. The inside/outside split IS the headline:

  * inside  -- should roughly match the ToF-scored numbers. If it does not, the ToF itself
               is biased and every other figure in the repo inherits that bias.
  * outside -- unconstrained monocular extrapolation carrying the affine fit. A large gap
               here is a REAL FINDING to publish, not a failure: it quantifies the limit the
               READMEs currently describe only qualitatively.

Reads the JSON written by tape_capture.py. Two corrections are applied here rather than at
capture time, so the raw measurements stay raw and re-scoring never needs a re-measure:

  1. ORIGIN OFFSET. The tape origin is the optical centre, inside the lens barrel, not the
     housing face you can put a tape against. `origin_offset_m` is ADDED to every range.
  2. SLANT RANGE -> AXIS DEPTH. The instrument gives range r along the ray; the pipeline
     predicts optical-axis depth z. z = r*cos(theta), theta from the pixel and K_rect.
     Skipping this fabricates a radial bias that grows toward the frame edge -- exactly the
     shape of a real error, which is what makes it dangerous.
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_perception'))
sys.path.insert(0, os.path.join(_REPO, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as M                                          # noqa: E402
from anchoring_bridge import calib_from_yaml                 # noqa: E402
# This workspace stores K as (fx, fy, cx, cy), not a 3x3 -- roi already has the one
# accessor that handles both, so use it rather than duplicating the unpacking here.
from ringfusion_perception.roi import _fxfycxcy              # noqa: E402

# Uncertainty budget, quoted in the output because a reviewer will ask.
SIGMA_INSTRUMENT = {'laser': 0.003, 'tape': 0.005}
SIGMA_ORIGIN = 0.010          # how well the housing->optical-centre offset is known
SIGMA_MARK = 0.005            # marker placement / click correspondence


def axis_depth(r, u, v, K):
    """Slant range -> optical-axis depth. z = r*cos(theta), cos(theta) = 1/|ray|."""
    fx, fy, cx, cy = _fxfycxcy(K)
    x = (np.asarray(u, float) - cx) / fx
    y = (np.asarray(v, float) - cy) / fy
    return np.asarray(r, float) / np.sqrt(1.0 + x * x + y * y)


def tof_footprint(calib, h, w):
    """Pixel rect covered by the ToF cone, from the calibrated FOV. Same construction the
    pipeline uses to place anchors, so 'inside' means the same thing here as everywhere."""
    fx, fy, cx, cy = _fxfycxcy(calib['K'])
    th = np.deg2rad(float(calib['fov_h']) / 2.0)
    tv = np.deg2rad(float(calib['fov_v']) / 2.0)
    du, dv = fx * np.tan(th), fy * np.tan(tv)
    return (max(0, int(cx - du)), max(0, int(cy - dv)),
            min(w, int(cx + du)), min(h, int(cy + dv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='dir written by tape_capture.py')
    ap.add_argument('--calib', required=True)
    ap.add_argument('--gt', default='', help='defaults to <dir>/tape_gt.json')
    ap.add_argument('--patch', type=int, default=1,
                    help='median over a (2k+1)^2 patch at the marked pixel; 1 = single px')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    gt_path = a.gt or os.path.join(a.dir, 'tape_gt.json')
    doc = json.load(open(gt_path))
    pts = doc['points']
    if not pts:
        sys.exit(f'no points in {gt_path}')
    off = float(doc.get('origin_offset_m', 0.0))
    instrument = doc.get('instrument', 'tape')

    first = np.load(os.path.join(a.dir, pts[0]['stem'] + '_depth.npy'))
    h, w = first.shape
    calib = calib_from_yaml(a.calib, train_size=(h, w))
    K = calib['K']
    x0, y0, x1, y1 = tof_footprint(calib, h, w)

    pred, gt, inside, labels, us, vs, sigma = [], [], [], [], [], [], []
    for q in pts:
        D = np.load(os.path.join(a.dir, q['stem'] + '_depth.npy'))
        u, v = int(q['u']), int(q['v'])
        if a.patch > 1:
            k = a.patch
            sub = D[max(0, v - k):v + k + 1, max(0, u - k):u + k + 1]
            sub = sub[np.isfinite(sub) & (sub > 0)]
            p = float(np.median(sub)) if sub.size else float('nan')
        else:
            p = float(D[v, u])
        # sigma comes straight from the published /depth_var, so this checks the DEPLOYED
        # uncertainty -- not the analytic term alone, which is ~0.1% of it and is the only
        # part any previous calibration result covered.
        vp = os.path.join(a.dir, q['stem'] + '_var.npy')
        sigma.append(float(np.sqrt(max(np.load(vp)[v, u], 0.0)))
                     if os.path.exists(vp) else float('nan'))
        pred.append(p)
        gt.append(axis_depth(float(q['range_m']) + off, u, v, K))
        inside.append(x0 <= u < x1 and y0 <= v < y1)
        labels.append(q.get('label', '?'))
        us.append(u); vs.append(v)

    pred = np.array(pred); gt = np.array(gt); inside = np.array(inside)
    sigma = np.array(sigma)

    sig = np.sqrt(SIGMA_INSTRUMENT.get(instrument, 0.005) ** 2
                  + SIGMA_ORIGIN ** 2 + SIGMA_MARK ** 2)
    print(f'{len(pts)} points from {gt_path}')
    print(f'ToF footprint: x {x0}-{x1}, y {y0}-{y1} of {w}x{h} '
          f'({100.0 * (x1 - x0) * (y1 - y0) / (w * h):.1f}% of frame)')
    print(f'inside {int(inside.sum())}   outside {int((~inside).sum())}')
    print(f'origin offset {off:+.3f} m applied; instrument "{instrument}"')
    print(f'GT uncertainty: {sig * 1000:.0f} mm '
          f'(instrument {SIGMA_INSTRUMENT.get(instrument, 0.005)*1000:.0f} + origin '
          f'{SIGMA_ORIGIN*1000:.0f} + marker {SIGMA_MARK*1000:.0f}, in quadrature)\n')

    if int((~inside).sum()) == 0:
        print('!! every point is inside the ToF cone -- the outside split is the entire\n'
              '   reason for this exercise. Capture more points near the frame edges.\n')

    print(f'{"#":>3} {"label":<18}{"u,v":>12}{"reg":>5}{"GT z":>8}{"pred":>8}'
          f'{"err":>8}{"rel":>8}')
    print('-' * 74)
    for i, (p, g, ins, lb, u, v) in enumerate(zip(pred, gt, inside, labels, us, vs)):
        e = p - g
        print(f'{i:>3} {lb[:18]:<18}{f"{u},{v}":>12}{"in" if ins else "out":>5}'
              f'{g:>8.3f}{p:>8.3f}{e:>+8.3f}{100*abs(e)/max(g,1e-9):>7.1f}%')

    rows, report = [], {}
    # Names kept under the 14-char column metrics.format_row reserves, so the table lines up.
    for sel, name, key in ((np.ones_like(inside), 'ALL points', 'all'),
                           (inside, 'INSIDE ToF', 'inside'),
                           (~inside, 'OUTSIDE ToF', 'outside')):
        if not sel.any():
            continue
        m = M.depth_metrics(pred[sel], gt[sel])
        rows.append((name, m))
        report[key] = m
    print('\n' + M.format_table(rows))
    # metrics.format_row does not carry an RMSE column, and reflowing it would change the
    # width of every table already quoted in the READMEs. RMSE matters most here of all
    # places -- the far-field tail is the one target ZJU-L5 says we miss (2.7x worse than
    # DEPTHOR-Small) and these are the only far, open-loop points we have -- so print it
    # alongside rather than silently leaving it to the JSON.
    print('\nRMSE (lower better), the far-field tail metric:')
    for name, m in rows:
        print(f'  {name:<14}{m["rmse"]:>9.3f} m'
              f'    (MAE {m["mae"]:.3f} m; RMSE >> MAE means a few large outliers)')

    # --- sigma calibration against INDEPENDENT ground truth ---------------------------
    # Every previous calibration number in this repo was scored against the ToF the pipeline
    # anchors to, and covered only the ANALYTIC variance term (~0.1% of what /depth_var
    # actually publishes). This is the first check of the DEPLOYED sigma against a
    # measurement the pipeline never saw.
    if np.isfinite(sigma).any():
        print('\n=== sigma calibration (deployed /depth_var, vs independent GT) ===')
        print('target is 0.683 EXACTLY -- this is a target, not a direction.')
        print('  below 0.683 = overconfident (sigma too small, the dangerous failure)')
        print('  above 0.683 = sigma inflated and useless (it flags everything)')
        hdr = f'{"region":<14}{"n":>5}{"cov@1σ":>9}{"cov@2σ":>9}{"med σ":>9}{"med |err|":>11}'
        print(hdr); print('-' * len(hdr))
        for sel, name in ((np.ones_like(inside), 'ALL'), (inside, 'INSIDE ToF'),
                          (~inside, 'OUTSIDE ToF')):
            ok = sel & np.isfinite(sigma) & np.isfinite(pred) & (sigma > 0)
            if not ok.any():
                continue
            err = np.abs(pred[ok] - gt[ok])
            c1 = float((err <= sigma[ok]).mean())
            c2 = float((err <= 2 * sigma[ok]).mean())
            flag = '' if abs(c1 - 0.683) < 0.15 else ('  <- OVERCONFIDENT' if c1 < 0.683
                                                      else '  <- inflated')
            print(f'{name:<14}{int(ok.sum()):>5}{c1:>9.3f}{c2:>9.3f}'
                  f'{np.median(sigma[ok]):>8.3f}m{np.median(err):>10.3f}m{flag}')
        n_ok = int((np.isfinite(sigma) & (sigma > 0)).sum())
        if n_ok < 20:
            print(f'\n  NOTE: only {n_ok} points. A coverage fraction from <20 samples has a '
                  f'±{100*0.5/max(n_ok,1)**0.5:.0f}% standard error —\n'
                  f'  enough to distinguish 0.23 from 0.68, not enough to certify 0.68.')
    else:
        print('\nsigma calibration: SKIPPED, no *_var.npy files '
              '(was /depth_var publishing during capture?)')

    print('\nlower is better: mae, medae, p95, rmse, absrel, |bias|')
    print('higher is better: coverage, d1, d2, d3')
    print(f'\nGT is +-{sig*1000:.0f} mm. Any error term below ~{2*sig:.3f} m is at the '
          f'noise floor of this experiment and must not be interpreted.')
    if 'inside' in report and 'outside' in report:
        ri, ro = report['inside'], report['outside']
        if np.isfinite(ri['absrel']) and np.isfinite(ro['absrel']) and ri['absrel'] > 0:
            print(f'\noutside/inside AbsRel ratio: {ro["absrel"]/ri["absrel"]:.2f}x '
                  f'-- this is the headline number of the whole validation plan.')

    if a.out:
        json.dump({'n': len(pts), 'gt_sigma_m': sig, 'footprint': [x0, y0, x1, y1],
                   'origin_offset_m': off, 'instrument': instrument,
                   'regions': report}, open(a.out, 'w'), indent=1)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()

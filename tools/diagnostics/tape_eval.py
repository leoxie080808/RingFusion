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
from tape_geometry import (axial_from_legs, slant_from_legs,  # noqa: E402
                           load_geometry, has_legs)

# Uncertainty budget, quoted in the output because a reviewer will ask.
SIGMA_INSTRUMENT = {'laser': 0.003, 'tape': 0.005}
SIGMA_ORIGIN = 0.010          # how well the housing->optical-centre offset is known
SIGMA_MARK = 0.001            # locating the marker centre (run sheet)

# --- legs route (T3) ---------------------------------------------------------------------
# Measuring three legs from the robot centre replaces the two largest terms above. SIGMA_ORIGIN
# (10 mm, the housing->optical-centre guess) drops out entirely: CAD gives the optical centre
# directly, and what is left is how well the floor chalk mark matches the CAD origin. The
# pointing error hidden inside the slant-range route also drops out, because theta now comes
# from the tape rather than from the marker's pixel.
#
# In exchange, two NEW terms appear, and both scale with how far off-axis the marker is:
# yaw uncertainty multiplied by the sideways leg, and pitch uncertainty multiplied by the
# height drop. They are small on-axis and are NOT small for the out-of-cone points that carry
# the headline inside/outside gap -- at 1.5 m sideways, half a degree of yaw uncertainty is
# 13 mm, larger than every other term combined. So the legs budget is computed PER POINT
# rather than quoted as one number for the session.
# Values from the T3 run sheet's own budget (t3_testRun.md), which is better founded than the
# placeholders these replace -- it assumes a steel tape read to 1 mm and a plumbed, chalked O.
#   tape reading, 1 mm graduations ....... 0.3 mm   -> SIGMA_LEG
#   locating O and the forward axis ...... 1   mm \
#   CAD offset to the lens front ......... 1   mm  |- 2.45 mm combined -> SIGMA_CAD
#   optical centre inside the lens ....... 2   mm /   (the 1.13 mm datum choice, carried
#                                                      rather than corrected)
#   locating the marker centre ........... 1   mm   -> SIGMA_MARK
# Root-sum-square of those five is 2.7 mm, which is the run sheet's headline number and what
# this reproduces on-axis. The angle terms below are what the run sheet's table omits.
SIGMA_LEG = 0.0003            # one steel-tape leg, 1 mm graduations, read on the floor
SIGMA_CAD = 0.00245           # locating O + CAD offset + optical-centre datum, in quadrature
# Pitch and yaw are kept SEPARATE because their lever arms differ by an order of magnitude and
# they are known to very different precisions. Pitch multiplies the height drop h (0.1-1.0 m on
# this rig); yaw multiplies the sideways leg y, which reaches 2.5 m. So yaw is ~3x the problem
# pitch is -- and it is the one with no independent constraint: the floor plane that bounds
# pitch is invariant to yaw. Defaults are the honest CURRENT state (see
# fov_v_session_2026-08-04.json, whose own note says the floor test "puts any pitch under ~2 deg
# and cannot resolve its sign or size"), NOT an aspiration. Lower them only once a boresight
# measurement exists -- see robot_geometry.yaml.
SIGMA_PITCH_DEG = 2.0
SIGMA_YAW_DEG = 2.0


def axis_depth(r, u, v, K):
    """Slant range -> optical-axis depth. z = r*cos(theta), cos(theta) = 1/|ray|."""
    fx, fy, cx, cy = _fxfycxcy(K)
    x = (np.asarray(u, float) - cx) / fx
    y = (np.asarray(v, float) - cy) / fy
    return np.asarray(r, float) / np.sqrt(1.0 + x * x + y * y)


def legs_to_axis_depth(x_m, y_m, z_m, g):
    """Three tape legs from the ROBOT CENTRE -> (axial depth Z, slant range r).

    This is the T3 reference, and it is strictly better than the slant-range route
    (axis_depth, below): that one takes theta from the marker's PIXEL, so a click landing off
    the marker centre feeds a pointing error straight into the ground truth, and section 3.12
    traced ~0.77 m of the largest out-of-cone error to exactly that. Legs take theta from the
    tape, so the reference no longer depends on the click at all.

    The conversion and both sign conventions live in tape_geometry, which also holds the
    checks that pin them. r is a CROSS-CHECK ONLY and is never the reference.
    """
    return (axial_from_legs(x_m, y_m, z_m, g), slant_from_legs(x_m, y_m, z_m, g))


def legs_sigma(x_m, y_m, z_m, g):
    """Per-point 1-sigma on the legs-derived Z, by first-order propagation.

    Z = d cos(t) cos(p) + y cos(t) sin(p) + h sin(t), so

        dZ/dx_m = cos(t) cos(p)        dZ/dt = -d sin(t) cos(p) - y sin(t) sin(p) + h cos(t)
        dZ/dy_m = cos(t) sin(p)        dZ/dp = -d cos(t) sin(p) + y cos(t) cos(p)
        dZ/dz_m = -sin(t)

    and the same magnitudes apply to x_c, y_c, z_c with the opposite sign, so each pairs with
    its own leg in quadrature. Near level the forward leg carries essentially all of the
    depth uncertainty and dZ/dp -> y: a sideways marker converts yaw error into depth error
    at almost 1:1, which is the term the old flat budget had no way to show.
    """
    d = np.asarray(x_m, float) - g['x_c']
    y = np.asarray(y_m, float) - g['y_c']
    h = g['z_c'] - np.asarray(z_m, float)
    t, p = np.deg2rad(g['pitch_deg']), np.deg2rad(g['yaw_deg'])
    ct, st, cp, sp = np.cos(t), np.sin(t), np.cos(p), np.sin(p)
    s_pitch, s_yaw = np.deg2rad(SIGMA_PITCH_DEG), np.deg2rad(SIGMA_YAW_DEG)
    leg2 = SIGMA_LEG ** 2 + SIGMA_CAD ** 2      # marker leg + robot-centre origin, per axis
    return np.sqrt((ct * cp) ** 2 * leg2
                   + (ct * sp) ** 2 * leg2
                   + st ** 2 * leg2
                   + ((-d * st * cp - y * st * sp + h * ct) * s_pitch) ** 2
                   + ((-d * ct * sp + y * ct * cp) * s_yaw) ** 2
                   + SIGMA_MARK ** 2)


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
    ap.add_argument('--geometry',
                    default=os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_bringup',
                                         'config', 'robot_geometry.yaml'),
                    help='items 27-29, needed only for sessions that recorded tape legs')
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

    geom = load_geometry(a.geometry)
    n_legs = sum(1 for q in pts if q.get('x_m') is not None)
    if n_legs:
        print(f'{n_legs}/{len(pts)} points recorded tape legs from the robot centre'
              + (f" -- geometry from {geom['source']}" if geom else ' -- GEOMETRY UNSET'))

    pred, gt, inside, labels, us, vs, sigma, slant = [], [], [], [], [], [], [], []
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
        # Two reference routes. LEGS are preferred and are what T3 captures: theta comes
        # from the tape, so the reference carries no pointing error. RANGE_M is the older
        # slant-range route, kept so sessions 1 and 2 keep scoring identically.
        if q.get('x_m') is not None:
            if geom is None:
                sys.exit(f"point {q['id']} recorded tape legs, but {a.geometry} has UNSET "
                         f"fields. Items 27-29 must be filled in before a legs session can "
                         f"be scored -- there is no defaulting them.")
            z_ref, r_ref = legs_to_axis_depth(q['x_m'], q['y_m'], q['z_m'], geom)
            gt.append(float(z_ref))
            slant.append(float(r_ref))
        else:
            gt.append(axis_depth(float(q['range_m']) + off, u, v, K))
            slant.append(float(q['range_m']) + off)
        inside.append(x0 <= u < x1 and y0 <= v < y1)
        labels.append(q.get('label', '?'))
        us.append(u); vs.append(v)

    pred = np.array(pred); gt = np.array(gt); inside = np.array(inside)
    sigma = np.array(sigma)

    # Two budgets, because the two reference routes have genuinely different error terms
    # (see SIGMA_LEG). A legs session gets a per-point sigma; `sig` below is then the WORST
    # point, not the typical one, so the "do not interpret below this" line stays honest for
    # every point in the table rather than only the on-axis ones.
    if n_legs and geom is not None:
        per_pt = np.array([float(legs_sigma(q['x_m'], q['y_m'], q['z_m'], geom))
                           if q.get('x_m') is not None else np.nan for q in pts])
        sig = float(np.nanmax(per_pt))
        sig_med = float(np.nanmedian(per_pt))
    else:
        per_pt = None
        sig = sig_med = float(np.sqrt(SIGMA_INSTRUMENT.get(instrument, 0.005) ** 2
                                      + SIGMA_ORIGIN ** 2 + SIGMA_MARK ** 2))
    print(f'{len(pts)} points from {gt_path}')
    print(f'ToF footprint: x {x0}-{x1}, y {y0}-{y1} of {w}x{h} '
          f'({100.0 * (x1 - x0) * (y1 - y0) / (w * h):.1f}% of frame)')
    print(f'inside {int(inside.sum())}   outside {int((~inside).sum())}')
    print(f'origin offset {off:+.3f} m applied; instrument "{instrument}"')
    if per_pt is not None:
        worst = int(np.nanargmax(per_pt))
        print(f'GT uncertainty (legs route): median {sig_med*1000:.1f} mm, '
              f'worst {sig*1000:.1f} mm at point {pts[worst]["id"]} '
              f'("{pts[worst].get("label","?")}", y={pts[worst]["y_m"]:+.2f} m)')
        print(f'  per-axis leg {SIGMA_LEG*1000:.0f} + robot-centre mark {SIGMA_CAD*1000:.0f} '
              f'+ marker {SIGMA_MARK*1000:.0f} mm, pitch +-{SIGMA_PITCH_DEG} deg, '
              f'yaw +-{SIGMA_YAW_DEG} deg; '
              f'the {SIGMA_ORIGIN*1000:.0f} mm origin term does not apply')
        if max(SIGMA_PITCH_DEG, SIGMA_YAW_DEG) > 0.5 and sig > 0.030:
            print(f'  !! the angle terms dominate at {sig*1000:.0f} mm, which is WORSE than the '
                  f'{np.sqrt(SIGMA_INSTRUMENT.get(instrument, 0.005)**2 + SIGMA_ORIGIN**2 + SIGMA_MARK**2)*1000:.0f} '
                  f'mm slant-range budget this scheme replaced. Measure pitch/yaw with a '
                  f'boresight target before quoting these numbers -- see robot_geometry.yaml.')
    else:
        print(f'GT uncertainty: {sig * 1000:.0f} mm '
          f'(instrument {SIGMA_INSTRUMENT.get(instrument, 0.005)*1000:.0f} + origin '
          f'{SIGMA_ORIGIN*1000:.0f} + marker {SIGMA_MARK*1000:.0f}, in quadrature)')
    print()

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
                   'reference': 'legs' if per_pt is not None else 'slant_range',
                   'n_legs': int(n_legs),
                   'geometry': geom,
                   'gt_sigma_median_m': sig_med,
                   'gt_sigma_per_point_m': (None if per_pt is None
                                            else [None if not np.isfinite(x) else float(x)
                                                  for x in per_pt]),
                   'slant_m': [float(x) for x in slant],
                   'regions': report}, open(a.out, 'w'), indent=1)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()

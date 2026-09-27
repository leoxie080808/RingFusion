#!/usr/bin/env python3
"""T6: how often the anchor fit fails, and on which guard.

The paper (r34+) describes what the code does: the frame is DROPPED. There is no
scale-only fallback and no N_min / v_min / kappa_max -- those belonged to the original
description and were never implemented. So this measures the real guards, all three of
which live in anchoring.solve_scale_shift:

    1. w.size < 2          fewer than 2 anchors
    2. w.sum() <= eps      zero total weight
    3. |den| < eps         singular normal matrix (den is its DETERMINANT, eps = 1e-9)

and one more that gates sigma rather than the fit: covariance() returns None at n <= 2,
because sigma^2 needs n-2 degrees of freedom. When that happens the node publishes depth
and cloud but NO /depth_var message at all (perception_node.py:210).

Also reported, because limitation 4 asserts it: the spread of anchor count, weighted
disparity variance and condition number, so the paper can say how often a nearly
uniform-depth scene passes the guards yet yields a poorly conditioned fit.

And a cross-check: baselines.py rejects frames with cond(X^T X) > 1e8, but the node
publishes them. This counts how many frames that eval-only guard removes.
"""
import argparse, json, os, sys, statistics
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path[:0] = [_HERE, os.path.join(_REPO, 'training'),
                os.path.join(_REPO, 'ros2_ws/src/ringfusion_perception')]
import envinfo
from anchoring_bridge import calib_from_yaml
from ringfusion_perception import pipeline, anchoring as anc, roi

EPS = 1e-9
REC = []            # one dict per solve_scale_shift call


def instrument():
    real = anc.solve_scale_shift
    def wrapped(disp, inv_depth, weights, eps=EPS, b_prior=0.0):
        w = np.asarray(weights, float); d = np.asarray(disp, float)
        r = {'n_anchors': int(w.size), 'w_sum': float(w.sum()) if w.size else 0.0,
             'guard': None, 'den': None, 'cond': None, 'disp_var_w': None}
        if w.size < 2 or w.sum() <= eps:
            r['guard'] = 'few_anchors' if w.size < 2 else 'zero_weight'
            REC.append(r); return None
        Sw = w.sum(); Swd = float(np.sum(w*d)); Swdd = float(np.sum(w*d*d))
        lam = 0.0 if not np.isfinite(b_prior) else float(b_prior)*Sw
        den = Swdd*(Sw+lam) - Swd*Swd
        r['den'] = float(den)
        # weighted variance of the predicted disparity -- the quantity v_min would have gated
        mu = Swd / Sw
        r['disp_var_w'] = float(np.sum(w*(d-mu)**2) / Sw)
        N = np.array([[Swdd, Swd], [Swd, Sw+lam]])
        try:
            r['cond'] = float(np.linalg.cond(N))
        except Exception:
            r['cond'] = float('inf')
        if abs(den) < eps:
            r['guard'] = 'singular_normal_matrix'
            REC.append(r); return None
        REC.append(r)
        return real(disp, inv_depth, weights, eps=eps, b_prior=b_prior)
    anc.solve_scale_shift = wrapped
    pipeline.anc.solve_scale_shift = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-dir', required=True); ap.add_argument('--tof-dir', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--backbone-engine', required=True); ap.add_argument('--residual-engine', required=True)
    ap.add_argument('--val-stems', default=''); ap.add_argument('--size', default='1232x1640')
    ap.add_argument('--eval-max-cond', type=float, default=1e8)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    import cv2
    from ringfusion_perception.backbone import TensorRTBackbone
    from ringfusion_perception.residual import ResidualRefiner
    H, W = (int(v) for v in a.size.lower().split('x'))
    bb = TensorRTBackbone(a.backbone_engine); rf = ResidualRefiner(a.residual_engine)
    calib = calib_from_yaml(a.calib, (H, W))
    val = set(open(a.val_stems).read().split()) if a.val_stems else set()
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a.tof_dir) if f.endswith('.npz'))
    instrument()

    frames = []
    tracker = roi.PlaneTracker(refit_every=1)
    for st in stems:
        p = os.path.join(a.rgb_dir, st + '.png')
        if not os.path.exists(p): continue
        rgb = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (W, H),
                         interpolation=cv2.INTER_AREA)
        z = np.load(os.path.join(a.tof_dir, st + '.npz'))
        dist = z['dist_m'].astype(np.float32); conf = z['confidence']
        # An expired half-map shows up as an entire parity of rows being NaN
        ev = bool(np.all(~np.isfinite(dist[0::2, :]))); od = bool(np.all(~np.isfinite(dist[1::2, :])))
        REC.clear()
        r = pipeline.run(rgb, dist, np.isfinite(dist), calib, bb, rf, confidence=conf,
                         min_confidence=-1, blend=True, roi_enable=True, plane_tracker=tracker)
        rec = REC[-1] if REC else {'n_anchors': 0, 'guard': 'no_call', 'cond': None,
                                   'disp_var_w': None, 'w_sum': 0.0, 'den': None}
        frames.append({'stem': st, 'in_val': st in val, 'ok': bool(r.get('ok')),
                       'n_anchors_pipeline': int(r.get('n_anchors', 0)),
                       'a': (None if not r.get('ok') else float(r['a'])),
                       'expired_even': ev, 'expired_odd': od, **rec})

    n = len(frames)
    drops = [f for f in frames if not f['ok']]
    guards = {}
    for f in frames:
        if f['guard']: guards[f['guard']] = guards.get(f['guard'], 0) + 1
    conds = [f['cond'] for f in frames if f['cond'] is not None and np.isfinite(f['cond'])]
    dvars = [f['disp_var_w'] for f in frames if f['disp_var_w'] is not None]
    nanc = [f['n_anchors'] for f in frames]
    q = lambda v, p: round(float(np.percentile(v, p)), 6) if v else None

    evalcut = [f for f in frames if f['cond'] is not None and f['cond'] > a.eval_max_cond]
    evalcut_val = [f for f in evalcut if f['in_val']]
    sigma_fail = [f for f in frames if f['ok'] and f['n_anchors'] <= 2]

    rep = {
      'label': 'T6: frame-drop frequency and real guard firing rates',
      'note': ('The paper (r34+) describes frame-dropping, which is what the code does. '
               'N_min / v_min / kappa_max were never implemented and are not measured.'),
      'frames': n, 'resolution': f'{H}x{W}',
      'drops': {'count': len(drops), 'rate_pct': round(100*len(drops)/max(n,1), 4),
                'stems': [f['stem'] for f in drops][:50]},
      'guard_fire_counts': guards,
      'zero_anchor_frames': sum(1 for f in frames if f['n_anchors_pipeline'] == 0),
      'expired_half_map_frames': sum(1 for f in frames if f['expired_even'] or f['expired_odd']),
      'fitted_scale_le_zero': sum(1 for f in frames if f['a'] is not None and f['a'] <= 0),
      'sigma_unavailable': {
          'count': len(sigma_fail),
          'rate_pct': round(100*len(sigma_fail)/max(n,1), 4),
          'consequence': ('covariance() returns None at n<=2, so var is None and the node '
                          'publishes depth and cloud but NO /depth_var message '
                          '(perception_node.py:210) -- the consumer gets silence, not a '
                          'high-sigma warning.')},
      'distributions': {
          'anchor_count':   {'min': min(nanc), 'p05': q(nanc,5), 'median': q(nanc,50),
                             'p95': q(nanc,95), 'max': max(nanc)},
          'disparity_var_weighted': {'p05': q(dvars,5), 'median': q(dvars,50), 'p95': q(dvars,95),
                                     'min': q(dvars,0), 'max': q(dvars,100)},
          'condition_number': {'p05': q(conds,5), 'median': q(conds,50), 'p95': q(conds,95),
                               'max': q(conds,100)}},
      'eval_only_cond_guard': {
          'threshold': a.eval_max_cond,
          'note': ('baselines.py:346 rejects frames with cond(X^T X) above this; the '
                   'deployed node has no such guard and publishes them.'),
          'frames_removed_full_set': len(evalcut),
          'frames_removed_val61': len(evalcut_val),
          'stems': [f['stem'] for f in evalcut][:50]},
      'env': envinfo.capture(note='t6_fallback_frequency')}

    print(f"frames {n}   drops {len(drops)} ({rep['drops']['rate_pct']}%)")
    print(f"guard fires: {guards or 'none'}")
    print(f"zero-anchor {rep['zero_anchor_frames']}  expired-half {rep['expired_half_map_frames']}"
          f"  scale<=0 {rep['fitted_scale_le_zero']}")
    print(f"sigma unavailable (n<=2): {len(sigma_fail)} ({rep['sigma_unavailable']['rate_pct']}%)")
    d = rep['distributions']
    print(f"anchors  min {d['anchor_count']['min']}  median {d['anchor_count']['median']}  max {d['anchor_count']['max']}")
    print(f"cond     median {d['condition_number']['median']}  p95 {d['condition_number']['p95']}  max {d['condition_number']['max']}")
    print(f"disp_var median {d['disparity_var_weighted']['median']}  p05 {d['disparity_var_weighted']['p05']}")
    print(f"eval-only cond>{a.eval_max_cond:g} guard removes: {len(evalcut)} of {n} full, {len(evalcut_val)} of val61")
    if a.out:
        json.dump(rep, open(a.out, 'w'), indent=1); print('wrote', a.out)


if __name__ == '__main__':
    main()

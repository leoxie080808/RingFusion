#!/usr/bin/env python3
"""Per-term ablation of the uncertainty estimate, replayed offline from a tape capture.

The reviewer asked for an ablation separating analytic anchoring, robust weighting, the
residual refiner, arbitration AND uncertainty estimation. The first four live in a
median-error table; the fifth cannot, because the quantity being ablated is not a depth.
This scores each arm the way an uncertainty estimate is actually judged: does sigma order
the errors, does its coverage match its nominal rate, and how bad is the worst residual.

WHY REPLAY RATHER THAN FIVE LIVE CAPTURES. The three sigma constants are module-level in
blend.py. Ablating them by editing the module would mean re-running the whole tape session
once per arm -- five sittings of a scene that must not move between them, and the brief
forbids changing constants mid-collection anyway. tape_capture.py already stores the raw
inputs per frozen frame (<stem>_rgb.png and <stem>_tof.npy), so every arm can be recomputed
from one capture.

DEPTH IS IDENTICAL ACROSS ARMS. Arms differ only in which variance terms are summed --
`learned_var` keeps the refined depth while dropping tau^2, and the sigma_terms dict zeroes
blend terms without touching the blend itself. Setting residual=None would have changed the
depth too and made the comparison meaningless.

THE VALIDATION ARM IS NOT OPTIONAL. Replay cannot be bit-identical to the live run: the
capture froze rgb/tof/depth/var from four independent subscriptions, so the stored image
may be a frame newer than the one that produced the stored variance, and PlaneTracker
carries an EMA that a fresh replay has to re-converge. `full` is therefore compared against
the saved *_var.npy and the agreement is reported. If that disagreement is large, no other
arm here means anything, and the tool says so rather than printing a table anyway.

    python3 tools/diagnostics/sigma_ablation.py --dir <tape capture dir> \
        --calib ros2_ws/src/ringfusion_bringup/config/calibration.yaml \
        --backbone-engine student_v4_heldout_fp16.engine \
        --residual-engine residual_v7_fov73_fp16.engine \
        --out docs/demo/benchmarks/sigma_ablation_r31.json
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anchoring_bridge import calib_from_yaml                      # noqa: E402
from bootstrap import wilson, frame_bootstrap                     # noqa: E402
import envinfo                                                    # noqa: E402

# Cumulative arms: each adds one variance source to the one above it. The last is the
# deployed configuration, which is what the validation check compares against.
ARMS = [
    ('A_vfit_only',   dict(learned_var=False, sigma_terms=dict(disagree_k=0.0, support_frac=0.0, spread_k=0.0)),
     'analytic delta-method variance alone'),
    ('B_plus_tau2',   dict(learned_var=True,  sigma_terms=dict(disagree_k=0.0, support_frac=0.0, spread_k=0.0)),
     '+ the refiner\'s learned tau^2'),
    ('C_plus_disagree', dict(learned_var=True, sigma_terms=dict(support_frac=0.0, spread_k=0.0)),
     '+ how far arbitration moved the depth'),
    ('D_plus_support', dict(learned_var=True, sigma_terms=dict(spread_k=0.0)),
     '+ angular distance to the nearest anchor'),
    ('E_full_deployed', dict(learned_var=True, sigma_terms=None),
     '+ zone spread -- the deployed configuration'),
]


def spearman(x, y):
    """Rank correlation without scipy. Ties averaged, which matters at n < 20."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return float('nan')

    def rank(v):
        o = np.argsort(v, kind='mergesort')
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        # average tied ranks
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = np.mean(r[o[i:j + 1]])
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def score(points):
    """Uncertainty-quality metrics for one arm. points: list of (err, sigma)."""
    e = np.array([abs(p[0]) for p in points], float)
    s = np.array([p[1] for p in points], float)
    ok = np.isfinite(e) & np.isfinite(s) & (s > 0)
    e, s = e[ok], s[ok]
    n = len(e)
    if n == 0:
        return {'n': 0}
    nsig = e / s
    k1, k2 = int((nsig <= 1).sum()), int((nsig <= 2).sum())
    lo1, hi1 = wilson(k1, n)
    lo2, hi2 = wilson(k2, n)
    return {
        'n': n,
        'rank_corr': round(spearman(s, e), 4),
        'cov1': {'k': k1, 'frac': round(k1 / n, 4),
                 'wilson': [round(lo1, 3), round(hi1, 3)], 'target': 0.683},
        'cov2': {'k': k2, 'frac': round(k2 / n, 4),
                 'wilson': [round(lo2, 3), round(hi2, 3)], 'target': 0.954},
        'worst_nsig': round(float(nsig.max()), 3),
        'median_sigma': round(float(np.median(s)), 4),
        'median_abs_err': round(float(np.median(e)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='dir written by tape_capture.py')
    ap.add_argument('--calib', required=True)
    ap.add_argument('--backbone-engine', required=True)
    ap.add_argument('--residual-engine', default='')
    ap.add_argument('--gt', default='', help='defaults to <dir>/tape_gt.json')
    ap.add_argument('--validate-tol', type=float, default=0.15,
                    help='max acceptable median relative sigma disagreement between the '
                         'replayed deployed arm and the saved *_var.npy')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import cv2
    from ringfusion_perception import pipeline
    from ringfusion_perception.backbone import TensorRTBackbone
    from ringfusion_perception.rectify import FisheyeRectifier
    from ringfusion_perception.perception_node import load_calib
    from ringfusion_perception import roi as roi_mod

    gt_path = a.gt or os.path.join(a.dir, 'tape_gt.json')
    gt = json.load(open(gt_path))
    pts = gt['points']
    stems = sorted({p['stem'] for p in pts})
    print(f'{len(pts)} tape points across {len(stems)} frozen frames in {a.dir}')

    raw = load_calib(a.calib)
    r = raw['rectify']
    rect = FisheyeRectifier(raw['K'], raw['dist'], raw['model'],
                            size_in=(raw['img_w'], raw['img_h']),
                            size_out=(r['width'], r['height']),
                            balance=r['balance'], fov_scale=r['fov_scale'])
    calib = calib_from_yaml(a.calib, train_size=(r['height'], r['width']))

    backbone = TensorRTBackbone(a.backbone_engine)
    residual = None
    if a.residual_engine:
        from ringfusion_perception.residual import ResidualRefiner
        residual = ResidualRefiner(a.residual_engine)

    # depth is identical across arms, so accumulate per-arm sigma only
    per_arm = {name: [] for name, _, _ in ARMS}
    validation = []
    missing = []

    for stem in stems:
        rgb_p = os.path.join(a.dir, stem + '_rgb.png')
        tof_p = os.path.join(a.dir, stem + '_tof.npy')
        if not (os.path.exists(rgb_p) and os.path.exists(tof_p)):
            missing.append(stem)
            continue
        bgr = cv2.imread(rgb_p)
        if bgr is None:
            missing.append(stem)
            continue
        # tape_capture converted rgb8 -> BGR before imwrite, so imread round-trips it.
        # The pipeline wants RGB and the RECTIFIED frame; the capture stored the raw one.
        rgb = rect.rectify(np.ascontiguousarray(bgr[:, :, ::-1]))
        dist = np.load(tof_p).astype(np.float32)
        valid = np.isfinite(dist)

        frame_pts = [p for p in pts if p['stem'] == stem]
        saved_var_p = os.path.join(a.dir, stem + '_var.npy')
        saved_var = np.load(saved_var_p) if os.path.exists(saved_var_p) else None

        for name, kw, _ in ARMS:
            tracker = roi_mod.PlaneTracker()
            res = pipeline.run(rgb, dist, valid, calib, backbone, residual,
                               blend=True, roi_enable=True, plane_tracker=tracker, **kw)
            if not res['ok'] or res['var'] is None:
                continue
            sig = np.sqrt(np.maximum(res['var'], 0.0))
            dep = res['metric']
            for p in frame_pts:
                u, v = int(p['u']), int(p['v'])
                if not (0 <= u < sig.shape[1] and 0 <= v < sig.shape[0]):
                    continue
                # tape_gt stores SLANT range; tape_eval converts to axis depth the same way
                fx, fy, cx, cy = [float(x) for x in np.asarray(calib['K'], float).ravel()[:4]]
                th = np.arctan(np.hypot((u - cx) / fx, (v - cy) / fy))
                gt_z = float(p['range_m']) * np.cos(th)
                err = float(dep[v, u]) - gt_z
                per_arm[name].append((err, float(sig[v, u])))
                if name == 'E_full_deployed' and saved_var is not None:
                    s_saved = float(np.sqrt(max(saved_var[v, u], 0.0)))
                    if s_saved > 0:
                        validation.append(abs(float(sig[v, u]) - s_saved) / s_saved)

    report = {
        'label': 'r31 per-term uncertainty ablation, replayed offline from ' + os.path.basename(a.dir.rstrip('/')),
        'env': envinfo.capture([a.backbone_engine, a.residual_engine], note='sigma_ablation'),
        'capture_dir': os.path.abspath(a.dir),
        'n_points': len(pts), 'n_frames': len(stems), 'frames_missing_raw': missing,
        'arms': {},
    }
    for name, kw, desc in ARMS:
        report['arms'][name] = {'description': desc,
                                'config': {k: v for k, v in kw.items()},
                                **score(per_arm[name])}

    if validation:
        med = float(np.median(validation))
        report['replay_validation'] = {
            'n': len(validation),
            'median_relative_sigma_disagreement': round(med, 4),
            'tolerance': a.validate_tol,
            'passed': bool(med <= a.validate_tol),
            'note': ('Replayed deployed arm vs the saved *_var.npy. Replay cannot be '
                     'bit-identical -- four independent subscriptions were frozen together '
                     'and PlaneTracker re-converges from scratch -- so this quantifies the '
                     'gap rather than assuming it away.')}
    else:
        report['replay_validation'] = {'n': 0, 'passed': None,
                                       'note': 'no saved *_var.npy to compare against'}

    print(f"\n  {'arm':<18}{'n':>4}{'rank r':>9}{'cov@1s':>9}{'cov@2s':>9}{'worst':>8}  description")
    for name, _, desc in ARMS:
        s = report['arms'][name]
        if not s.get('n'):
            print(f'  {name:<18}   -- no data'); continue
        print(f"  {name:<18}{s['n']:>4}{s['rank_corr']:>9.3f}{s['cov1']['frac']:>9.3f}"
              f"{s['cov2']['frac']:>9.3f}{s['worst_nsig']:>8.2f}  {desc}")
    print(f"\n  targets: cov@1s 0.683, cov@2s 0.954. Wilson intervals are in the JSON --")
    print(f"  at this n they are wide, and differences between arms will mostly not be")
    print(f"  resolvable. That is a finding, not a defect of the ablation.")

    v = report['replay_validation']
    if v.get('n'):
        verdict = 'PASS' if v['passed'] else '** FAIL -- do not trust the arms above **'
        print(f"\n  replay validation: median relative sigma disagreement "
              f"{v['median_relative_sigma_disagreement']:.3f} vs tol {v['tolerance']:.2f}  {verdict}")
    if missing:
        print(f"\n  {len(missing)} frames had no raw inputs and were skipped: {missing[:6]}")

    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()

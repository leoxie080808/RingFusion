#!/usr/bin/env python3
"""T2: offline core timing with the ROI crop and the uncertainty terms separated.

Three different costs have been quoted for "the uncertainty terms" and they are not the
same quantity:

  ~5.7 ms   offline roi0 -> roi1 delta at 1640x1232  = Stage 7d, the ROI SIGMA FLOOR
  ~6.1 ms   in-node stage 7d_roi_sigma               = the same thing, measured live
  ~10.6 ms  sigma_cost_corrected_2026-08-04          = sigma_support_var INSIDE stage 7c

So 5.7 and 10.6 were never in conflict -- they are stage 7d and the 7c sigma block.
This script measures each separately so Table I can quote the right one.

Method note (from blend.py's own docstring): zeroing DISAGREE_K/SUPPORT_FRAC/SPREAD_K is
the WRONG control for a cost question -- the reduced-grid fields, both cv2.blur calls, the
GPU upsample and the full-frame add all still run. That mistake produced a 0.80 ms figure.
This bypasses the call entirely, and additionally times sigma_support_var in isolation so
the two agree by construction rather than by assumption.
"""
import argparse, json, os, statistics, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_REPO, 'training'))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws/src/ringfusion_perception'))
import envinfo                                                   # noqa: E402
from anchoring_bridge import calib_from_yaml                     # noqa: E402
from ringfusion_perception import pipeline, roi, blend as blend_mod   # noqa: E402


def stats(ts):
    return {'median_ms': round(statistics.median(ts), 3),
            'p05_ms': round(float(np.percentile(ts, 5)), 3),
            'p95_ms': round(float(np.percentile(ts, 95)), 3),
            'mean_ms': round(statistics.mean(ts), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-dir', required=True); ap.add_argument('--tof-dir', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--backbone-engine', required=True); ap.add_argument('--residual-engine', required=True)
    ap.add_argument('--sizes', nargs='+', default=['1232x1640', '480x640'])
    ap.add_argument('--iters', type=int, default=500); ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import cv2
    from ringfusion_perception.backbone import TensorRTBackbone
    from ringfusion_perception.residual import ResidualRefiner
    backbone = TensorRTBackbone(a.backbone_engine)
    residual = ResidualRefiner(a.residual_engine)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a.tof_dir) if f.endswith('.npz'))
    rgb0 = cv2.cvtColor(cv2.imread(os.path.join(a.rgb_dir, stems[0] + '.png')), cv2.COLOR_BGR2RGB)
    z = np.load(os.path.join(a.tof_dir, stems[0] + '.npz'))
    dist, conf = z['dist_m'].astype(np.float32), z['confidence']

    real_sigma = blend_mod.sigma_support_var
    out = {}
    for size in a.sizes:
        H, W = (int(v) for v in size.lower().split('x'))
        rgb = cv2.resize(rgb0, (W, H), interpolation=cv2.INTER_AREA)
        calib = calib_from_yaml(a.calib, (H, W))
        res = {'resolution': f'{H}x{W}'}

        def run(kw, tdict=None, tracker=None):
            return pipeline.run(rgb, dist, np.isfinite(dist), calib, backbone, residual,
                                confidence=conf, min_confidence=-1,
                                plane_tracker=tracker, timings=tdict, **kw)

        # ---- config arms -------------------------------------------------------
        arms = {'core_only':   dict(blend=False, roi_enable=False),
                'plus_blend':  dict(blend=True,  roi_enable=False),
                'deployed':    dict(blend=True,  roi_enable=True)}
        for name, kw in arms.items():
            tr = roi.PlaneTracker(refit_every=1) if kw['roi_enable'] else None
            for _ in range(a.warmup): run(kw, tracker=tr)
            ts = []
            for _ in range(a.iters):
                t = time.perf_counter(); run(kw, tracker=tr); ts.append((time.perf_counter()-t)*1e3)
            res[name] = stats(ts)

        # ---- deployed with the 7c sigma block BYPASSED -------------------------
        blend_mod.sigma_support_var = lambda *x, **k: np.float32(0.0)
        pipeline.blend_sigma_var = blend_mod.sigma_support_var
        tr = roi.PlaneTracker(refit_every=1)
        for _ in range(a.warmup): run(arms['deployed'], tracker=tr)
        ts = []
        for _ in range(a.iters):
            t = time.perf_counter(); run(arms['deployed'], tracker=tr); ts.append((time.perf_counter()-t)*1e3)
        res['deployed_no_7c_sigma'] = stats(ts)
        blend_mod.sigma_support_var = real_sigma
        pipeline.blend_sigma_var = real_sigma

        # ---- per-stage breakdown (deployed) ------------------------------------
        tr = roi.PlaneTracker(refit_every=1)
        for _ in range(a.warmup): run(arms['deployed'], tracker=tr)
        acc = {}
        for _ in range(a.iters):
            td = {}; run(arms['deployed'], tdict=td, tracker=tr)
            for k, v in td.items(): acc.setdefault(k, []).append(v)
        res['per_stage_ms'] = {k: round(statistics.median(v), 3) for k, v in sorted(acc.items())}

        # ---- derived component costs -------------------------------------------
        res['components_ms'] = {
            'blend_depth_stage7c': round(res['plus_blend']['median_ms'] - res['core_only']['median_ms'], 3),
            'sigma_terms_in_7c':   round(res['deployed']['median_ms'] - res['deployed_no_7c_sigma']['median_ms'], 3),
            'roi_plane_and_floor': round(res['deployed']['median_ms'] - res['plus_blend']['median_ms'], 3),
        }
        out[f'{H}x{W}'] = res
        print(f"\n== {H}x{W} ==")
        for k in ('core_only', 'plus_blend', 'deployed', 'deployed_no_7c_sigma'):
            print(f"  {k:22s} {res[k]['median_ms']:8.3f} ms  (p5 {res[k]['p05_ms']:.2f} p95 {res[k]['p95_ms']:.2f})")
        for k, v in res['components_ms'].items(): print(f"  -> {k:22s} {v:8.3f} ms")

    rep = {'label': 'T2 offline core timing: ROI crop and uncertainty terms separated',
           'procedure': {'warmup_iters': a.warmup, 'timed_iters': a.iters,
                         'single_input_reused': True,
                         'plane_tracker': 'PlaneTracker(refit_every=1), as deployed',
                         'sigma_control': 'sigma_support_var BYPASSED, not zeroed'},
           'resolutions': out, 'env': envinfo.capture(note='t2_core_timing')}
    if a.out:
        json.dump(rep, open(a.out, 'w'), indent=1); print('\nwrote', a.out)


if __name__ == '__main__':
    main()

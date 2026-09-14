#!/usr/bin/env python3
"""Repeat captures of a static tape scene, and the interval across them.

Two repeats of a MOTIONLESS scene two minutes apart differ in rank correlation by 0.13
from pipeline noise alone. Any single capture therefore cannot be read more finely than
that, which is why the LIVE-4 result was reported as a mean over five repeats -- but that
mean and its bounds were assembled by hand outside any script, so they cannot be
regenerated or checked. This does both halves: takes the repeats, and aggregates them.

WHAT A REPEAT IS HERE. The scene is static and the camera does not move, so the marker
pixels are the same in every repeat -- the operator does NOT re-click twenty markers five
times. Click once with tape_capture.py, then run this to latch N further snapshots of
/depth and /depth_var (plus /image and /tof, so the same repeats can drive the offline
sigma ablation). Each repeat is scored at the pixels already recorded in tape_gt.json.

TWO DIFFERENT INTERVALS, AND THEY ARE NOT INTERCHANGEABLE. The spread reported here is
across repeats of the same points: it measures PIPELINE NOISE on a fixed scene. It says
nothing about how well this many points pin down coverage for the population of points we
could have placed -- that is the binomial interval, it is much wider, and tape_stats_r31
computes it. A claim that coverage sits near its target needs the binomial one; a claim
that two configurations differ needs this one.

    # capture 5 repeats, 20 s apart, into an existing capture dir
    python3 tools/diagnostics/tape_repeats.py --dir <dir> --repeats 5 --interval 20

    # score them
    python3 tools/diagnostics/tape_repeats.py --dir <dir> --score-only \\
        --calib ros2_ws/src/ringfusion_bringup/config/calibration.yaml \\
        --out docs/demo/benchmarks/tape_repeats_r31.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anchoring_bridge import calib_from_yaml                      # noqa: E402
from sigma_ablation import score                                  # noqa: E402
import envinfo                                                    # noqa: E402


def capture(a):
    import cv2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from ringfusion_msgs.msg import ToFFrame

    class Rep(Node):
        def __init__(self):
            super().__init__('tape_repeats')
            self.rgb = self.depth = self.var = self.tof = None
            self.create_subscription(Image, a.image_topic, self.on_img, 5)
            self.create_subscription(Image, a.depth_topic, self.on_depth, 5)
            self.create_subscription(Image, a.var_topic, self.on_var, 5)
            self.create_subscription(ToFFrame, a.tof_topic, self.on_tof, 5)

        def on_img(self, m):
            b = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
            self.rgb = b[:, :, ::-1].copy() if m.encoding == 'rgb8' else b.copy()

        def on_depth(self, m):
            self.depth = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()

        def on_var(self, m):
            self.var = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()

        def on_tof(self, m):
            if len(m.dist_m) == m.rows * m.cols:
                self.tof = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols).copy()

    rclpy.init()
    n = Rep()
    t0 = time.time()
    while time.time() - t0 < 15 and (n.depth is None or n.rgb is None):
        rclpy.spin_once(n, timeout_sec=0.2)
    if n.depth is None:
        n.destroy_node(); rclpy.shutdown()
        sys.exit('no /depth -- is the perception stack running?')

    os.makedirs(a.dir, exist_ok=True)
    existing = len(glob.glob(os.path.join(a.dir, 'rep*_depth.npy')))
    for k in range(existing, existing + a.repeats):
        # settle: spin for the interval so each repeat is an independent draw of pipeline
        # noise rather than the same cached frame written N times
        t1 = time.time()
        while time.time() - t1 < (a.interval if k > existing else 2.0):
            rclpy.spin_once(n, timeout_sec=0.2)
        p = os.path.join(a.dir, f'rep{k:02d}')
        np.save(p + '_depth.npy', n.depth)
        if n.var is not None:
            np.save(p + '_var.npy', n.var)
        if n.tof is not None:
            np.save(p + '_tof.npy', n.tof)
        if n.rgb is not None:
            cv2.imwrite(p + '_rgb.png', n.rgb)
        print(f'  wrote rep{k:02d}  (depth median {np.nanmedian(n.depth):.3f} m)')
    n.destroy_node()
    rclpy.shutdown()
    print(f'{a.repeats} repeats written to {a.dir}')


def score_only(a):
    gt_path = a.gt or os.path.join(a.dir, 'tape_gt.json')
    meta = json.load(open(gt_path))
    pts = meta['points']
    origin = float(meta.get('origin_offset_m', 0.0))
    calib = calib_from_yaml(a.calib, train_size=(1232, 1640))
    fx, fy, cx, cy = [float(x) for x in np.asarray(calib['K'], float).ravel()[:4]]

    reps = sorted(glob.glob(os.path.join(a.dir, 'rep*_depth.npy')))
    if not reps:
        sys.exit(f'no rep*_depth.npy in {a.dir} -- run the capture mode first')

    per_rep = []
    for dp in reps:
        tag = os.path.basename(dp).replace('_depth.npy', '')
        vp = os.path.join(a.dir, tag + '_var.npy')
        if not os.path.exists(vp):
            print(f'  {tag}: no _var.npy, skipped'); continue
        depth, var = np.load(dp), np.load(vp)
        obs = []
        for p in pts:
            u, v = int(p['u']), int(p['v'])
            if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                continue
            th = np.arctan(np.hypot((u - cx) / fx, (v - cy) / fy))
            gt_z = (float(p['range_m']) + origin) * np.cos(th)   # slant -> axial
            obs.append((float(depth[v, u]) - gt_z, float(np.sqrt(max(var[v, u], 0.0)))))
        s = score(obs)
        s['repeat'] = tag
        per_rep.append(s)
        print(f"  {tag}: n={s['n']} rank r={s['rank_corr']:.3f} "
              f"cov@1s={s['cov1']['frac']:.3f} worst={s['worst_nsig']:.2f}")

    def agg(key, get):
        v = np.array([get(s) for s in per_rep if s.get('n')], float)
        v = v[np.isfinite(v)]
        if not v.size:
            return None
        return {'mean': round(float(v.mean()), 4), 'lo': round(float(v.min()), 4),
                'hi': round(float(v.max()), 4), 'sd': round(float(v.std(ddof=1)) if v.size > 1 else 0.0, 4),
                'n_repeats': int(v.size)}

    report = {
        'label': 'r31 tape repeats, spread across repeats of a static scene',
        'env': envinfo.capture(note='tape_repeats'),
        'capture_dir': os.path.abspath(a.dir),
        'n_points': len(pts), 'n_repeats': len(per_rep),
        'across_repeats': {
            'rank_corr': agg('rank_corr', lambda s: s['rank_corr']),
            'cov1': agg('cov1', lambda s: s['cov1']['frac']),
            'cov2': agg('cov2', lambda s: s['cov2']['frac']),
            'worst_nsig': agg('worst', lambda s: s['worst_nsig']),
            'median_abs_err': agg('medae', lambda s: s['median_abs_err']),
        },
        'per_repeat': per_rep,
        'interval_meaning': ('spread across repeats of the SAME points, i.e. pipeline noise '
                             'on a static scene. NOT the sampling uncertainty from having '
                             'this many points -- that is the binomial interval in '
                             'tape_stats_r31.json and it is much wider.'),
    }

    print('\n  across repeats:')
    for k, v in report['across_repeats'].items():
        if v:
            print(f"    {k:<16} mean {v['mean']:.4f}  range [{v['lo']:.4f}, {v['hi']:.4f}]  "
                  f"sd {v['sd']:.4f}  (n={v['n_repeats']})")
    if len(per_rep) < 3:
        print(f'\n  NOTE: only {len(per_rep)} repeats. Five was the basis for the published '
              f'+/-0.05 noise figure; fewer makes the range unreliable.')

    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'\nwrote {a.out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--interval', type=float, default=20.0,
                    help='seconds between repeats; the published noise figure used ~2 min')
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--calib', default='')
    ap.add_argument('--gt', default='')
    ap.add_argument('--image-topic', default='/image')
    ap.add_argument('--depth-topic', default='/depth')
    ap.add_argument('--var-topic', default='/depth_var')
    ap.add_argument('--tof-topic', default='/tof')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.score_only:
        if not a.calib:
            sys.exit('--score-only needs --calib')
        score_only(a)
    else:
        capture(a)


if __name__ == '__main__':
    main()

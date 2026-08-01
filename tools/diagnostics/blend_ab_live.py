#!/usr/bin/env python3
"""Live A/B of Stage 7c (the ToF/network blend), scored against held-out ToF zones.

WHY THIS DOES NOT NEED TWO DRIVES. The obvious protocol is to drive a route with blend on,
drive it again with blend off, and compare. That comparison is confounded: no two drives see
the same scene, and depth error depends far more on what is in front of the robot than on the
blend. Here both arms are computed from ONE backbone + residual pass on the SAME frame, so
every difference is the blend and nothing else. Drive once, however you like.

THE TRAP THIS AVOIDS. The blend pulls the depth map toward nearby ToF readings. If it is
allowed to see the zones we then score against, it wins trivially and the result means
nothing -- it would be reporting how well ToF depth predicts ToF depth. So the zones are
split first: a central 16x16 island ANCHORS the fit and feeds the blend, and everything
outside it is held out and used only for scoring. The blend never sees a scored zone.

That split is also the deployment-relevant question. `center` asks whether the map is right
where the ToF is NOT, which is 93% of the frame and the whole reason a camera is involved.

Reports, per arm:
  * accuracy at held-out zones, binned by angular distance from the ToF island
  * TEMPORAL stability -- frame-to-frame change under motion, which a static scene cannot
    reveal and which is what a downstream consumer actually feels
  * far-field behaviour (max depth, fraction beyond the ToF's range)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as M                                                    # noqa: E402
from baselines import project, split_zones, MIN_RANGE, MAX_RANGE       # noqa: E402
from ringfusion_perception import anchoring as anc                     # noqa: E402
from ringfusion_perception.blend import blend_depth                    # noqa: E402
from ringfusion_perception.pipeline import splat_anchors               # noqa: E402
from ringfusion_perception.rectify import FisheyeRectifier             # noqa: E402
from ringfusion_perception.perception_node import load_calib           # noqa: E402

WARMUP = 5


class BlendAB(Node):
    def __init__(self, a):
        super().__init__('blend_ab_live')
        self.a = a
        raw = load_calib(a.calib)
        r = raw['rectify']
        self.rect = FisheyeRectifier(raw['K'], raw['dist'], raw['model'],
                                     size_in=(raw['img_w'], raw['img_h']),
                                     size_out=(r['width'], r['height']),
                                     balance=r['balance'], fov_scale=r['fov_scale'])
        self.calib = dict(raw)
        self.calib['K'] = self.rect.K_rect
        self.calib['model'] = 'pinhole'
        self.calib['dist'] = np.zeros(4)
        from ringfusion_perception.backbone import TensorRTBackbone
        from ringfusion_perception.residual import ResidualRefiner
        self.backbone = TensorRTBackbone(a.backbone_engine)
        self.residual = ResidualRefiner(a.residual_engine) if a.residual_engine else None
        self.fx = float(np.asarray(self.calib['K']).ravel()[0])
        self.img_raw = None
        self.n = 0
        self.rows = {'A_noblend': {'p': [], 'g': [], 'ang': []},
                     'B_blend': {'p': [], 'g': [], 'ang': []}}
        self.prev = {}
        self.jump = {'A_noblend': [], 'B_blend': []}
        self.far = {'A_noblend': [], 'B_blend': []}
        self.create_subscription(Image, '/image', self.on_image, 1)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 1)
        print(f'warming up {WARMUP} frames, then DRIVE. Target {a.frames} frames.',
              flush=True)

    def on_image(self, m):
        if m.encoding == 'rgb8':
            self.img_raw = np.frombuffer(m.data, np.uint8).reshape(
                m.height, m.width, 3).copy()

    def on_tof(self, m):
        if self.img_raw is None:
            return
        rgb = self.rect.rectify(self.img_raw)
        h, w = rgb.shape[:2]
        d = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols)
        valid = np.isfinite(d) & (d >= MIN_RANGE) & (d <= MAX_RANGE)
        # Island anchors the fit AND feeds the blend; everything else is scoring-only.
        fit_z, hold_z = split_zones(valid, 'center', np.random.default_rng(self.n))
        if fit_z.sum() < 16 or hold_z.sum() < 8:
            return

        disp = self.backbone.infer(rgb)
        u, v, z, _ = project(d, fit_z, self.calib, h, w)
        uh, vh, zh, _ = project(d, hold_z, self.calib, h, w)
        if u.size < 16 or uh.size < 8:
            return
        fit = anc.solve_robust(disp[v, u].astype(np.float64), 1.0 / z,
                               np.ones_like(z), iters=1)
        if fit is None:
            return
        a_, b_ = fit
        metric = anc.to_metric_depth(disp, a_, b_).astype(np.float32)

        inb = np.zeros(m.rows * m.cols, bool)
        ad = np.zeros((h, w), np.float32)
        am = np.zeros((h, w), np.float32)
        ad[v, u] = z.astype(np.float32)      # FIT anchors only -- never the held-out ones
        am[v, u] = 1.0
        if self.residual is not None:
            metric, _ = self.residual.refine(rgb, metric, disp, ad, am, a_, b_)
            metric = np.asarray(metric, np.float32)
        metric = np.clip(metric, None, 20.0)

        blended, _ = blend_depth(metric, ad, am, fx=self.fx)

        self.n += 1
        if self.n <= WARMUP:
            return

        # angular distance of each held-out zone from the island centre, in degrees
        cx, cy = np.asarray(self.calib['K']).ravel()[2:4]
        ang = np.degrees(np.arctan2(
            np.hypot(uh - cx, vh - cy), self.fx))
        for name, D in (('A_noblend', metric), ('B_blend', blended)):
            self.rows[name]['p'].append(D[vh, uh].astype(np.float64))
            self.rows[name]['g'].append(zh.astype(np.float64))
            self.rows[name]['ang'].append(ang)
            self.far[name].append((float(np.nanmax(D)),
                                   float(np.mean(D > MAX_RANGE))))
            # temporal: how much did this arm move between consecutive frames?
            sub = D[::16, ::16]
            if name in self.prev:
                self.jump[name].append(float(np.nanmedian(np.abs(sub - self.prev[name]))))
            self.prev[name] = sub

        if (self.n - WARMUP) % 25 == 0:
            print(f'  {self.n - WARMUP}/{self.a.frames} frames', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--backbone-engine', required=True)
    p.add_argument('--residual-engine', default='')
    p.add_argument('--frames', type=int, default=250)
    p.add_argument('--secs', type=float, default=180.0)
    p.add_argument('--out', default='')
    a = p.parse_args()

    rclpy.init()
    n = BlendAB(a)
    end = time.time() + a.secs
    while rclpy.ok() and (n.n - WARMUP) < a.frames and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)

    got = n.n - WARMUP
    if got < 5:
        print(f'only {got} frames -- is the stack running and the ToF returning?')
        rclpy.shutdown()
        return

    print(f'\n=== Stage 7c blend A/B, {got} frames, scored on HELD-OUT ToF zones ===')
    print('Both arms share one backbone+residual pass per frame, so every difference is')
    print('the blend. The blend saw only the central island; these zones are outside it.\n')
    rows, report = [], {}
    for name in ('A_noblend', 'B_blend'):
        pr = np.concatenate(n.rows[name]['p'])
        gt = np.concatenate(n.rows[name]['g'])
        mm = M.depth_metrics(pr, gt)
        rows.append((name, mm))
        report[name] = mm
    print(M.format_table(rows))

    # binned by angle -- the blend can only help near its anchors, so the interesting
    # question is how far out its influence reaches before the two arms converge
    EDGES = [0, 3, 6, 10, 15, 30, 90]
    print(f'\nmedAE ↓ by angular distance from the ToF island:')
    hdr = f'{"arm":<12}' + ''.join(f'{f"{lo}-{hi}°":>11}' for lo, hi in zip(EDGES, EDGES[1:]))
    print(hdr); print('-' * len(hdr))
    binned = {}
    for name in ('A_noblend', 'B_blend'):
        pr = np.concatenate(n.rows[name]['p']); gt = np.concatenate(n.rows[name]['g'])
        an = np.concatenate(n.rows[name]['ang'])
        line = f'{name:<12}'; b = []
        for lo, hi in zip(EDGES, EDGES[1:]):
            sel = (an >= lo) & (an < hi) & np.isfinite(pr)
            v = float(np.median(np.abs(pr[sel] - gt[sel]))) if sel.sum() > 30 else float('nan')
            b.append(v)
            line += (f'{v:>10.3f}m' if np.isfinite(v) else f'{"-":>11}')
        print(line); binned[name] = b
    print('\n(blend wins where it has anchors; the arms should converge as angle grows —')
    print(' if B is WORSE far out, the blend is leaking near-field depth into the far field)')

    print(f'\ntemporal stability under motion (median frame-to-frame |Δdepth| ↓):')
    for name in ('A_noblend', 'B_blend'):
        j = np.array(n.jump[name])
        if j.size:
            print(f'  {name:<12} median {np.median(j):.4f} m   p90 {np.percentile(j,90):.4f} m')
    print(f'\nfar field ↓:')
    for name in ('A_noblend', 'B_blend'):
        f = np.array(n.far[name])
        print(f'  {name:<12} max depth median {np.median(f[:,0]):.2f} m   '
              f'fraction beyond ToF range {np.median(f[:,1])*100:.2f}%')

    if a.out:
        json.dump({'frames': got, 'metrics': report, 'binned_medae': binned,
                   'edges_deg': EDGES,
                   'jump': {k: float(np.median(v)) for k, v in n.jump.items() if v}},
                  open(a.out, 'w'), indent=1)
        print(f'\nwrote {a.out}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()

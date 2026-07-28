#!/usr/bin/env python3
"""Moving-robot A/B: run BOTH residual engines on the same live frames while the robot moves.

Static-scene stability is trivially easy -- nothing changes, so nothing can flicker. This
drives both residuals off one backbone pass per frame over a motion sequence and reports:

  * per-frame quality (far-field blowup, banding, anchor agreement)
  * TEMPORAL behaviour under motion (frame-to-frame jump, worst-case spikes)

Because both arms see byte-identical input every frame, any difference is the residual.
"""
import argparse
import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

from ringfusion_perception import geometry as geo
from ringfusion_perception import anchoring as anc
from ringfusion_perception import gpu_ops
from ringfusion_perception.pipeline import splat_anchors
from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.perception_node import load_calib


class MovingAB(Node):
    def __init__(self, a):
        super().__init__('moving_ab')
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
        from ringfusion_perception.backbone import TensorRTBackbone
        from ringfusion_perception.residual import ResidualRefiner
        self.backbone = TensorRTBackbone(a.backbone_engine)
        self.res = {'v1': ResidualRefiner(a.engine_v1), 'v2': ResidualRefiner(a.engine_v2)}
        self.img = self.tof = None
        self.rec = []
        self.prev = {}
        self.t0 = None
        self.warm = 0
        self.shots = 0
        self.create_subscription(Image, '/image', self.on_image, 5)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 10)
        self.create_timer(1.0 / a.hz, self.tick)
        print(f"warming up... move the robot when you see 'RECORDING'", flush=True)

    def on_image(self, m):
        if m.encoding == 'rgb8':
            rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.img = self.rect.rectify(rgb[:, :, ::-1])

    def on_tof(self, m):
        self.tof = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols)

    def tick(self):
        if self.img is None or self.tof is None:
            return
        self.warm += 1
        if self.warm < 5:
            return
        if self.t0 is None:
            self.t0 = time.time()
            print(">>> RECORDING — move the robot now <<<", flush=True)
        el = time.time() - self.t0
        if el > self.a.secs:
            raise SystemExit(0)

        rgb_bgr = self.img
        rgb = rgb_bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        td = self.tof
        tv = np.isfinite(td)
        rows, cols = td.shape
        disp = self.backbone.infer(rgb)
        proj = geo.project_zone_to_pixel(td, tv, cols, rows, self.calib['fov_h'],
                                         self.calib['fov_v'], self.calib['T_cam_tof'],
                                         self.calib['K'], self.calib['dist'], model='pinhole')
        uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']
        fin = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
        u = np.round(np.where(fin, uv[:, 0], -1)).astype(int)
        v = np.round(np.where(fin, uv[:, 1], -1)).astype(int)
        inb = ok & fin & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
        if int(inb.sum()) < 4:
            return
        disp_at = disp[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)][inb]
        fit = anc.solve_robust(disp_at, 1.0 / z[inb], np.ones(int(inb.sum())), iters=1)
        if fit is None:
            return
        a_, b_ = fit
        D0 = gpu_ops.to_metric_depth(disp, a_, b_)
        ad, am = splat_anchors(u, v, z, inb, (h, w))
        ys, xs = np.nonzero(am > 0)

        out = {'t': el, 'anchors': int(inb.sum()), 'tof_max': float(np.nanmax(td)),
               'img_mean': float(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).mean())}
        maps = {'A': D0}
        for nm, eng in self.res.items():
            maps[nm], _ = eng.refine(rgb, D0, disp, ad, am, a_, b_)
        for nm, D in maps.items():
            m = D > 0
            s = D[::4, ::4]
            e = np.abs(D[ys, xs] - ad[ys, xs])
            d = {'median': float(np.median(D[m])), 'max': float(D[m].max()),
                 'p99': float(np.percentile(D[m], 99)),
                 'frac_gt3': float(np.mean(D[m] > 3)),
                 'frac_gt5': float(np.mean(D[m] > 5)),
                 'clamp': int(D[m].max() >= 19.9),
                 'anchor_mae': float(np.mean(e))}
            if nm in self.prev:
                p = self.prev[nm]
                mm = (p > 0) & (s > 0)
                if mm.sum():
                    d['jump'] = float(np.mean(np.abs(s[mm] - p[mm])))
                    d['jump_rel'] = float(np.mean(np.abs(s[mm] - p[mm]) / np.maximum(p[mm], 1e-3)))
            self.prev[nm] = s
            out[nm] = d
        self.rec.append(out)

        if self.a.save_prefix and self.shots < 3 and el > (self.shots + 1) * self.a.secs / 4:
            lo, hi = np.percentile(D0[D0 > 0], [2, 98])

            def col(X):
                n = np.clip((X - lo) / max(1e-6, hi - lo), 0, 1)
                c = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                c[X <= 0] = 0
                return c
            p = f"{self.a.save_prefix}_t{self.shots}"
            cv2.imwrite(f"{p}_cam.png", rgb_bgr)
            for nm, D in maps.items():
                cv2.imwrite(f"{p}_{nm}.png", col(D))
            self.shots += 1

        if len(self.rec) % 10 == 0:
            print(f"  t={el:4.1f}s  n={len(self.rec):3d}  "
                  f"v1 max {out['v1']['max']:6.2f}  v2 max {out['v2']['max']:6.2f}", flush=True)


def summarize(rec, out):
    res = {'frames': len(rec), 'duration_s': round(rec[-1]['t'], 2),
           'rate_hz': round(len(rec) / max(rec[-1]['t'], 1e-9), 2),
           'anchors_mean': round(float(np.mean([r['anchors'] for r in rec])), 1),
           'tof_max_over_run': round(float(np.max([r['tof_max'] for r in rec])), 2),
           'img_mean': round(float(np.mean([r['img_mean'] for r in rec])), 1)}
    for nm in ('A', 'v1', 'v2'):
        g = lambda k: np.array([r[nm][k] for r in rec if k in r[nm]])
        res[nm] = {
            'median_depth_mean': round(float(g('median').mean()), 4),
            'max_mean': round(float(g('max').mean()), 3),
            'max_worst': round(float(g('max').max()), 3),
            'p99_mean': round(float(g('p99').mean()), 3),
            'p99_worst': round(float(g('p99').max()), 3),
            'frac_gt3_mean_pct': round(float(g('frac_gt3').mean() * 100), 3),
            'frac_gt5_mean_pct': round(float(g('frac_gt5').mean() * 100), 3),
            'frames_at_clamp': int(g('clamp').sum()),
            'anchor_mae_mean': round(float(g('anchor_mae').mean()), 4),
            'jump_mean_m': round(float(g('jump').mean()), 4) if len(g('jump')) else None,
            'jump_worst_m': round(float(g('jump').max()), 4) if len(g('jump')) else None,
            'jump_rel_pct': round(float(g('jump_rel').mean() * 100), 2) if len(g('jump_rel')) else None,
        }
    json.dump({'summary': res, 'per_frame': rec}, open(out, 'w'), indent=2)
    print()
    print(f"MOVING RUN: {res['frames']} frames over {res['duration_s']}s "
          f"({res['rate_hz']} Hz), ToF max seen {res['tof_max_over_run']} m, "
          f"brightness {res['img_mean']}")
    print()
    ks = ['median_depth_mean', 'max_mean', 'max_worst', 'p99_mean', 'p99_worst',
          'frac_gt3_mean_pct', 'frac_gt5_mean_pct', 'frames_at_clamp', 'anchor_mae_mean',
          'jump_mean_m', 'jump_worst_m', 'jump_rel_pct']
    print(f"{'metric':<22}{'A':>12}{'v1':>12}{'v2':>12}")
    print('-' * 58)
    for k in ks:
        row = ''.join(f"{res[n][k]:>12}" if res[n][k] is not None else f"{'-':>12}"
                      for n in ('A', 'v1', 'v2'))
        print(f"{k:<22}{row}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--backbone-engine', required=True)
    p.add_argument('--engine-v1', required=True)
    p.add_argument('--engine-v2', required=True)
    p.add_argument('--secs', type=float, default=20.0)
    p.add_argument('--hz', type=float, default=6.0)
    p.add_argument('--out', required=True)
    p.add_argument('--save-prefix', default='')
    a = p.parse_args()
    rclpy.init()
    n = MovingAB(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if len(n.rec) > 2:
            summarize(n.rec, a.out)
        else:
            print('not enough frames')
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

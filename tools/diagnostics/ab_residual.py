#!/usr/bin/env python3
"""Controlled A/B of two residual engines on the SAME live frames.

One backbone pass per frame feeds both residuals, so Network A's output, the anchors and
the affine fit are byte-identical between the two arms -- every difference is the residual.
"""
import argparse
import json

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


def band_power(field, period, region):
    """Power of `field` at horizontal `period` px, normalised by its broadband median."""
    r = field[region].astype(np.float64)
    r = r - r.mean(axis=1, keepdims=True)
    F = np.abs(np.fft.rfft(r, axis=1)).mean(0)
    f = np.fft.rfftfreq(r.shape[1])
    per = 1.0 / np.maximum(f, 1e-9)
    i = np.argmin(np.abs(per - period))
    return float(F[i] / np.median(F[3:]))


class AB(Node):
    def __init__(self, a):
        super().__init__('ab_residual')
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
        self.res1 = ResidualRefiner(a.engine_v1)
        self.res2 = ResidualRefiner(a.engine_v2)
        self.img = self.tof = None
        self.n = 0
        self.acc = []
        self.create_subscription(Image, '/image', self.on_image, 5)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 10)
        self.create_timer(0.35, self.tick)

    def on_image(self, m):
        if m.encoding == 'rgb8':
            rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.img = self.rect.rectify(rgb[:, :, ::-1])

    def on_tof(self, m):
        self.tof = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols)

    def tick(self):
        if self.img is None or self.tof is None:
            return
        self.n += 1
        if self.n < 4:
            return
        rgb_bgr = self.img
        rgb = rgb_bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        K = self.calib['K']
        td = self.tof
        tv = np.isfinite(td)
        rows, cols = td.shape
        disp = self.backbone.infer(rgb)
        proj = geo.project_zone_to_pixel(td, tv, cols, rows, self.calib['fov_h'],
                                         self.calib['fov_v'], self.calib['T_cam_tof'],
                                         K, self.calib['dist'], model='pinhole')
        uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']
        fin = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
        u = np.round(np.where(fin, uv[:, 0], -1)).astype(int)
        v = np.round(np.where(fin, uv[:, 1], -1)).astype(int)
        inb = ok & fin & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
        if int(inb.sum()) < 4:
            return
        disp_at = disp[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)][inb]
        a_, b_ = anc.solve_robust(disp_at, 1.0 / z[inb], np.ones(int(inb.sum())), iters=1)
        D0 = gpu_ops.to_metric_depth(disp, a_, b_)
        ad, am = splat_anchors(u, v, z, inb, (h, w))
        D1, t1 = self.res1.refine(rgb, D0, disp, ad, am, a_, b_)
        D2, t2 = self.res2.refine(rgb, D0, disp, ad, am, a_, b_)

        ys, xs = np.nonzero(am > 0)
        box = (slice(ys.min(), ys.max()), slice(xs.min(), xs.max()))
        reg = (slice(370, 590), slice(600, 1120))
        rec = {'anchors': int(inb.sum()), 'a': float(a_), 'b': float(b_),
               'tof_max': float(np.nanmax(td)), 'tof_ctr': float(np.nanmedian(td[14:18, 14:18])),
               'img_mean': float(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).mean())}
        for nm, D, t2v in (('A', D0, None), ('v1', D1, t1), ('v2', D2, t2)):
            m = D > 0
            e = np.abs(D[ys, xs] - ad[ys, xs])          # consistency with its own anchors
            rec[nm] = {
                'median': float(np.median(D[m])),
                'max': float(D[m].max()),
                'frac_gt3': float(np.mean(D[m] > 3)),
                'frac_gt5': float(np.mean(D[m] > 5)),
                'frac_clamp': float(np.mean(D[m] >= 19.9)),
                'ctr': float(np.median(D[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8])),
                'anchor_mae': float(np.mean(e)),
                'anchor_p90': float(np.percentile(e, 90)),
                'band173': band_power(D, 173, reg),
                'band13': band_power(D, 13, reg),
                'box_median': float(np.median(D[box])),
                'mean_abs_dev_from_A': float(np.mean(np.abs(D[m] - D0[m]))),
            }
            if t2v is not None:
                s = np.sqrt(np.clip(t2v, 0, None))
                rec[nm]['sigma_med'] = float(np.median(s))
                rec[nm]['sigma_p5'] = float(np.percentile(s, 5))
                rec[nm]['sigma_p95'] = float(np.percentile(s, 95))
        self.acc.append(rec)
        print(f"frame {len(self.acc)}: A max {rec['A']['max']:6.2f} | v1 max {rec['v1']['max']:6.2f}"
              f" v2 max {rec['v2']['max']:6.2f} | v1 MAE {rec['v1']['anchor_mae']:.3f}"
              f" v2 MAE {rec['v2']['anchor_mae']:.3f}", flush=True)

        if len(self.acc) == 1 and self.a.save_prefix:
            dv = D0 > 0
            lo, hi = np.percentile(D0[dv], [2, 98])

            def col(X, cmap=cv2.COLORMAP_TURBO):
                n = np.clip((X - lo) / max(1e-6, hi - lo), 0, 1)
                c = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
                c[X <= 0] = 0
                return c
            p = self.a.save_prefix
            cv2.imwrite(f"{p}_cam.png", rgb_bgr)
            cv2.imwrite(f"{p}_A.png", col(D0))
            cv2.imwrite(f"{p}_v1.png", col(D1))
            cv2.imwrite(f"{p}_v2.png", col(D2))
            y0, y1, x0, x1 = 370, 590, 600, 1120
            for nm, D in (('A', D0), ('v1', D1), ('v2', D2)):
                cv2.imwrite(f"{p}_crop_{nm}.png",
                            cv2.resize(col(D)[y0:y1, x0:x1], (760, 760 * (y1 - y0) // (x1 - x0))))

        if len(self.acc) >= self.a.frames:
            raise SystemExit(0)


def summarize(acc, out):
    keys = ['median', 'max', 'frac_gt3', 'frac_gt5', 'frac_clamp', 'ctr', 'anchor_mae',
            'anchor_p90', 'band173', 'band13', 'box_median', 'mean_abs_dev_from_A']
    res = {'frames': len(acc),
           'anchors': float(np.mean([r['anchors'] for r in acc])),
           'tof_max': float(np.mean([r['tof_max'] for r in acc])),
           'tof_ctr': float(np.mean([r['tof_ctr'] for r in acc])),
           'img_mean': float(np.mean([r['img_mean'] for r in acc]))}
    for nm in ('A', 'v1', 'v2'):
        res[nm] = {k: float(np.mean([r[nm][k] for r in acc])) for k in keys}
        if 'sigma_med' in acc[0][nm]:
            for k in ('sigma_med', 'sigma_p5', 'sigma_p95'):
                res[nm][k] = float(np.mean([r[nm][k] for r in acc]))
    json.dump(res, open(out, 'w'), indent=2)
    print()
    print(f"{'metric':<24}{'A':>12}{'v1':>12}{'v2':>12}")
    print('-' * 60)
    for k in keys:
        print(f"{k:<24}{res['A'][k]:>12.4f}{res['v1'][k]:>12.4f}{res['v2'][k]:>12.4f}")
    for k in ('sigma_med', 'sigma_p5', 'sigma_p95'):
        print(f"{k:<24}{'-':>12}{res['v1'][k]:>12.4f}{res['v2'][k]:>12.4f}")
    print()
    print(f"ToF: max {res['tof_max']:.2f} m, centre {res['tof_ctr']:.2f} m, "
          f"anchors {res['anchors']:.0f} | frame brightness {res['img_mean']:.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--backbone-engine', required=True)
    p.add_argument('--engine-v1', required=True)
    p.add_argument('--engine-v2', required=True)
    p.add_argument('--frames', type=int, default=10)
    p.add_argument('--out', required=True)
    p.add_argument('--save-prefix', default='')
    args = p.parse_args()
    rclpy.init()
    n = AB(args)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if n.acc:
            summarize(n.acc, args.out)
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

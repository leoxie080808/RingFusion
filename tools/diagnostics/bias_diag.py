#!/usr/bin/env python3
"""Root-cause the ~14% ground-plane under-read.

At every ToF anchor we have a real distance z and the pipeline's depth at that pixel.
The shape of the error tells us which stage is wrong:

  error grows in PROPORTION to depth      -> scale error, i.e. the affine 'a'
  error is a CONSTANT offset              -> offset error, i.e. the affine 'b'
  error varies with IMAGE ROW at fixed z  -> geometry: extrinsic / FOV / rectification
  affine fit residual already large       -> a 2-param global fit cannot fit this disparity

We also fit an oracle affine (least squares on the anchors we actually have) to see how
much of the error a perfect global fit could remove -- that bounds what re-tuning can buy.
"""
import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

from ringfusion_perception import geometry as geo
from ringfusion_perception import anchoring as anc
from ringfusion_perception import gpu_ops
from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.perception_node import load_calib


class Bias(Node):
    def __init__(self, a):
        super().__init__('bias_diag')
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
        self.backbone = TensorRTBackbone(a.backbone_engine)
        self.img = self.tof = None
        self.acc = []
        self.n = 0
        self.create_subscription(Image, '/image', self.on_img, 5)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 10)
        self.create_timer(0.4, self.tick)

    def on_img(self, m):
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
        rgb = self.img[:, :, ::-1]
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
        if inb.sum() < 20:
            return
        du = disp[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)][inb]
        zz = z[inb]
        vv = v[inb]
        a_, b_ = anc.solve_robust(du, 1.0 / zz, np.ones(inb.sum()), iters=1)
        D0 = gpu_ops.to_metric_depth(disp, a_, b_)
        d_at = D0[vv, u[inb]]
        # oracle: plain least squares on ALL anchors (no robust down-weighting)
        A = np.stack([du, np.ones_like(du)], 1)
        oa, ob = np.linalg.lstsq(A, 1.0 / zz, rcond=None)[0]
        d_or = 1.0 / np.maximum(oa * du + ob, 1e-6)
        self.acc.append({'z': zz, 'd': d_at, 'v': vv, 'disp': du,
                         'a': a_, 'b': b_, 'oa': oa, 'ob': ob, 'd_or': d_or})
        print(f"  frame {len(self.acc)}: {inb.sum()} anchors  a={a_:.4f} b={b_:.4f} "
              f"| oracle a={oa:.4f} b={ob:.4f}", flush=True)
        if len(self.acc) >= self.a.frames:
            raise SystemExit(0)


def analyse(acc):
    z = np.concatenate([r['z'] for r in acc])
    d = np.concatenate([r['d'] for r in acc])
    v = np.concatenate([r['v'] for r in acc])
    dr = np.concatenate([r['d_or'] for r in acc])
    err = d - z
    print(f"\n{len(z)} anchor samples over {len(acc)} frames")
    print(f"ToF range {z.min():.2f}-{z.max():.2f} m, median {np.median(z):.2f} m")
    print(f"\nOVERALL: mean err {err.mean():+.3f} m, median {np.median(err):+.3f} m, "
          f"MAE {np.abs(err).mean():.3f} m")
    print(f"         mean ratio D/z = {np.mean(d/z):.3f}  (1.0 = perfect)")

    print("\n--- IS IT SCALE OR OFFSET? regress D = alpha*z + beta ---")
    al, be = np.polyfit(z, d, 1)
    print(f"  alpha = {al:.3f}   beta = {be:+.3f} m")
    print(f"  -> {'SCALE error (proportional)' if abs(al-1) > 0.05 else 'alpha ~ 1'}"
          f"; {'plus a constant offset' if abs(be) > 0.05 else 'no meaningful offset'}")

    print("\n--- ERROR vs ToF DISTANCE (is it depth-dependent?) ---")
    qs = np.quantile(z, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        m = (z >= qs[i]) & (z <= qs[i + 1])
        if m.sum() > 10:
            print(f"  z {qs[i]:.2f}-{qs[i+1]:.2f} m (n={m.sum():5d}): "
                  f"mean err {err[m].mean():+.3f} m   ratio {np.mean(d[m]/z[m]):.3f}")

    print("\n--- ERROR vs IMAGE ROW (is it geometric?) ---")
    rq = np.quantile(v, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        m = (v >= rq[i]) & (v <= rq[i + 1])
        if m.sum() > 10:
            print(f"  row {rq[i]:4.0f}-{rq[i+1]:4.0f} (n={m.sum():5d}): "
                  f"mean err {err[m].mean():+.3f} m   ratio {np.mean(d[m]/z[m]):.3f}"
                  f"   mean z {z[m].mean():.2f} m")

    print("\n--- HOW MUCH COULD A PERFECT GLOBAL FIT RECOVER? ---")
    eo = dr - z
    print(f"  deployed robust fit : MAE {np.abs(err).mean():.3f} m, "
          f"mean ratio {np.mean(d/z):.3f}")
    print(f"  oracle least-squares: MAE {np.abs(eo).mean():.3f} m, "
          f"mean ratio {np.mean(dr/z):.3f}")
    print(f"  -> refitting can remove at most "
          f"{100*(1-np.abs(eo).mean()/np.abs(err).mean()):.0f}% of the error")

    print("\n--- HOW WELL CAN 2 PARAMS FIT THIS DISPARITY AT ALL? ---")
    dp = np.concatenate([r['disp'] for r in acc])
    iv = 1.0 / z
    pr = np.polyfit(dp, iv, 1)
    res = iv - np.polyval(pr, dp)
    print(f"  corr(disp, 1/z) = {np.corrcoef(dp, iv)[0,1]:.4f}")
    print(f"  residual std in inverse-depth = {res.std():.4f} 1/m "
          f"(= {res.std()*np.median(z)**2*100:.1f} cm at the median depth)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--backbone-engine', required=True)
    p.add_argument('--frames', type=int, default=8)
    a = p.parse_args()
    rclpy.init()
    n = Bias(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if n.acc:
            analyse(n.acc)
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

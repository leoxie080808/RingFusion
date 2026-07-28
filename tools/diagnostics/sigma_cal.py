#!/usr/bin/env python3
"""On-robot check of the uncertainty channel -- the live analogue of training 'coverage'.

Training reports coverage: the fraction of true depths landing inside the predicted +/-1
sigma band. A perfectly calibrated Gaussian gives 0.6827. Too low = overconfident (the
band is too narrow); too high = underconfident (sigma is inflated and carries no
information, which is what v1/v2 did -- sigma pinned near ~0.95 m everywhere).

Here the "true depth" is the real ToF reading at each projected anchor, so this measures
calibration against the sensor rather than against a held-out split.

Also reports the SPATIAL signal: if sigma is informative it should be LARGER where the
error is larger. A near-zero correlation means sigma is just a constant with noise on top.
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
from ringfusion_perception.pipeline import splat_anchors
from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.perception_node import load_calib

IDEAL_1S = 0.6827
IDEAL_2S = 0.9545


class SigCal(Node):
    def __init__(self, a):
        super().__init__('sigma_cal')
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
        self.eng = {}
        for spec in a.engine:
            nm, path = spec.split('=', 1)
            self.eng[nm] = ResidualRefiner(path)
        self.img = self.tof = None
        self.acc = {k: {'e': [], 's': []} for k in self.eng}
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
        if inb.sum() < 50:
            return
        fit = anc.solve_robust(disp[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)][inb],
                               1.0 / z[inb], np.ones(int(inb.sum())), iters=1)
        if fit is None:
            return
        a_, b_ = fit
        D0 = gpu_ops.to_metric_depth(disp, a_, b_)
        ad, am = splat_anchors(u, v, z, inb, (h, w))
        vv, uu, zz = v[inb], u[inb], z[inb]
        for nm, eng in self.eng.items():
            D, tau2 = eng.refine(rgb, D0, disp, ad, am, a_, b_)
            self.acc[nm]['e'].append(D[vv, uu] - zz)
            self.acc[nm]['s'].append(np.sqrt(np.clip(tau2[vv, uu], 1e-12, None)))
        print(f"  frame {len(self.acc[list(self.eng)[0]]['e'])}: {inb.sum()} anchors",
              flush=True)
        if len(self.acc[list(self.eng)[0]]['e']) >= self.a.frames:
            raise SystemExit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--backbone-engine', required=True)
    p.add_argument('--engine', nargs='+', required=True, metavar='NAME=PATH')
    p.add_argument('--frames', type=int, default=8)
    a = p.parse_args()
    rclpy.init()
    n = SigCal(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    if not n.acc:
        return
    print(f"\n{'engine':<10}{'MAE':>9}{'med sigma':>11}{'cov 1s':>9}{'cov 2s':>9}"
          f"{'corr(s,|e|)':>13}")
    print('-' * 62)
    for nm in n.eng:
        e = np.concatenate(n.acc[nm]['e'])
        s = np.concatenate(n.acc[nm]['s'])
        c1 = float(np.mean(np.abs(e) <= s))
        c2 = float(np.mean(np.abs(e) <= 2 * s))
        cr = float(np.corrcoef(s, np.abs(e))[0, 1])
        print(f"{nm:<10}{np.abs(e).mean():>8.3f}m{np.median(s):>10.3f}m"
              f"{c1:>9.3f}{c2:>9.3f}{cr:>13.3f}")
    print(f"\nideal: cov 1s = {IDEAL_1S:.3f}, cov 2s = {IDEAL_2S:.3f}")
    print("cov below ideal = OVERconfident (band too narrow)")
    print("cov above ideal = UNDERconfident (sigma inflated, carries no information)")
    print("corr(sigma,|error|) near 0 = sigma is spatially uninformative; higher is better")


if __name__ == '__main__':
    main()

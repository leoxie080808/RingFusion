#!/usr/bin/env python3
"""Are the ToF anchors landing on the right pixels?

Student and teacher agree with each other (rho 0.99) but each only scores rho ~0.74
against the ToF. Either monocular depth is genuinely this hard here, OR the ToF-zone ->
image-pixel correspondence is wrong, in which case we are comparing disparity at the
wrong pixels and BOTH models get penalised identically.

Test: sweep the projection parameters (ToF field of view, and the ToF->camera offset).
If the current calibration is right, rho peaks at the current values. If rho climbs
sharply somewhere else, the calibration is wrong and that peak is the correction.
"""
import argparse
import itertools

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

from ringfusion_perception import geometry as geo
from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.perception_node import load_calib


class Sweep(Node):
    def __init__(self, a):
        super().__init__('proj_sweep')
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
        self.grab = []
        self.n = 0
        self.create_subscription(Image, '/image', self.on_img, 5)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 10)
        self.create_timer(0.5, self.tick)

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
        self.grab.append((self.backbone.infer(rgb), self.tof.copy(), rgb.shape[:2]))
        print(f"  captured {len(self.grab)}", flush=True)
        if len(self.grab) >= self.a.frames:
            raise SystemExit(0)


def rho_for(grab, calib, fh, fv, T):
    """Correlation of disparity with true inverse depth under a given projection."""
    ds, ivs = [], []
    for disp, td, (h, w) in grab:
        tv = np.isfinite(td)
        rows, cols = td.shape
        proj = geo.project_zone_to_pixel(td, tv, cols, rows, fh, fv, T,
                                         calib['K'], calib['dist'], model='pinhole')
        uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']
        fin = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
        u = np.round(np.where(fin, uv[:, 0], -1)).astype(int)
        v = np.round(np.where(fin, uv[:, 1], -1)).astype(int)
        inb = ok & fin & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
        if inb.sum() < 50:
            continue
        ds.append(disp[v[inb], u[inb]])
        ivs.append(1.0 / z[inb])
    if not ds:
        return -1.0, 0
    d, iv = np.concatenate(ds), np.concatenate(ivs)
    return float(np.corrcoef(d, iv)[0, 1]), len(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--backbone-engine', required=True)
    p.add_argument('--frames', type=int, default=4)
    a = p.parse_args()
    rclpy.init()
    n = Sweep(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    if not n.grab:
        print('no frames')
        return

    cal = n.calib
    fh0, fv0 = cal['fov_h'], cal['fov_v']
    T0 = np.asarray(cal['T_cam_tof'], np.float64)
    base, nn = rho_for(n.grab, cal, fh0, fv0, T0)
    print(f"\nbaseline (deployed calibration): fov_h={fh0:.1f} fov_v={fv0:.1f}  "
          f"rho={base:.4f}  n={nn}")

    print("\n--- sweep ToF field of view ---")
    best = (base, fh0, fv0)
    for sh, sv in itertools.product([.6, .7, .8, .9, 1.0, 1.1, 1.25, 1.4],
                                    [.6, .7, .8, .9, 1.0, 1.1, 1.25, 1.4]):
        r, _ = rho_for(n.grab, cal, fh0 * sh, fv0 * sv, T0)
        if r > best[0]:
            best = (r, fh0 * sh, fv0 * sv)
    print(f"  best rho {best[0]:.4f} at fov_h={best[1]:.1f} fov_v={best[2]:.1f}"
          f"  (baseline {base:.4f} at {fh0:.1f}/{fv0:.1f})")

    print("\n--- sweep ToF->camera offset (metres), at best FOV ---")
    b2 = (best[0], T0.copy())
    for dy, dz in itertools.product([-.10, -.05, -.02, 0, .02, .05, .10],
                                    [-.10, -.05, 0, .05, .10]):
        T = T0.copy()
        Tf = T.ravel()
        if Tf.size >= 3:
            T2 = T.copy().astype(np.float64)
            flat = T2.ravel()
            flat[1] += dy
            flat[2] += dz
            T2 = flat.reshape(T.shape)
        else:
            continue
        r, _ = rho_for(n.grab, cal, best[1], best[2], T2)
        if r > b2[0]:
            b2 = (r, T2)
    print(f"  best rho {b2[0]:.4f}")
    print(f"  T deployed: {T0.ravel()[:3]}")
    print(f"  T best    : {np.asarray(b2[1]).ravel()[:3]}")

    print(f"\nSUMMARY: rho {base:.4f} (deployed) -> {b2[0]:.4f} (best found)")
    if b2[0] - base < 0.03:
        print("  => calibration is NOT the bottleneck; the projection is already near-optimal.")
        print("     The rho ceiling is real monocular limitation in this scene.")
    else:
        print("  => calibration IS materially off; the peak above is the correction to apply.")


if __name__ == '__main__':
    main()

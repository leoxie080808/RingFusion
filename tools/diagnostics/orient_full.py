#!/usr/bin/env python3
"""Full search over ToF grid orientation x FOV assignment, plus a visual proof.

Scores every combination by how well backbone disparity tracks true inverse depth at the
projected anchors, then renders the ToF depth splatted onto the camera image for the
deployed mapping vs the winner -- if the winner is right, its ToF pattern lines up with
the scene (near floor at the bottom, far wall at the top) and the deployed one does not.
"""
import argparse
import itertools

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

from ringfusion_perception import geometry as geo
from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.perception_node import load_calib

ORIENT = {
    'as-is': lambda a: a,
    'fliplr': lambda a: np.fliplr(a),
    'flipud': lambda a: np.flipud(a),
    'rot90': lambda a: np.rot90(a, 1),
    'rot180': lambda a: np.rot90(a, 2),
    'rot270': lambda a: np.rot90(a, 3),
    'transpose': lambda a: a.T,
    'fliplr+flipud': lambda a: np.flipud(np.fliplr(a)),
    'transpose+fliplr': lambda a: np.fliplr(a.T),
    'transpose+flipud': lambda a: np.flipud(a.T),
}


class Cap(Node):
    def __init__(self, a):
        super().__init__('orient_full')
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
        self.grab.append((self.backbone.infer(self.img[:, :, ::-1]),
                          self.tof.copy(), self.img.copy()))
        print(f"  captured {len(self.grab)}", flush=True)
        if len(self.grab) >= self.a.frames:
            raise SystemExit(0)


def project(disp, td, shape, calib, fn, fh, fv):
    td = np.ascontiguousarray(fn(td))
    tv = np.isfinite(td)
    rows, cols = td.shape
    h, w = shape
    proj = geo.project_zone_to_pixel(td, tv, cols, rows, fh, fv, calib['T_cam_tof'],
                                     calib['K'], calib['dist'], model='pinhole')
    uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']
    fin = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
    u = np.round(np.where(fin, uv[:, 0], -1)).astype(int)
    v = np.round(np.where(fin, uv[:, 1], -1)).astype(int)
    inb = ok & fin & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    return u, v, z, inb


def score(grab, calib, fn, fh, fv):
    ds, ivs = [], []
    for disp, td, img in grab:
        u, v, z, inb = project(disp, td, img.shape[:2], calib, fn, fh, fv)
        if inb.sum() < 50:
            continue
        ds.append(disp[v[inb], u[inb]])
        ivs.append(1.0 / z[inb])
    if not ds:
        return -1.0, 9.9
    d, iv = np.concatenate(ds), np.concatenate(ivs)
    rho = float(np.corrcoef(d, iv)[0, 1])
    A = np.stack([d, np.ones_like(d)], 1)
    aa, bb = np.linalg.lstsq(A, iv, rcond=None)[0]
    dd = 1.0 / np.maximum(aa * d + bb, 1e-6)
    return rho, float(np.abs(dd - 1.0 / iv).mean())


def overlay(img, u, v, z, inb, title):
    o = (img * 0.35).astype(np.uint8)
    zz = z[inb]
    lo, hi = np.percentile(zz, [5, 95])
    nn = np.clip((zz - lo) / max(1e-6, hi - lo), 0, 1)
    cols = cv2.applyColorMap((nn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0]
    for (uu, vv, c) in zip(u[inb], v[inb], cols):
        cv2.circle(o, (int(uu), int(vv)), 7, tuple(int(x) for x in c), -1)
    cv2.rectangle(o, (0, 0), (o.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(o, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--backbone-engine', required=True)
    p.add_argument('--frames', type=int, default=4)
    p.add_argument('--out', default='')
    a = p.parse_args()
    rclpy.init()
    n = Cap(a)
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
    f0, f1 = cal['fov_h'], cal['fov_v']
    print(f"\nnominal fov {f0:.1f} x {f1:.1f}, {len(n.grab)} frames\n")
    res = []
    for nm, (fh, fv) in itertools.product(ORIENT, [(f0, f1), (f1, f0)]):
        rho, mae = score(n.grab, cal, ORIENT[nm], fh, fv)
        res.append((rho, mae, nm, fh, fv))
    res.sort(reverse=True)
    print(f"{'orientation':<20}{'fov h x v':>14}{'rho':>9}{'MAE':>10}")
    print('-' * 54)
    for rho, mae, nm, fh, fv in res[:8]:
        print(f"{nm:<20}{f'{fh:.0f} x {fv:.0f}':>14}{rho:>9.4f}{mae:>9.3f} m")
    print('  ...')
    for rho, mae, nm, fh, fv in res:
        if nm == 'as-is' and abs(fh - f0) < .1:
            print(f"{'as-is (DEPLOYED)':<20}{f'{fh:.0f} x {fv:.0f}':>14}{rho:>9.4f}{mae:>9.3f} m")

    best = res[0]
    print(f"\nWINNER: {best[2]}  fov {best[3]:.0f} x {best[4]:.0f}  "
          f"rho {best[0]:.4f}  MAE {best[1]:.3f} m")

    if a.out:
        disp, td, img = n.grab[0]
        u0, v0, z0, i0 = project(disp, td, img.shape[:2], cal, ORIENT['as-is'], f0, f1)
        ub, vb, zb, ib = project(disp, td, img.shape[:2], cal,
                                 ORIENT[best[2]], best[3], best[4])
        dep_rho = next(r for r, _, nm, fh, _ in res if nm == 'as-is' and abs(fh - f0) < .1)
        a_img = overlay(img, u0, v0, z0, i0,
                        f"DEPLOYED  as-is {f0:.0f}x{f1:.0f}   rho={dep_rho:.3f}")
        b_img = overlay(img, ub, vb, zb, ib,
                        f"FIXED  {best[2]} {best[3]:.0f}x{best[4]:.0f}   rho={best[0]:.3f}")
        cv2.imwrite(f"{a.out}_deployed.png", a_img)
        cv2.imwrite(f"{a.out}_fixed.png", b_img)
        cv2.imwrite(f"{a.out}_cam.png", img)
        print(f"wrote {a.out}_deployed.png / _fixed.png / _cam.png")


if __name__ == '__main__':
    main()

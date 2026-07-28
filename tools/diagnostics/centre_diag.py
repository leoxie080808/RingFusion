#!/usr/bin/env python3
"""Why does the ToF overlay read a different centre distance than the fused-depth overlay?

Compares three things that all get loosely called "the centre":
  1. what the ToF panel prints  -- median of the central 4x4 ToF zones
  2. what the depth panel prints -- median of the central 16x16 image pixels
  3. the apples-to-apples number -- fused depth sampled AT the pixels those centre ToF
     zones actually project onto

If (1) and (2) disagree but (1) and (3) agree, the gap is sampling geometry, not error.
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

from ringfusion_perception import geometry as geo
from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.perception_node import load_calib


class Diag(Node):
    def __init__(self, a):
        super().__init__('centre_diag')
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
        self.tof = self.dep = None
        self.n = 0
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 10)
        self.create_subscription(Image, '/depth', self.on_dep, 5)
        self.create_timer(0.4, self.tick)

    def on_tof(self, m):
        self.tof = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols)

    def on_dep(self, m):
        self.dep = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)

    def tick(self):
        if self.tof is None or self.dep is None:
            return
        self.n += 1
        if self.n < 3:
            return
        td, D = self.tof, self.dep
        h, w = D.shape
        K = self.calib['K']
        rows, cols = td.shape
        tv = np.isfinite(td)

        # ---- 1. what the ToF panel prints
        c = td[14:18, 14:18]
        c = c[np.isfinite(c)]
        tof_panel = float(np.median(c))

        # ---- 2. what the depth panel prints
        p = D[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8]
        p = p[p > 0]
        dep_panel = float(np.median(p))

        # ---- 3. project the centre ToF zones and read fused depth there
        proj = geo.project_zone_to_pixel(td, tv, cols, rows, self.calib['fov_h'],
                                         self.calib['fov_v'], self.calib['T_cam_tof'],
                                         K, self.calib['dist'], model='pinhole')
        uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']
        fin = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
        u = np.round(np.where(fin, uv[:, 0], -1)).astype(int)
        v = np.round(np.where(fin, uv[:, 1], -1)).astype(int)
        inb = ok & fin & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)

        zi = np.arange(rows * cols).reshape(rows, cols)
        centre_ids = zi[14:18, 14:18].ravel()
        sel = np.zeros(rows * cols, bool)
        sel[centre_ids] = True
        m3 = sel & inb
        tof_c = z[m3]
        dep_at = D[v[m3], u[m3]]
        good = dep_at > 0

        print(f"\n--- frame {self.n} ---")
        print(f"1. ToF panel  (central 4x4 of 32x32 zones)      : {tof_panel:.3f} m")
        print(f"2. Depth panel(central 16x16 of {w}x{h} px)   : {dep_panel:.3f} m")
        print(f"3. Fused depth AT the centre ToF zones' pixels  : "
              f"{np.median(dep_at[good]):.3f} m   (ToF there: {np.median(tof_c):.3f} m)")
        print(f"   -> apples-to-apples error: "
              f"{np.median(dep_at[good]) - np.median(tof_c):+.3f} m "
              f"({100*(np.median(dep_at[good])/np.median(tof_c)-1):+.1f}%)")

        # where do those zones actually land, and how big is each window?
        print(f"   centre ToF zones project to px "
              f"x {u[m3].min()}-{u[m3].max()}, y {v[m3].min()}-{v[m3].max()}")
        print(f"   image centre patch is        px "
              f"x {w//2-8}-{w//2+8}, y {h//2-8}-{h//2+8}")
        Kf = np.asarray(K, np.float64).ravel()
        fx = Kf[0] if Kf.size == 4 else Kf.reshape(3, 3)[0, 0]   # (fx,fy,cx,cy) or 3x3
        ang_img = 2 * np.degrees(np.arctan(8 / fx))
        span_x = (u[m3].max() - u[m3].min())
        ang_tof = 2 * np.degrees(np.arctan(span_x / 2 / fx))
        print(f"   angular window: ToF 4x4 ~{ang_tof:.1f} deg  vs  image 16px ~{ang_img:.2f} deg"
              f"  ({ang_tof/max(ang_img,1e-6):.0f}x wider)")

        if self.n >= self.a.frames + 2:
            raise SystemExit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--frames', type=int, default=3)
    a = p.parse_args()
    rclpy.init()
    n = Diag(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

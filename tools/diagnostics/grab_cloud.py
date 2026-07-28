#!/usr/bin/env python3
"""Capture one colour point cloud for the web viewer.

/cloud is XYZ-only, so instead of subscribing to it we rebuild the same cloud from
/depth + the rectified image -- identical geometry (same K, same unprojection), but every
point also carries its camera colour, which makes the 3D view actually readable.

Writes a compact binary: int16 quantised positions + uint8 RGB, plus a JSON sidecar with
the scale factors needed to decode.
"""
import argparse
import json
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.perception_node import load_calib


class Grab(Node):
    def __init__(self, a):
        super().__init__('grab_cloud')
        self.a = a
        raw = load_calib(a.calib)
        r = raw['rectify']
        self.rect = FisheyeRectifier(raw['K'], raw['dist'], raw['model'],
                                     size_in=(raw['img_w'], raw['img_h']),
                                     size_out=(r['width'], r['height']),
                                     balance=r['balance'], fov_scale=r['fov_scale'])
        K = np.asarray(self.rect.K_rect, np.float64).ravel()
        if K.size == 9:
            m = K.reshape(3, 3)
            K = np.array([m[0, 0], m[1, 1], m[0, 2], m[1, 2]])
        self.K = K
        self.img = self.dep = None
        self.n = 0
        self.create_subscription(Image, '/image', self.on_img, 5)
        self.create_subscription(Image, '/depth', self.on_dep, 5)
        self.create_timer(0.4, self.tick)

    def on_img(self, m):
        if m.encoding == 'rgb8':
            rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.img = self.rect.rectify(rgb[:, :, ::-1])       # BGR rectified

    def on_dep(self, m):
        self.dep = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)

    def tick(self):
        if self.img is None or self.dep is None:
            return
        self.n += 1
        if self.n < 4:
            return
        s = self.a.step
        d = self.dep[::s, ::s]
        bgr = self.img[::s, ::s]
        fx, fy, cx, cy = self.K
        us = (np.arange(0, self.dep.shape[1], s) - cx) / fx
        vs = (np.arange(0, self.dep.shape[0], s) - cy) / fy
        X = d * us[None, :]
        Y = d * vs[:, None]
        m = (d > self.a.zmin) & (d < self.a.zmax) & np.isfinite(d)
        x, y, z = X[m], Y[m], d[m]
        col = bgr[m][:, ::-1]                                    # -> RGB

        if x.size > self.a.max_points:                           # uniform thin
            idx = np.random.default_rng(0).choice(x.size, self.a.max_points, replace=False)
            x, y, z, col = x[idx], y[idx], z[idx], col[idx]

        # centre so the viewer orbits the cloud, not the origin
        cxx, cyy, czz = float(x.mean()), float(y.mean()), float(z.mean())
        P = np.stack([x - cxx, y - cyy, z - czz], 1)
        scale = float(np.abs(P).max()) / 32000.0
        q = np.clip(P / scale, -32768, 32767).astype('<i2')

        with open(self.a.out, 'wb') as f:
            f.write(q.tobytes())
            f.write(col.astype(np.uint8).tobytes())
        meta = {'n': int(x.size), 'scale': scale,
                'centre': [cxx, cyy, czz],
                'extent_m': [float(P[:, 0].ptp()), float(P[:, 1].ptp()), float(P[:, 2].ptp())],
                'z_range_m': [float(z.min()), float(z.max())]}
        json.dump(meta, open(self.a.out + '.json', 'w'), indent=2)
        print(json.dumps(meta, indent=2))
        print(f"wrote {self.a.out} ({(q.nbytes + col.size) / 1e6:.2f} MB raw)")
        raise SystemExit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--step', type=int, default=3)
    p.add_argument('--zmin', type=float, default=0.15)
    p.add_argument('--zmax', type=float, default=6.0)
    p.add_argument('--max-points', type=int, default=90000)
    a = p.parse_args()
    rclpy.init()
    n = Grab(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()

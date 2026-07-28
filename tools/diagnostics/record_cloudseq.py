#!/usr/bin/env python3
"""Record a colour+depth sequence for the in-page animated point-cloud viewer.

Embedding a point cloud per frame is hopeless (90k pts x 500 frames). Instead we save a
side-by-side frame -- LEFT: rectified colour, RIGHT: depth encoded as 8-bit luma over a
FIXED range -- and let ffmpeg compress the sequence. The page then unprojects it to points
in a vertex shader, so page weight is video-sized rather than point-count-sized.

Depth is quantised to 8 bits over [zmin, zmax]; at the default 0.15-6 m that is ~2.3 cm
per level, which is well under the pipeline's own error and invisible in a 3D overview.
0 is reserved for "no data" so holes stay holes instead of collapsing to zmin.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.perception_node import load_calib


class Rec(Node):
    def __init__(self, a):
        super().__init__('record_cloudseq')
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
        self.K_full = K
        self.src_wh = (raw['rectify']['width'], raw['rectify']['height'])
        self.img = self.dep = None
        self.n = 0
        self.t0 = None
        os.makedirs(a.dir, exist_ok=True)
        self.create_subscription(Image, '/image', self.on_img, 5)
        self.create_subscription(Image, '/depth', self.on_dep, 5)
        self.create_timer(1.0 / a.fps, self.tick)
        print('waiting for /image + /depth ...', flush=True)

    def on_img(self, m):
        if m.encoding == 'rgb8':
            rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.img = self.rect.rectify(rgb[:, :, ::-1])

    def on_dep(self, m):
        self.dep = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)

    def tick(self):
        if self.img is None or self.dep is None:
            return
        if self.t0 is None:
            self.t0 = time.time()
            print('>>> RECORDING — drive the robot now <<<', flush=True)
        el = time.time() - self.t0
        if el > self.a.secs:
            self.finish()
            raise SystemExit(0)
        W, H = self.a.width, self.a.height
        colr = cv2.resize(self.img, (W, H), interpolation=cv2.INTER_AREA)
        d = cv2.resize(self.dep, (W, H), interpolation=cv2.INTER_NEAREST)
        lo, hi = self.a.zmin, self.a.zmax
        q = np.zeros((H, W), np.uint8)
        ok = np.isfinite(d) & (d > lo) & (d < hi)
        q[ok] = np.clip(1 + (d[ok] - lo) / (hi - lo) * 254.0, 1, 255).astype(np.uint8)
        frame = np.hstack([colr, cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)])
        cv2.imwrite(f"{self.a.dir}/f{self.n:05d}.png", frame)
        self.n += 1
        if self.n % 25 == 0:
            print(f"  {el:4.1f}s  {self.n} frames", flush=True)

    def finish(self):
        W, H = self.a.width, self.a.height
        sx = W / float(self.src_wh[0])
        sy = H / float(self.src_wh[1])
        meta = {
            'w': W, 'h': H, 'fps': self.a.fps, 'frames': self.n,
            'zmin': self.a.zmin, 'zmax': self.a.zmax,
            # intrinsics rescaled to the stored resolution
            'fx': self.K_full[0] * sx, 'fy': self.K_full[1] * sy,
            'cx': self.K_full[2] * sx, 'cy': self.K_full[3] * sy,
        }
        json.dump(meta, open(os.path.join(self.a.dir, 'meta.json'), 'w'), indent=2)
        print(json.dumps(meta, indent=2))
        print(f'wrote {self.n} frames')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--dir', required=True)
    p.add_argument('--secs', type=float, default=32.0)
    p.add_argument('--fps', type=float, default=12.0)
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--height', type=int, default=240)
    p.add_argument('--zmin', type=float, default=0.15)
    p.add_argument('--zmax', type=float, default=6.0)
    a = p.parse_args()
    rclpy.init()
    n = Rec(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()

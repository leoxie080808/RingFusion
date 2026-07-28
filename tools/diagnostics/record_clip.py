#!/usr/bin/env python3
"""Record a synchronised Camera | ToF | Depth clip to numbered frames for ffmpeg.

Colour scales are FIXED (not per-frame percentiles) so the clip does not strobe as the
autoscale wanders -- a fixed scale is also the only way the colours mean the same thing
from one second to the next.
"""
import argparse
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

FONT = cv2.FONT_HERSHEY_SIMPLEX
PW, PH = 420, 315                      # per-panel size


def colorize(a, mask, lo, hi, cmap=cv2.COLORMAP_TURBO):
    n = np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    c[~mask] = (18, 18, 18)
    return c


def label(img, txt, sub=""):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 18), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (8, img.shape[0] - 8), FONT, 0.46, (90, 230, 255), 1, cv2.LINE_AA)
    return img


def topdown(depth, K, z_max, x_half, step=6, floor_margin=0.12):
    """Bird's-eye occupancy: unproject /depth, drop the ground plane, bin the rest.

    Camera optical frame is x right, y DOWN, z forward, so the top-down plane is (x, z)
    and the floor is at LARGE y. Estimating the floor as a high percentile of y (rather
    than a fixed camera height, which we never measured) keeps this honest if the mount
    changes; points within `floor_margin` of it are ground and are drawn faintly, points
    above it are obstacles and are drawn bright. Returns a PH x PW BGR panel with the
    robot at bottom-centre looking up the image.
    """
    d = depth[::step, ::step]
    h, w = d.shape
    fx, fy, cx, cy = K
    us = (np.arange(0, depth.shape[1], step) - cx) / fx
    vs = (np.arange(0, depth.shape[0], step) - cy) / fy
    X = d * us[None, :]
    Y = d * vs[:, None]
    m = (d > 0.15) & (d < z_max) & np.isfinite(d)
    x, y, z = X[m], Y[m], d[m]

    panel = np.full((PH, PW, 3), 16, np.uint8)
    if x.size < 50:
        return panel, 0
    floor_y = np.percentile(y, 92)                 # ground sits at the largest y
    obst = y < (floor_y - floor_margin)

    px = ((x + x_half) / (2 * x_half) * (PW - 1)).astype(np.int32)
    pz = ((1.0 - z / z_max) * (PH - 1)).astype(np.int32)
    ok = (px >= 0) & (px < PW) & (pz >= 0) & (pz < PH)

    # ground first (faint), obstacles on top (bright, coloured by distance)
    g = ok & ~obst
    panel[pz[g], px[g]] = (44, 44, 44)
    o = ok & obst
    if o.any():
        nz = np.clip(z[o] / z_max, 0, 1)
        cols = cv2.applyColorMap((nz * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0]
        panel[pz[o], px[o]] = cols
    panel = cv2.dilate(panel, np.ones((2, 2), np.uint8))

    # range rings + robot marker at bottom-centre
    ox, oy = PW // 2, PH - 1
    for r in (1.0, 2.0, 3.0):
        if r < z_max:
            rr = int(r / z_max * (PH - 1))
            cv2.circle(panel, (ox, oy), rr, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.putText(panel, f"{r:.0f}m", (ox + 4, oy - rr + 12), FONT, 0.36,
                        (120, 120, 120), 1, cv2.LINE_AA)
    cv2.circle(panel, (ox, oy), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.line(panel, (ox, oy), (ox, oy - 22), (255, 255, 255), 2, cv2.LINE_AA)
    return panel, int(o.sum())


class Rec(Node):
    def __init__(self, a):
        super().__init__('record_clip')
        self.a = a
        self.img = self.tof = self.dep = None
        self.n = 0
        self.t0 = None
        # rectified (pinhole) intrinsics -- /depth is published in this frame
        from ringfusion_perception.rectify import FisheyeRectifier
        from ringfusion_perception.perception_node import load_calib
        raw = load_calib(a.calib)
        r = raw['rectify']
        rect = FisheyeRectifier(raw['K'], raw['dist'], raw['model'],
                                size_in=(raw['img_w'], raw['img_h']),
                                size_out=(r['width'], r['height']),
                                balance=r['balance'], fov_scale=r['fov_scale'])
        self.K = np.asarray(rect.K_rect, np.float64).ravel()
        if self.K.size == 9:                       # accept 3x3 or (fx,fy,cx,cy)
            m = self.K.reshape(3, 3)
            self.K = np.array([m[0, 0], m[1, 1], m[0, 2], m[1, 2]])
        print(f'rectified K (fx,fy,cx,cy) = {self.K}', flush=True)
        os.makedirs(a.dir, exist_ok=True)
        self.create_subscription(Image, '/image', self.on_img, 5)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 10)
        self.create_subscription(Image, '/depth', self.on_dep, 5)
        self.create_timer(1.0 / a.fps, self.tick)
        print('waiting for all three topics...', flush=True)

    def on_img(self, m):
        if m.encoding == 'rgb8':
            self.img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)[:, :, ::-1]

    def on_tof(self, m):
        self.tof = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols)

    def on_dep(self, m):
        self.dep = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)

    def tick(self):
        if self.img is None or self.tof is None or self.dep is None:
            return
        if self.t0 is None:
            self.t0 = time.time()
            print('>>> RECORDING — drive the robot now <<<', flush=True)
        el = time.time() - self.t0
        if el > self.a.secs:
            raise SystemExit(0)

        cam = cv2.resize(self.img, (PW, PH))
        tv = np.isfinite(self.tof)
        tofc = cv2.resize(colorize(np.nan_to_num(self.tof), tv, self.a.tof_lo, self.a.tof_hi),
                          (PW, PH), interpolation=cv2.INTER_NEAREST)
        d = self.dep
        depc = cv2.resize(colorize(d, d > 0, self.a.d_lo, self.a.d_hi), (PW, PH))

        # Both readouts must describe the SAME patch of the world, or they are not
        # comparable -- the ToF grid centre and the image centre look ~50 px apart
        # (the camera aims downward, so the image centre sees nearer floor).
        tc = self.tof[14:18, 14:18]
        tc = tc[np.isfinite(tc)]
        y0, y1, x0, x1 = self.a.roi                    # where the centre ToF zones land
        dc = d[y0:y1, x0:x1]
        dc = dc[dc > 0]

        td_panel, n_obst = topdown(d, self.K, self.a.td_zmax, self.a.td_xhalf)

        cam = label(cam.copy(), 'Camera')
        tofc = label(tofc, f'ToF 32x32  ({self.a.tof_lo:.1f}-{self.a.tof_hi:.1f} m)',
                     f'centre zones {np.median(tc):.2f} m' if tc.size else '')
        depc = label(depc, f'Fused depth  ({self.a.d_lo:.1f}-{self.a.d_hi:.1f} m)',
                     f'same zones {np.median(dc):.2f} m' if dc.size else '')
        td_panel = label(td_panel, f'Top-down  ({self.a.td_zmax:.0f} m ahead)',
                         f'{n_obst} obstacle pts')

        grid = np.hstack([cam, tofc, depc, td_panel])
        bar = np.zeros((26, grid.shape[1], 3), np.uint8)
        w = int(grid.shape[1] * el / self.a.secs)
        cv2.rectangle(bar, (0, 10), (w, 16), (90, 210, 200), -1)
        cv2.putText(bar, f't = {el:4.1f}s', (grid.shape[1] - 88, 19), FONT, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imwrite(f"{self.a.dir}/f{self.n:05d}.jpg", np.vstack([grid, bar]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.n += 1
        if self.n % 30 == 0:
            print(f'  {el:4.1f}s  {self.n} frames', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    p.add_argument('--secs', type=float, default=30.0)
    p.add_argument('--fps', type=float, default=15.0)
    p.add_argument('--tof-lo', type=float, default=0.15)
    p.add_argument('--tof-hi', type=float, default=5.0)
    p.add_argument('--d-lo', type=float, default=0.15)
    p.add_argument('--d-hi', type=float, default=4.0)
    p.add_argument('--calib', required=True, help='calibration.yaml (for the top-down unprojection)')
    p.add_argument('--td-zmax', type=float, default=4.0, help='top-down forward extent (m)')
    p.add_argument('--td-xhalf', type=float, default=2.5, help='top-down half-width (m)')
    p.add_argument('--roi', type=int, nargs=4, default=[534, 576, 825, 855],
                   metavar=('Y0', 'Y1', 'X0', 'X1'),
                   help='image ROI the centre ToF zones project onto (measured with '
                        'centre_diag.py) -- keeps the two readouts comparable')
    a = p.parse_args()
    rclpy.init()
    n = Rec(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print(f'wrote {n.n} frames to {a.dir}')
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

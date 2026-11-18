#!/usr/bin/env python3
"""Record a synchronised Camera | ToF | Depth | Uncertainty clip to numbered frames for ffmpeg.

Same recorder as record_clip.py, with two changes for the ICRAE talk:
  * a sigma panel from /depth_var (per-pixel variance, 32FC1, m^2), shown as
    sigma = sqrt(var) on a FIXED log scale, so "bright" always means the same uncertainty
  * ONE shared colour scale for the ToF and fused-depth panels (--scale-lo/--scale-hi),
    so a colour means the same distance in both and the far-field ceiling is visible

Lives next to record_clip.py and reuses its colorize/label/topdown helpers.
Output: <dir>/f00000.jpg ... ; encode with
    ffmpeg -framerate 15 -i <dir>/f%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 18 <dir>.mp4
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_clip import colorize, label, topdown, PW, PH, FONT   # noqa: E402

SIG_CMAP = cv2.COLORMAP_VIRIDIS        # dark = confident, bright yellow = unsure


def sigma_panel(var, lo, hi):
    """Per-pixel sigma on a log scale. Invalid (non-finite or <= 0 variance) is drawn black."""
    ok = np.isfinite(var) & (var > 0)
    sig = np.sqrt(np.where(ok, var, 1.0))
    n = (np.log10(np.clip(sig, lo, hi)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    c = cv2.applyColorMap((np.clip(n, 0, 1) * 255).astype(np.uint8), SIG_CMAP)
    c[~ok] = (0, 0, 0)
    return c, sig, ok


def draw_sigma_bar(panel, lo, hi):
    """Small log colour bar, bottom-right, with 1 cm / 10 cm / 1 m ticks where in range."""
    x0, x1, y0, y1 = PW - 158, PW - 12, PH - 24, PH - 16
    ramp = np.linspace(0, 255, x1 - x0).astype(np.uint8)[None, :].repeat(y1 - y0, 0)
    panel[y0:y1, x0:x1] = cv2.applyColorMap(ramp, SIG_CMAP)
    cv2.rectangle(panel, (x0 - 1, y0 - 1), (x1, y1), (200, 200, 200), 1)
    for v, txt in ((0.01, '1cm'), (0.1, '10cm'), (1.0, '1m')):
        if lo <= v <= hi:
            x = int(x0 + (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) * (x1 - x0 - 1))
            cv2.line(panel, (x, y0 - 3), (x, y0 - 1), (220, 220, 220), 1)
            cv2.putText(panel, txt, (x - 12, y0 - 6), FONT, 0.33, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


class Rec(Node):
    def __init__(self, a):
        super().__init__('record_clip_sigma')
        self.a = a
        self.img = self.tof = self.dep = self.var = None
        self.n = 0
        self.t0 = None
        from ringfusion_perception.rectify import FisheyeRectifier
        from ringfusion_perception.perception_node import load_calib
        raw = load_calib(a.calib)
        r = raw['rectify']
        rect = FisheyeRectifier(raw['K'], raw['dist'], raw['model'],
                                size_in=(raw['img_w'], raw['img_h']),
                                size_out=(r['width'], r['height']),
                                balance=r['balance'], fov_scale=r['fov_scale'])
        self.K = np.asarray(rect.K_rect, np.float64).ravel()
        if self.K.size == 9:
            m = self.K.reshape(3, 3)
            self.K = np.array([m[0, 0], m[1, 1], m[0, 2], m[1, 2]])
        os.makedirs(a.dir, exist_ok=True)
        if a.npz_every:
            os.makedirs(os.path.join(a.dir, 'raw'), exist_ok=True)
        self.create_subscription(Image, a.image_topic, self.on_img, 5)
        self.create_subscription(ToFFrame, a.tof_topic, self.on_tof, 10)
        self.create_subscription(Image, a.depth_topic, self.on_dep, 5)
        self.create_subscription(Image, a.var_topic, self.on_var, 5)
        self.create_timer(1.0 / a.fps, self.tick)
        print(f'waiting for {a.image_topic} {a.tof_topic} {a.depth_topic} {a.var_topic} ...', flush=True)

    def on_img(self, m):
        if m.encoding == 'rgb8':
            self.img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)[:, :, ::-1]

    def on_tof(self, m):
        self.tof = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols)

    def on_dep(self, m):
        self.dep = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)

    def on_var(self, m):
        self.var = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)

    def tick(self):
        if self.img is None or self.tof is None or self.dep is None or self.var is None:
            return
        if self.t0 is None:
            self.t0 = time.time()
            print('>>> RECORDING, drive the robot now <<<', flush=True)
        el = time.time() - self.t0
        if el > self.a.secs:
            raise SystemExit(0)

        lo, hi = self.a.scale_lo, self.a.scale_hi
        d, var = self.dep, self.var
        y0, y1, x0, x1 = self.a.roi

        cam = label(cv2.resize(self.img, (PW, PH)).copy(), 'Camera')

        tv = np.isfinite(self.tof)
        tofc = cv2.resize(colorize(np.nan_to_num(self.tof), tv, lo, hi), (PW, PH),
                          interpolation=cv2.INTER_NEAREST)
        tc = self.tof[14:18, 14:18]
        tc = tc[np.isfinite(tc)]
        tofc = label(tofc, f'dToF 32x32  ({lo:g}-{hi:g} m)',
                     f'centre zones {np.median(tc):.2f} m' if tc.size else '')

        depc = cv2.resize(colorize(d, d > 0, lo, hi), (PW, PH))
        dc = d[y0:y1, x0:x1]
        dc = dc[dc > 0]
        depc = label(depc, f'Fused depth  ({lo:g}-{hi:g} m)',
                     f'same zones {np.median(dc):.2f} m' if dc.size else '')

        sc, sig, ok = sigma_panel(var, self.a.sig_lo, self.a.sig_hi)
        sigc = cv2.resize(sc, (PW, PH), interpolation=cv2.INTER_AREA)
        s_roi = sig[y0:y1, x0:x1][ok[y0:y1, x0:x1]]
        s_all = sig[ok]
        sub = ''
        if s_roi.size and s_all.size:
            sub = f'centre {100 * np.median(s_roi):.0f} cm   p90 {np.percentile(s_all, 90):.2f} m'
        sigc = label(sigc, f'Uncertainty  sigma ({self.a.sig_lo:g}-{self.a.sig_hi:g} m, log)', sub)
        sigc = draw_sigma_bar(sigc, self.a.sig_lo, self.a.sig_hi)

        panels = [cam, tofc, depc, sigc]
        if self.a.topdown:
            td, n_obst = topdown(d, self.K, self.a.td_zmax, self.a.td_xhalf)
            panels.append(label(td, f'Top-down  ({self.a.td_zmax:.0f} m ahead)', f'{n_obst} obstacle pts'))

        grid = np.hstack(panels)
        bar = np.zeros((26, grid.shape[1], 3), np.uint8)
        w = int(grid.shape[1] * el / self.a.secs)
        cv2.rectangle(bar, (0, 10), (w, 16), (90, 210, 200), -1)
        cv2.putText(bar, f't = {el:4.1f}s', (grid.shape[1] - 88, 19), FONT, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imwrite(f"{self.a.dir}/f{self.n:05d}.jpg", np.vstack([grid, bar]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if self.a.npz_every and self.n % self.a.npz_every == 0:
            np.savez_compressed(f"{self.a.dir}/raw/r{self.n:05d}.npz", rgb=self.img[:, :, ::-1],
                                tof=self.tof, depth=d, var=var, t=el)
        self.n += 1
        if self.n % 30 == 0:
            print(f'  {el:4.1f}s  {self.n} frames', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    p.add_argument('--secs', type=float, default=30.0)
    p.add_argument('--fps', type=float, default=15.0)
    p.add_argument('--scale-lo', type=float, default=0.15, help='shared ToF + depth colour scale (m)')
    p.add_argument('--scale-hi', type=float, default=5.0)
    p.add_argument('--sig-lo', type=float, default=0.01, help='sigma colour scale, log (m)')
    p.add_argument('--sig-hi', type=float, default=3.0)
    p.add_argument('--topdown', action='store_true', help='add the top-down panel as a 5th panel')
    p.add_argument('--td-zmax', type=float, default=4.0)
    p.add_argument('--td-xhalf', type=float, default=2.5)
    p.add_argument('--npz-every', type=int, default=0,
                   help='also save raw rgb/tof/depth/var every N frames (0 = off)')
    p.add_argument('--calib', required=True)
    p.add_argument('--roi', type=int, nargs=4, default=[534, 576, 825, 855],
                   metavar=('Y0', 'Y1', 'X0', 'X1'))
    p.add_argument('--image-topic', default='/image')
    p.add_argument('--tof-topic', default='/tof')
    p.add_argument('--depth-topic', default='/depth')
    p.add_argument('--var-topic', default='/depth_var')
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

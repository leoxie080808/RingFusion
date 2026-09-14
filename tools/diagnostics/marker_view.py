#!/usr/bin/env python3
"""Live marker-placement aid: rectified camera with the ToF cone drawn ON it.

Publishes an image topic for web_video_server -- no cv2.imshow, no local window, so it
works over SSH with no display attached.

WHY THIS EXISTS. dual_view and live_view show the camera and the ToF as SEPARATE panels,
the ToF at 32x32 zone resolution. Deciding "is this marker inside the cone?" then means
mentally mapping between two panels at different scales, and getting it wrong is not
hypothetical: the 2026-08-05 tape session produced two points that sit 12-27 px from an
older marker but at a completely different depth, and which of the two surfaces was
intended can no longer be recovered. Drawing the boundary on the image the marker appears
in removes the judgement call.

WHAT IS DRAWN
  yellow rect    the ToF cone footprint, from the calibrated FOV -- inside is covered
  dashed rects   +/- edge_deg around that boundary; between them is the "cone edge" band,
                 which the session needs ~3 markers in, placed deliberately rather than
                 by accident
  coloured dots  every zone returning a valid range, at the pixel it projects to, tinted
                 by depth. A marker inside the rect with no dots near it is inside the
                 cone but UNSUPPORTED -- a real and useful distinction when choosing
                 where the hard cases go
  header         valid-zone count, so a thin scene is obvious before capturing anything

Zones with no return cannot be drawn: their pixel position is computed from the measured
range, so without a range there is no position. That is why coverage shows as missing
dots rather than as a marked gap.

    python3 tools/diagnostics/marker_view.py \
        --calib ros2_ws/src/ringfusion_bringup/config/calibration.yaml

    then browse  http://<jetson-ip>:8080/stream?topic=/marker_view
    (needs `ros2 run web_video_server web_video_server` up)
"""
import argparse
import os
import sys

import cv2
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'training'))

from anchoring_bridge import calib_from_yaml                       # noqa: E402
from ringfusion_perception import geometry as geo                  # noqa: E402
from ringfusion_perception.rectify import FisheyeRectifier         # noqa: E402
from ringfusion_perception.perception_node import load_calib       # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def cone_rect(calib, half_h_deg, half_v_deg):
    """Pixel rect for a given angular half-extent. Same construction tape_eval uses."""
    fx, fy, cx, cy = [float(v) for v in np.asarray(calib['K'], float).ravel()[:4]]
    dx = fx * np.tan(np.deg2rad(half_h_deg))
    dy = fy * np.tan(np.deg2rad(half_v_deg))
    return int(cx - dx), int(cy - dy), int(cx + dx), int(cy + dy)


def dashed_rect(img, x0, y0, x1, y1, colour, dash=18, thick=1):
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), colour, thick)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), colour, thick)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), colour, thick)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), colour, thick)


class MarkerView(Node):
    def __init__(self, a):
        super().__init__('marker_view')
        self.a = a
        raw = load_calib(a.calib)
        r = raw['rectify']
        self.rect = FisheyeRectifier(raw['K'], raw['dist'], raw['model'],
                                     size_in=(raw['img_w'], raw['img_h']),
                                     size_out=(r['width'], r['height']),
                                     balance=r['balance'], fov_scale=r['fov_scale'])
        self.calib = calib_from_yaml(a.calib, train_size=(r['height'], r['width']))
        self.rgb = None
        self.tof = None
        self.create_subscription(Image, a.image_topic, self.on_img, 1)
        self.create_subscription(ToFFrame, a.tof_topic, self.on_tof, 1)
        self.pub = self.create_publisher(Image, a.out_topic, 1)
        self.create_timer(1.0 / max(a.rate, 0.5), self.tick)

        h, w = r['height'], r['width']
        self.cone = cone_rect(self.calib, self.calib['fov_h'] / 2.0, self.calib['fov_v'] / 2.0)
        self.inner = cone_rect(self.calib, self.calib['fov_h'] / 2.0 - a.edge_deg,
                               self.calib['fov_v'] / 2.0 - a.edge_deg)
        self.outer = cone_rect(self.calib, self.calib['fov_h'] / 2.0 + a.edge_deg,
                               self.calib['fov_v'] / 2.0 + a.edge_deg)
        self.get_logger().info(
            f'cone rect x {self.cone[0]}..{self.cone[2]}, y {self.cone[1]}..{self.cone[3]} '
            f'in the {w}x{h} rectified frame; publishing {a.out_topic}')

    def on_img(self, m):
        if m.encoding != 'rgb8':
            return
        buf = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        self.rgb = buf.copy()

    def on_tof(self, m):
        if len(m.dist_m) == m.rows * m.cols:
            self.tof = (np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols).copy(),
                        int(m.cols), int(m.rows))

    def tick(self):
        if self.rgb is None:
            return
        full = cv2.cvtColor(self.rect.rectify(self.rgb), cv2.COLOR_RGB2BGR)
        sc = float(self.a.scale)
        view = (cv2.resize(full, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                if sc != 1.0 else full)
        h, w = view.shape[:2]
        n_valid = 0

        if self.tof is not None:
            dist, cols, rows = self.tof
            valid = np.isfinite(dist) & (dist > 0.05) & (dist < 10.0)
            n_valid = int(valid.sum())
            if n_valid:
                proj = geo.project_zone_to_pixel(
                    dist, valid, cols, rows, self.calib['fov_h'], self.calib['fov_v'],
                    self.calib['T_cam_tof'], self.calib['K'], self.calib['dist'],
                    model=self.calib['model'])
                uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']
                d = dist.reshape(-1)
                keep = (ok & valid.reshape(-1) & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
                        & (z > 0))
                u = np.round(uv[keep, 0]).astype(int)
                v = np.round(uv[keep, 1]).astype(int)
                dk = d[keep]
                u = np.round(u * sc).astype(int)
                v = np.round(v * sc).astype(int)
                inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                u, v, dk = u[inb], v[inb], dk[inb]
                # Tint by depth so the scene's range spread is visible at a glance --
                # a session needs markers from 0.3 to 4 m and this shows whether the
                # scene actually offers that before any tape comes out.
                lo, hi = self.a.min_range, self.a.max_range
                t = np.clip((dk - lo) / max(hi - lo, 1e-6), 0, 1)
                cols_ = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                cols_ = cols_.reshape(-1, 3)
                for i in range(len(u)):
                    cv2.circle(view, (int(u[i]), int(v[i])), 3,
                               tuple(int(c) for c in cols_[i]), -1)

        sr = lambda r: tuple(int(round(c * sc)) for c in r)
        x0, y0, x1, y1 = sr(self.cone)
        dashed_rect(view, *sr(self.outer), (0, 200, 255))
        dashed_rect(view, *sr(self.inner), (0, 200, 255))
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(view, 'ToF cone', (x0 + 6, y0 - 8), FONT, 0.6, (0, 255, 255), 2)
        cv2.putText(view, f'edge band +/-{self.a.edge_deg:g} deg',
                    (sr(self.outer)[0] + 6, sr(self.outer)[1] - 8), FONT, 0.5, (0, 200, 255), 1)

        bar = f'valid zones {n_valid}  |  cone {x1-x0}x{y1-y0} px  |  dots = ToF returns, tinted by depth'
        cv2.rectangle(view, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(view, bar, (10, 24), FONT, 0.6, (255, 255, 255), 2)
        if n_valid and n_valid < 400:
            cv2.putText(view, 'LOW COVERAGE -- scene is thinner than deployment (~876)',
                        (10, 62), FONT, 0.7, (0, 165, 255), 2)

        # Snapshot mode: write one rendered frame and stop. Checking a scene before a
        # session is the same picture as steering marker placement during one, so it is
        # the same renderer rather than a second one that could drift out of agreement.
        if self.a.snapshot:
            self.n_ticks = getattr(self, 'n_ticks', 0) + 1
            # Skip the first few ticks so auto-exposure has settled and at least one ToF
            # frame has arrived -- a snapshot of a dark frame with zero zones is useless.
            if self.n_ticks >= 4 and n_valid:
                cv2.imwrite(self.a.snapshot, view)
                print(f'wrote {self.a.snapshot}  ({w}x{h}, {n_valid} valid zones)')
                self.done = True
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = h, w
        msg.encoding, msg.is_bigendian, msg.step = 'bgr8', 0, w * 3
        msg.data = view.tobytes()
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True)
    ap.add_argument('--image-topic', default='/image')
    ap.add_argument('--tof-topic', default='/tof')
    ap.add_argument('--out-topic', default='/marker_view')
    ap.add_argument('--rate', type=float, default=5.0,
                    help='publish Hz -- low on purpose, this is a placement aid and should '
                         'not take compute from the node it is watching')
    ap.add_argument('--edge-deg', type=float, default=3.0,
                    help='half-width of the cone-edge band, degrees')
    ap.add_argument('--scale', type=float, default=0.5,
                    help='publish at this fraction of full resolution -- a full-size bgr8 '
                         'frame is ~6 MB per message and the view is only eyeballed')
    ap.add_argument('--snapshot', default='',
                   help='render one frame to this PNG and exit, instead of publishing')
    ap.add_argument('--min-range', type=float, default=0.2)
    ap.add_argument('--max-range', type=float, default=4.0)
    a = ap.parse_args()
    rclpy.init()
    node = MarkerView(a)
    try:
        if a.snapshot:
            t0 = time.time()
            while not getattr(node, 'done', False) and time.time() - t0 < 30.0:
                rclpy.spin_once(node, timeout_sec=0.2)
            if not getattr(node, 'done', False):
                print('no snapshot written -- no image, or no valid ToF zones in 30 s')
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

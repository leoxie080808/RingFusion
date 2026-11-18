#!/usr/bin/env python3
"""Record paired frames for an offline map. Presentation only, not in the paper.

For every /depth the perception node publishes, this saves the rectified camera frame the
node used, the metric depth, its variance and the stamps. The map is built afterwards from
these files, so nothing about mapping has to work live on the robot.

Pairing. perception_node stamps /depth and /depth_var with the ToF frame's stamp and runs on
the newest /image it holds when that ToF frame arrives. So each depth is paired with the
newest image whose stamp is at or before the depth stamp (within --max-skew). The error is at
most one camera period (33 ms at 30 fps), which is fine at walking speed.

Output in --dir:
  rgb/NNNNN.jpg     rectified colour, resized to the depth grid
  depth/NNNNN.npy   metric depth, float16, metres (0 = no depth)
  var/NNNNN.npy     depth variance, float16, m^2 (missing if /depth_var is not published)
  frames.csv        n, t_depth, t_image, skew_ms, has_var
  meta.json         intrinsics of the saved grid, rectifier settings, args, counts
  calibration.yaml  copy of the calibration used

Usage, with the stack already running (see ros2_ws/README.md):
  python3 record_map_frames.py --dir ~/icrae_rerun/map1 --secs 60 \
      --calib ~/RingFusion/ros2_ws/src/ringfusion_bringup/config/calibration.yaml
"""
import argparse
import collections
import csv
import json
import os
import queue
import shutil
import sys
import threading
import time

import cv2
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_perception'))
from ringfusion_perception.rectify import FisheyeRectifier   # noqa: E402  (numpy + cv2 only)


def make_rectifier(calib_path):
    """Same Stage 1 rectifier the perception node builds, from the same calibration file."""
    with open(calib_path) as f:
        c = yaml.safe_load(f)
    cam, rect = c['camera'], c.get('rectify', {})
    r = FisheyeRectifier((cam['fx'], cam['fy'], cam['cx'], cam['cy']), cam.get('dist', [0, 0, 0, 0]),
                         cam.get('model', 'fisheye'), size_in=(cam['width'], cam['height']),
                         size_out=(rect.get('width', cam['width']), rect.get('height', cam['height'])),
                         balance=rect.get('balance', 0.0), fov_scale=rect.get('fov_scale', 1.0))
    return r, cam, rect


class Recorder:
    """ROS-free core: pairing, decimation and a background writer. The node only feeds it."""

    def __init__(self, a, rectifier, cam, rect):
        self.a, self.rect = a, rectifier
        self.cam, self.rect_cfg = cam, rect
        for sub in ('rgb', 'depth', 'var'):
            os.makedirs(os.path.join(a.dir, sub), exist_ok=True)
        self.images = collections.deque(maxlen=16)          # (t, raw rgb8 array)
        self.pending = collections.OrderedDict()            # t_depth -> (depth, t_img, img, t_recv)
        self.q = queue.Queue(maxsize=a.queue)
        self.n = self.dropped = self.unpaired = 0
        self.bytes = 0
        self.t0 = None
        self.done = False
        self.depth_shape = None
        self.csv_f = open(os.path.join(a.dir, 'frames.csv'), 'w', newline='')
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(['n', 't_depth', 't_image', 'skew_ms', 'has_var'])
        self.worker = threading.Thread(target=self._write_loop, daemon=True)
        self.worker.start()

    # ---- inputs -------------------------------------------------------------------------
    def add_image(self, t, rgb):
        self.images.append((t, rgb))

    def add_depth(self, t, depth):
        if self.done:
            return
        img = None
        for ti, im in reversed(self.images):
            if ti <= t + 1e-4:
                img = (ti, im)
                break
        if img is None or t - img[0] > self.a.max_skew:
            self.unpaired += 1
            return
        self.pending[t] = (depth, img[0], img[1], time.monotonic())
        if not self.a.wait_var:
            self._emit(t, None)

    def add_var(self, t, var):
        if t in self.pending:
            self._emit(t, var)

    def flush_stale(self, age=0.5):
        """Depth whose variance never came: save it without variance."""
        now = time.monotonic()
        for t in [t for t, v in self.pending.items() if now - v[3] > age]:
            self._emit(t, None)

    # ---- internals ----------------------------------------------------------------------
    def _emit(self, t, var):
        depth, ti, img, _ = self.pending.pop(t)
        if self.t0 is None:
            self.t0 = t
            print(f'first paired frame, recording {self.a.secs:g} s', flush=True)
        if t - self.t0 > self.a.secs:
            self.done = True
            return
        try:
            self.q.put_nowait((self.n, t, ti, depth, var, img))
            self.n += 1
        except queue.Full:
            self.dropped += 1

    def _write_loop(self):
        d = self.a.decimate
        while True:
            item = self.q.get()
            if item is None:
                break
            n, t, ti, depth, var, img = item
            rgb = self.rect.rectify(img)
            h, w = depth.shape
            dd = depth[::d, ::d].astype(np.float16)
            dd[~np.isfinite(dd)] = 0
            self.depth_shape = dd.shape
            col = cv2.resize(rgb, (dd.shape[1], dd.shape[0]), interpolation=cv2.INTER_AREA)
            pj = os.path.join(self.a.dir, 'rgb', f'{n:05d}.jpg')
            cv2.imwrite(pj, col[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            pd = os.path.join(self.a.dir, 'depth', f'{n:05d}.npy')
            np.save(pd, dd)
            self.bytes += os.path.getsize(pj) + os.path.getsize(pd)
            if var is not None:
                pv = os.path.join(self.a.dir, 'var', f'{n:05d}.npy')
                vv = var[::d, ::d].astype(np.float16)
                np.save(pv, vv)
                self.bytes += os.path.getsize(pv)
            self.csv.writerow([n, f'{t:.6f}', f'{ti:.6f}', f'{1000 * (t - ti):.1f}', int(var is not None)])
            if (n + 1) % 30 == 0:
                self.csv_f.flush()
                print(f'  {t - self.t0:5.1f} s  {n + 1} frames  {self.bytes / 1e6:6.0f} MB  '
                      f'dropped {self.dropped}  unpaired {self.unpaired}  queue {self.q.qsize()}', flush=True)

    def close(self):
        self.q.put(None)
        self.worker.join()
        self.csv_f.close()
        d = self.a.decimate
        fx, fy, cx, cy = self.rect.K_rect
        k = [fx / d, fy / d, (cx + 0.5) / d - 0.5, (cy + 0.5) / d - 0.5]
        meta = {
            'what': 'paired frames for an offline map, presentation only, not in the paper',
            'frames': self.n, 'dropped': self.dropped, 'unpaired': self.unpaired,
            'depth_shape_hw': list(self.depth_shape) if self.depth_shape else None,
            'K_saved_grid': k, 'K_rect_full': list(self.rect.K_rect), 'decimate': d,
            'rectify': self.rect_cfg, 'camera_model': self.cam.get('model', 'fisheye'),
            'units': {'depth': 'm', 'var': 'm^2'}, 'args': vars(self.a),
        }
        with open(os.path.join(self.a.dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        shutil.copy(self.a.calib, os.path.join(self.a.dir, 'calibration.yaml'))
        print(f'saved {self.n} frames ({self.bytes / 1e6:.0f} MB) to {self.a.dir}, '
              f'dropped {self.dropped}, unpaired {self.unpaired}', flush=True)


def stamp_s(h):
    return h.stamp.sec + 1e-9 * h.stamp.nanosec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', required=True)
    p.add_argument('--secs', type=float, default=60.0)
    p.add_argument('--calib', required=True)
    p.add_argument('--decimate', type=int, default=4, help='keep every Nth depth pixel (4 = 410x308, the cloud grid)')
    p.add_argument('--max-skew', type=float, default=0.08, help='max image-to-depth age, s')
    p.add_argument('--queue', type=int, default=40, help='frames buffered before dropping')
    p.add_argument('--no-wait-var', dest='wait_var', action='store_false',
                   help='save depth without waiting for /depth_var')
    p.add_argument('--image-topic', default='/image')
    p.add_argument('--depth-topic', default='/depth')
    p.add_argument('--var-topic', default='/depth_var')
    a = p.parse_args()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    rectifier, cam, rect = make_rectifier(a.calib)
    rec = Recorder(a, rectifier, cam, rect)

    def as_array(msg, dtype, ch):
        arr = np.frombuffer(msg.data, dtype=dtype)
        return arr.reshape(msg.height, msg.width, ch).copy() if ch > 1 else arr.reshape(msg.height, msg.width).copy()

    class MapFrames(Node):
        def __init__(self):
            super().__init__('record_map_frames')
            self.create_subscription(Image, a.image_topic, lambda m: rec.add_image(stamp_s(m.header), as_array(m, np.uint8, 3)), 5)
            self.create_subscription(Image, a.depth_topic, lambda m: rec.add_depth(stamp_s(m.header), as_array(m, np.float32, 1)), 5)
            self.create_subscription(Image, a.var_topic, lambda m: rec.add_var(stamp_s(m.header), as_array(m, np.float32, 1)), 5)
            self.create_timer(0.1, rec.flush_stale)
            print(f'waiting for {a.image_topic} {a.depth_topic} {a.var_topic} ...', flush=True)

    rclpy.init()
    node = MapFrames()
    try:
        while rclpy.ok() and not rec.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    rec.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

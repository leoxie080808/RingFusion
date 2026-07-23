"""Log paired (rectified image, 32x32 ToF) samples for Network B (residual) training.

Subscribes: image  (sensor_msgs/Image, rgb8)      from camera_node
            tof    (ringfusion_msgs/ToFFrame)     from tof_driver

Writes the exact on-disk format training/residual_real_data.py expects, matched by stem:
    out_dir/rgb/000123.png    the RECTIFIED (pinhole) image -- SAME FisheyeRectifier +
                              calibration.yaml the robot runs, so the net trains on the
                              true deployment representation (see calib_from_yaml).
    out_dir/tof/000123.npz    np.savez(dist_m=(rows,cols) float32 NaN-invalid,
                                        confidence=(rows,cols) uint8)

Held-out-anchor training (train_residual.py --real) turns each frame's ToF into both the
net's anchor input and a sparse supervision target, so we do NOT need dense GT -- the ToF
IS the measurement. See training/README section 2a.

Two modes (param `mode`):
  station     (default) live preview; press 'c' to capture ONE station -- averages the
              last `avg` ToF frames (denoises + lifts coverage) with the current image.
              Best pairs: stop, hold still, capture. 'q'/ESC quits.
  continuous  auto-save deduped, sharp, well-synced pairs as you push slowly (headless-ok).

Only saves when a ToF frame is within `sync_thresh` seconds of the image, so a moving
robot never banks a mismatched pair. Re-running appends (continues the numbering).

Parameters:
  out_dir (str)        root; writes out_dir/rgb and out_dir/tof   (default data/real)
  calib (str)          path to calibration.yaml
  mode (str)           'station' | 'continuous'                    (default station)
  avg (int)            ToF frames to average per station           (default 3)
  sync_thresh (float)  max |t_img - t_tof| seconds to pair         (default 0.08)
  target (int)         stop after this many pairs on disk          (default 5000)
  dedup_thresh (float) continuous: mean abs gray diff for "new"    (default 8.0)
  min_interval (float) continuous: min seconds between saves       (default 0.5)
  blur_thresh (float)  continuous: min variance-of-Laplacian       (default 60.0)
  preview (bool)       show the live window (needs the monitor)    (default True)
"""
import glob
import os
import re
import time
import warnings
from collections import deque

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

from .rectify import FisheyeRectifier
from .perception_node import load_calib

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = "paired_logger  |  c=capture station  q/ESC=quit"


class PairedLogger(Node):
    def __init__(self):
        super().__init__('paired_logger')
        self.declare_parameter('out_dir', 'data/real')
        self.declare_parameter('calib', 'calibration.yaml')
        self.declare_parameter('mode', 'station')
        self.declare_parameter('avg', 3)
        self.declare_parameter('sync_thresh', 0.08)
        self.declare_parameter('target', 5000)
        self.declare_parameter('dedup_thresh', 8.0)
        self.declare_parameter('min_interval', 0.5)
        self.declare_parameter('blur_thresh', 60.0)
        self.declare_parameter('preview', True)

        self.out_dir = self.get_parameter('out_dir').value
        self.mode = self.get_parameter('mode').value
        self.avg = int(self.get_parameter('avg').value)
        self.sync_thresh = float(self.get_parameter('sync_thresh').value)
        self.target = int(self.get_parameter('target').value)
        self.dedup_thresh = float(self.get_parameter('dedup_thresh').value)
        self.min_interval = float(self.get_parameter('min_interval').value)
        self.blur_thresh = float(self.get_parameter('blur_thresh').value)
        self.preview = bool(self.get_parameter('preview').value)

        self.rgb_dir = os.path.join(self.out_dir, 'rgb')
        self.tof_dir = os.path.join(self.out_dir, 'tof')
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.tof_dir, exist_ok=True)

        c = load_calib(self._resolve_calib(self.get_parameter('calib').value))
        r = c['rectify']
        self.rect = FisheyeRectifier(
            c['K'], c['dist'], c['model'], size_in=(c['img_w'], c['img_h']),
            size_out=(r['width'], r['height']), balance=r['balance'],
            fov_scale=r['fov_scale'])

        # resume: continue numbering after whatever is already on disk
        existing = glob.glob(os.path.join(self.rgb_dir, '*.png'))
        idxs = [int(m.group(1)) for m in
                (re.search(r'(\d+)\.png$', p) for p in existing) if m]
        self.next_idx = (max(idxs) + 1) if idxs else 0
        self.on_disk = len(existing)
        self.saved = 0

        # latest rectified image + a short ring of recent ToF frames (t, dist, conf)
        self.img = None
        self.img_t = 0.0
        self.tof_buf = deque(maxlen=16)
        self.ref_small = None
        self.last_save_t = 0.0

        if self.preview:
            try:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            except cv2.error as e:
                self.get_logger().warn(f"no display ({e}); forcing headless continuous mode")
                self.preview = False
                self.mode = 'continuous'
        if not self.preview and self.mode == 'station':
            self.get_logger().warn("station mode needs the preview window; using continuous")
            self.mode = 'continuous'

        self.create_subscription(ToFFrame, 'tof', self.on_tof, 10)
        self.create_subscription(Image, 'image', self.on_image, 5)
        self.get_logger().info(
            f"paired_logger [{self.mode}]: image+tof -> {self.out_dir}/ | "
            f"identity={self.rect.is_identity} | {self.on_disk} on disk, target {self.target}")

    def _resolve_calib(self, path):
        if os.path.exists(path):
            return path
        try:
            from ament_index_python.packages import get_package_share_directory
            cand = os.path.join(get_package_share_directory('ringfusion_bringup'),
                                'config', 'calibration.yaml')
            if os.path.exists(cand):
                self.get_logger().info(f"calib '{path}' not found; using {cand}")
                return cand
        except Exception as e:
            self.get_logger().warn(f"could not locate bringup calib: {e}")
        return path

    # --- ToF ---------------------------------------------------------------
    def on_tof(self, msg):
        rows, cols = int(msg.rows), int(msg.cols)
        dist = np.asarray(msg.dist_m, np.float32).reshape(rows, cols)
        conf = np.asarray(msg.confidence, np.uint8).reshape(rows, cols)
        self.tof_buf.append((time.monotonic(), dist, conf))

    def _synced_tof(self, n=1):
        """Return up to n most-recent ToF frames within sync_thresh of the current
        image, newest first, or [] if the latest is too stale (robot moved)."""
        if self.img is None or not self.tof_buf:
            return []
        cand = [(t, d, c) for (t, d, c) in self.tof_buf
                if abs(t - self.img_t) <= self.sync_thresh]
        cand.sort(key=lambda x: x[0], reverse=True)
        return cand[:n]

    def _merge_tof(self, frames):
        """NaN-aware average of several ToF frames (denoise + fill dropouts);
        confidence = per-zone max. `frames` is a list of (t, dist, conf)."""
        dists = np.stack([f[1] for f in frames], 0)
        confs = np.stack([f[2] for f in frames], 0)
        with warnings.catch_warnings():                 # all-NaN zones -> NaN, no warning
            warnings.simplefilter('ignore', RuntimeWarning)
            dist = np.nanmean(dists, axis=0).astype(np.float32)
        conf = confs.max(axis=0).astype(np.uint8)
        return dist, conf

    # --- image / preview / capture ----------------------------------------
    def _is_new(self, rect_bgr):
        gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (128, 128)).astype(np.float32)
        if self.ref_small is None:
            return True, small
        return float(np.abs(small - self.ref_small).mean()) > self.dedup_thresh, small

    def _sharpness(self, rect_bgr):
        gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] != 480:
            s = 480.0 / gray.shape[0]
            gray = cv2.resize(gray, (int(gray.shape[1] * s), 480))
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _save(self, rect_bgr, dist, conf, small=None):
        stem = f"{self.next_idx:06d}"
        cv2.imwrite(os.path.join(self.rgb_dir, stem + '.png'), rect_bgr)
        np.savez(os.path.join(self.tof_dir, stem + '.npz'), dist_m=dist, confidence=conf)
        self.next_idx += 1
        self.saved += 1
        if small is not None:
            self.ref_small = small
        self.last_save_t = time.time()
        if self.saved % 25 == 0 or self.saved == 1:
            n_valid = int(np.isfinite(dist).sum())
            self.get_logger().info(
                f"  saved {self.on_disk + self.saved}/{self.target}  ({n_valid} valid zones)")

    def on_image(self, msg):
        if msg.encoding != 'rgb8':
            self.get_logger().warn(f"expected rgb8, got {msg.encoding}")
            return
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        bgr = rgb[:, :, ::-1].copy()
        self.img = self.rect.rectify(bgr)
        self.img_t = time.monotonic()

        synced = self._synced_tof(n=1)
        is_new, small = self._is_new(self.img)
        sharp = self._sharpness(self.img)
        sharp_ok = sharp >= self.blur_thresh

        # continuous: auto-save new + sharp + spaced + synced pairs
        if (self.mode == 'continuous' and synced and is_new and sharp_ok
                and (time.time() - self.last_save_t) >= self.min_interval):
            self._save(self.img, synced[0][1], synced[0][2], small)

        if self.preview:
            self._draw(is_new, sharp, sharp_ok, synced)

    def _draw(self, is_new, sharp, sharp_ok, synced):
        disp = self.img.copy()
        total = self.on_disk + self.saved
        cv2.putText(disp, f"[{total}/{self.target}]  {self.mode.upper()}", (24, 44),
                    FONT, 1.1, (0, 165, 255) if self.mode == 'continuous' else (0, 255, 0),
                    3, cv2.LINE_AA)
        if synced:
            dt = abs(synced[0][0] - self.img_t) * 1e3
            nv = int(np.isfinite(synced[0][1]).sum())
            sync_txt, sync_col = f"ToF synced  dt {dt:.0f}ms  {nv} zones", (0, 255, 0)
        else:
            sync_txt, sync_col = "NO synced ToF (stale/none)", (0, 0, 255)
        cv2.putText(disp, sync_txt, (24, 88), FONT, 0.9, sync_col, 2, cv2.LINE_AA)
        blur_tag = f"sharp {sharp:.0f}" if sharp_ok else f"BLURRY {sharp:.0f}"
        cv2.putText(disp, f"{'NEW' if is_new else 'similar'}   {blur_tag}", (24, 128),
                    FONT, 0.8, (0, 255, 0) if (is_new and sharp_ok) else (0, 0, 255),
                    2, cv2.LINE_AA)
        cv2.putText(disp, "c capture station   q quit", (24, disp.shape[0] - 24),
                    FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            self.get_logger().info(f"done: {self.saved} saved this run, {total} total.")
            rclpy.shutdown()
        elif key == ord('c'):
            self._capture_station()

        if total >= self.target:
            self.get_logger().info(f"reached target {self.target}. Stopping.")
            rclpy.shutdown()

    def _capture_station(self):
        frames = self._synced_tof(n=self.avg)
        if not frames:
            self.get_logger().warn("no synced ToF frame -- hold still and try again")
            return
        dist, conf = self._merge_tof(frames)
        _, small = self._is_new(self.img)
        self._save(self.img, dist, conf, small)
        self.get_logger().info(
            f"station captured (averaged {len(frames)} ToF frame(s), "
            f"{int(np.isfinite(dist).sum())} valid zones)")


def main(args=None):
    rclpy.init(args=args)
    node = PairedLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

"""Collect RECTIFIED training frames for backbone distillation.

Subscribes: image   (sensor_msgs/Image, rgb8)   from camera_node

The frames come off `/image`, which camera_node has ALREADY color-corrected
(gray-world white balance + contrast), and this node then rectifies each with the
SAME FisheyeRectifier + calibration.yaml the perception pipeline uses. So every
saved frame goes through the exact color-correct -> rectify path the robot runs at
inference -- the teacher (cache_teacher.py) trains on the true deployment
distribution. Output is a folder of rectified PNGs, ready for `cache_teacher.py`.

Live preview (needs the Jetson's monitor). Controls, shown in the window:
  SPACE / y   save the current rectified frame
  c           toggle CONTINUOUS auto-capture (saves deduped frames as you walk)
  q / ESC     quit

Dedup: a frame is "new" only if it differs enough from the last SAVED frame
(mean abs gray diff > dedup_thresh). Auto-capture uses this to bank variety, not
duplicates; manual save shows a NEW/similar hint but always saves on your keypress.
Re-running appends -- it continues numbering after the frames already in out_dir.

Parameters:
  out_dir (str)        where to write frame_000000.png ...   (default data/rect)
  calib (str)          path to calibration.yaml
  target (int)         stop after this many total frames on disk (default 20000)
  dedup_thresh (float) mean abs gray diff (0-255) to count as a new frame (default 8.0)
  min_interval (float) min seconds between auto-capture saves (default 0.3)
  blur_thresh (float)  min sharpness (variance-of-Laplacian) for auto-capture to save
                       a frame -- rejects motion blur while walking (default 60.0).
                       Watch the live "sharp NNN" readout and tune to your scene/light.
"""
import glob
import os
import re
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from .rectify import FisheyeRectifier
from .perception_node import load_calib

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = "collect_frames  |  SPACE/y=save  c=continuous  q/ESC=quit"


class CollectFrames(Node):
    def __init__(self):
        super().__init__('collect_frames')
        self.declare_parameter('out_dir', 'data/rect')
        self.declare_parameter('calib', 'calibration.yaml')
        self.declare_parameter('target', 20000)
        self.declare_parameter('dedup_thresh', 8.0)
        self.declare_parameter('min_interval', 0.3)
        self.declare_parameter('blur_thresh', 60.0)

        self.out_dir = self.get_parameter('out_dir').value
        self.target = int(self.get_parameter('target').value)
        self.dedup_thresh = float(self.get_parameter('dedup_thresh').value)
        self.min_interval = float(self.get_parameter('min_interval').value)
        self.blur_thresh = float(self.get_parameter('blur_thresh').value)
        os.makedirs(self.out_dir, exist_ok=True)

        c = load_calib(self._resolve_calib(self.get_parameter('calib').value))
        r = c['rectify']
        self.rect = FisheyeRectifier(
            c['K'], c['dist'], c['model'], size_in=(c['img_w'], c['img_h']),
            size_out=(r['width'], r['height']), balance=r['balance'],
            fov_scale=r['fov_scale'])

        # resume: continue numbering after whatever is already on disk
        existing = glob.glob(os.path.join(self.out_dir, 'frame_*.png'))
        idxs = [int(m.group(1)) for m in
                (re.search(r'frame_(\d+)\.png$', p) for p in existing) if m]
        self.next_idx = (max(idxs) + 1) if idxs else 0
        self.on_disk = len(existing)      # count already collected
        self.saved = 0                    # saved this session
        self.continuous = False
        self.last_save_t = 0.0
        self.ref_small = None             # downsampled gray of the last SAVED frame

        try:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            self.get_logger().error(
                f"No display for the preview window ({e}). This tool needs the "
                f"Jetson's monitor. Aborting.")
            raise SystemExit(1)

        self.create_subscription(Image, 'image', self.on_image, 5)
        self.get_logger().info(
            f"collect_frames: /image -> {self.out_dir}/  | identity={self.rect.is_identity} "
            f"| {self.on_disk} already on disk, target {self.target}")

    def _resolve_calib(self, path):
        """Use the given calib path if it exists; otherwise fall back to the
        installed ringfusion_bringup config, so `ros2 run collect_frames` works with
        no -p calib and from any directory."""
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
        return path   # let load_calib raise a clear FileNotFoundError

    def _is_new(self, rect_bgr):
        """True if rect_bgr differs enough from the last saved frame."""
        gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (128, 128)).astype(np.float32)
        if self.ref_small is None:
            return True, small
        diff = float(np.abs(small - self.ref_small).mean())
        return diff > self.dedup_thresh, small

    def _sharpness(self, rect_bgr):
        """Variance of the Laplacian -- a standard focus/motion-blur measure. Higher
        is sharper. Computed on a fixed 480-tall gray so the number is stable across
        resolutions (a moving/blurred frame drops sharply below a static one)."""
        gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] != 480:
            s = 480.0 / gray.shape[0]
            gray = cv2.resize(gray, (int(gray.shape[1] * s), 480))
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _save(self, rect_bgr, small):
        path = os.path.join(self.out_dir, f"frame_{self.next_idx:06d}.png")
        cv2.imwrite(path, rect_bgr)       # clean frame, no overlay
        self.next_idx += 1
        self.saved += 1
        self.ref_small = small
        self.last_save_t = time.time()
        if self.saved % 50 == 0 or self.saved == 1:
            self.get_logger().info(f"  saved {self.on_disk + self.saved}/{self.target}")

    def on_image(self, msg):
        if msg.encoding != 'rgb8':
            self.get_logger().warn(f"expected rgb8, got {msg.encoding}")
            return
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        bgr = rgb[:, :, ::-1].copy()
        rect = self.rect.rectify(bgr)     # clean frame we save/preview
        is_new, small = self._is_new(rect)
        sharp = self._sharpness(rect)
        sharp_ok = sharp >= self.blur_thresh

        # continuous auto-capture: save frames that are new, sharp, and spaced out
        if (self.continuous and is_new and sharp_ok
                and (time.time() - self.last_save_t) >= self.min_interval):
            self._save(rect, small)

        total = self.on_disk + self.saved
        disp = rect.copy()
        mode = "CONTINUOUS" if self.continuous else "manual"
        mode_color = (0, 165, 255) if self.continuous else (0, 255, 0)
        cv2.putText(disp, f"[{total}/{self.target}]  {mode}", (24, 44),
                    FONT, 1.1, mode_color, 3, cv2.LINE_AA)
        tag = "NEW" if is_new else "similar to last"
        blur_tag = f"sharp {sharp:.0f}" if sharp_ok else f"BLURRY {sharp:.0f}"
        ok = is_new and sharp_ok
        cv2.putText(disp, f"{tag}   {blur_tag}", (24, 88), FONT, 0.9,
                    (0, 255, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "SPACE/y save   c continuous   q quit", (24, disp.shape[0] - 24),
                    FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            self.get_logger().info(f"done: {self.saved} saved this run, {total} total.")
            rclpy.shutdown()
        elif key in (ord('y'), ord(' ')):
            self._save(rect, small)       # manual save always honours the keypress
        elif key == ord('c'):
            self.continuous = not self.continuous
            self.get_logger().info(f"continuous {'ON' if self.continuous else 'OFF'}")

        if total >= self.target:
            self.get_logger().info(f"reached target {self.target}. Stopping.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CollectFrames()
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

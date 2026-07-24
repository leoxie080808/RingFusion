"""Live 2x2 viewer: rectified camera | ToF heatmap | Network A depth | Network B (A+B).

Self-contained -- runs its OWN backbone + residual inference (so it can show A-only
*and* A+B side by side, which the pipeline's single /depth can't). Needs only the
camera + ToF drivers publishing /image and /tof; the perception node is not required.

    ros2 run ringfusion_perception live_view --ros-args \
        -p backbone_engine:=$HOME/RingFusion/student_fp16.engine \
        -p residual_engine:=$HOME/RingFusion/residual_fp16.engine

Controls (window focused): q / ESC = quit.  Needs the Jetson's monitor.
Parameters: calib, backbone_engine, residual_engine (empty -> mock),
            rate (display Hz, default 10), panel_w (px per panel, default 480).
"""
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

from . import geometry as geo
from . import anchoring as anc
from . import gpu_ops
from .pipeline import splat_anchors
from .rectify import FisheyeRectifier
from .perception_node import load_calib

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = "RingFusion live  |  camera | ToF | Network A | Network B  (q=quit)"


def _colorize(a, mask, lo, hi, cmap=cv2.COLORMAP_TURBO):
    n = np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    c[~mask] = (0, 0, 0)
    return c


class LiveView(Node):
    def __init__(self):
        super().__init__('live_view')
        self.declare_parameter('calib', 'calibration.yaml')
        self.declare_parameter('backbone_engine', '')
        self.declare_parameter('residual_engine', '')
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('panel_w', 480)
        self.panel_w = int(self.get_parameter('panel_w').value)

        raw = load_calib(self._resolve_calib(self.get_parameter('calib').value))
        r = raw['rectify']
        self.rect = FisheyeRectifier(raw['K'], raw['dist'], raw['model'],
                                     size_in=(raw['img_w'], raw['img_h']),
                                     size_out=(r['width'], r['height']),
                                     balance=r['balance'], fov_scale=r['fov_scale'])
        # everything downstream runs on the rectified (pinhole) image
        self.calib = dict(raw)
        self.calib['K'] = self.rect.K_rect
        self.calib['model'] = 'pinhole'
        self.calib['img_w'], self.calib['img_h'] = self.rect.size

        self.backbone = self._load_backbone()
        self.residual = self._load_residual()

        self.img = None
        self.tof = None
        self.conf = None
        self.t_last = time.time()
        self.fps = 0.0

        try:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            self.get_logger().error(f"no display for the viewer ({e}); needs the monitor.")
            raise SystemExit(1)

        self.create_subscription(Image, 'image', self.on_image, 5)
        self.create_subscription(ToFFrame, 'tof', self.on_tof, 10)
        period = 1.0 / max(1.0, float(self.get_parameter('rate').value))
        self.timer = self.create_timer(period, self.on_tick)
        self.get_logger().info(
            f"live_view: backbone={self.backbone.name} residual={self.residual.name} "
            f"| waiting for /image + /tof")

    def _resolve_calib(self, path):
        import os
        if os.path.exists(path):
            return path
        try:
            from ament_index_python.packages import get_package_share_directory
            cand = os.path.join(get_package_share_directory('ringfusion_bringup'),
                                'config', 'calibration.yaml')
            if os.path.exists(cand):
                return cand
        except Exception:
            pass
        return path

    def _load_backbone(self):
        p = self.get_parameter('backbone_engine').value
        if p:
            from .backbone import TensorRTBackbone
            return TensorRTBackbone(p)
        from .backbone import MockBackbone
        return MockBackbone()

    def _load_residual(self):
        p = self.get_parameter('residual_engine').value
        if p:
            from .residual import ResidualRefiner
            return ResidualRefiner(p)
        from .residual import MockResidual
        return MockResidual()

    def on_image(self, m):
        if m.encoding != 'rgb8':
            return
        rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        self.img = self.rect.rectify(rgb[:, :, ::-1])          # BGR rectified

    def on_tof(self, m):
        self.tof = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols)
        self.conf = np.asarray(m.confidence, np.uint8).reshape(m.rows, m.cols)

    def _infer(self, rgb_bgr):
        """Backbone once -> A-only (D0) and A+B (D_AB). None if anchoring fails."""
        rgb = rgb_bgr[:, :, ::-1]                               # to RGB for the net
        h, w = rgb.shape[:2]
        K = self.calib['K']
        td, tv = self.tof, np.isfinite(self.tof)
        rows, cols = td.shape
        disp = self.backbone.infer(rgb)
        proj = geo.project_zone_to_pixel(td, tv, cols, rows, self.calib['fov_h'],
                                         self.calib['fov_v'], self.calib['T_cam_tof'],
                                         K, self.calib['dist'], model='pinhole')
        uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']
        finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
        u = np.round(np.where(finite, uv[:, 0], -1)).astype(int)
        v = np.round(np.where(finite, uv[:, 1], -1)).astype(int)
        inb = ok & finite & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
        if int(inb.sum()) < 4:
            return None
        disp_at = disp[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)][inb]
        inv_depth = 1.0 / z[inb]
        fit = anc.solve_robust(disp_at, inv_depth, np.ones_like(inv_depth), iters=1)
        if fit is None:
            return None
        a, b = fit
        gpu = gpu_ops.available()
        D0 = gpu_ops.to_metric_depth(disp, a, b) if gpu else anc.to_metric_depth(disp, a, b)
        ad, am = splat_anchors(u, v, z, inb, (h, w))
        D_AB, _ = self.residual.refine(rgb, D0, disp, ad, am, a, b)
        return D0, D_AB

    def _panel(self, img):
        return cv2.resize(img, (self.panel_w, self.panel_w * 3 // 4))

    def _tag(self, img, txt, sub=""):
        cv2.putText(img, txt, (10, 26), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if sub:
            cv2.putText(img, sub, (10, img.shape[0] - 12), FONT, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
        return img

    def _ctr(self, d):
        h, w = d.shape
        p = d[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8]
        p = p[np.isfinite(p) & (p > 0)]
        return float(np.median(p)) if p.size else float('nan')

    def on_tick(self):
        if self.img is None or self.tof is None:
            return
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(1e-3, now - self.t_last)
        self.t_last = now

        rgb_bgr = self.img
        res = self._infer(rgb_bgr)
        tv = np.isfinite(self.tof)
        tof_c = _colorize(np.nan_to_num(self.tof), tv,
                          float(np.nanmin(self.tof[tv])) if tv.any() else 0.0,
                          float(np.nanmax(self.tof[tv])) if tv.any() else 1.0)
        tof_c = cv2.resize(tof_c, (self.panel_w, self.panel_w * 3 // 4),
                           interpolation=cv2.INTER_NEAREST)
        tc = self.tof[self.tof.shape[0] // 2 - 2:self.tof.shape[0] // 2 + 3,
                      self.tof.shape[1] // 2 - 2:self.tof.shape[1] // 2 + 3]
        tc = tc[np.isfinite(tc)]
        tof_ctr = float(np.median(tc)) if tc.size else float('nan')

        cam = self._tag(self._panel(rgb_bgr).copy(), f"Camera  {self.fps:.1f} fps")
        tof_p = self._tag(tof_c, "ToF", f"ctr {tof_ctr:.2f} m")
        if res is None:
            blank = np.zeros((self.panel_w * 3 // 4, self.panel_w, 3), np.uint8)
            a_p = self._tag(blank.copy(), "Network A", "no anchors")
            b_p = self._tag(blank.copy(), "Network B", "no anchors")
        else:
            D0, D_AB = res
            dv = D0 > 0
            lo, hi = np.percentile(D0[dv], [2, 98]) if dv.any() else (0.0, 1.0)
            a_p = self._tag(self._panel(_colorize(D0, dv, lo, hi)), "Network A",
                            f"ctr {self._ctr(D0):.2f} m")
            b_p = self._tag(self._panel(_colorize(D_AB, D_AB > 0, lo, hi)), "Network B (A+B)",
                            f"ctr {self._ctr(D_AB):.2f} m")
        grid = np.vstack([np.hstack([cam, tof_p]), np.hstack([a_p, b_p])])
        cv2.imshow(WIN, grid)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LiveView()
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

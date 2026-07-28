#!/usr/bin/env python3
"""Is the depth compression the STUDENT's fault, or is mono depth just hard here?

The pipeline's accuracy ceiling is how well the backbone's disparity tracks true inverse
depth. Measured against ToF the student scores rho ~0.73. This runs the Depth Anything V2
teacher on the SAME frames and scores it the same way:

  teacher >> student  -> distillation gap; fix by retraining the student (more data)
  teacher ~= student  -> monocular depth is intrinsically hard in this scene; no amount of
                         distillation helps and the fix has to come from elsewhere
"""
import argparse

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

from ringfusion_perception import geometry as geo
from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.perception_node import load_calib


def score(disp, iv):
    """rho, plus the best achievable depth MAE under a 2-param affine fit."""
    rho = float(np.corrcoef(disp, iv)[0, 1])
    A = np.stack([disp, np.ones_like(disp)], 1)
    a, b = np.linalg.lstsq(A, iv, rcond=None)[0]
    d = 1.0 / np.maximum(a * disp + b, 1e-6)
    z = 1.0 / iv
    return rho, float(np.abs(d - z).mean()), float(np.polyfit(z, d, 1)[0])


class TvS(Node):
    def __init__(self, a):
        super().__init__('teacher_vs_student')
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
        from ringfusion_perception.backbone import TensorRTBackbone
        self.student = TensorRTBackbone(a.backbone_engine)

        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        mid = 'depth-anything/Depth-Anything-V2-Large-hf'
        print('loading teacher...', flush=True)
        self.proc = AutoImageProcessor.from_pretrained(mid)
        self.teacher = AutoModelForDepthEstimation.from_pretrained(mid).to('cuda').eval()
        print('teacher ready', flush=True)

        self.img = self.tof = None
        self.acc = []
        self.n = 0
        self.create_subscription(Image, '/image', self.on_img, 5)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 10)
        self.create_timer(1.0, self.tick)

    def on_img(self, m):
        if m.encoding == 'rgb8':
            rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.img = self.rect.rectify(rgb[:, :, ::-1])

    def on_tof(self, m):
        self.tof = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols)

    @torch.no_grad()
    def teacher_disp(self, rgb, hw):
        inp = self.proc(images=rgb, return_tensors='pt').to('cuda')
        out = self.teacher(**inp).predicted_depth[None]      # DA V2 emits DISPARITY
        out = torch.nn.functional.interpolate(out, size=hw, mode='bicubic',
                                              align_corners=False)[0, 0]
        return out.float().cpu().numpy()

    def tick(self):
        if self.img is None or self.tof is None:
            return
        self.n += 1
        if self.n < 3:
            return
        rgb = self.img[:, :, ::-1]
        h, w = rgb.shape[:2]
        td, tv = self.tof, np.isfinite(self.tof)
        rows, cols = td.shape
        ds = self.student.infer(rgb)
        dt = self.teacher_disp(np.ascontiguousarray(rgb), (h, w))
        proj = geo.project_zone_to_pixel(td, tv, cols, rows, self.calib['fov_h'],
                                         self.calib['fov_v'], self.calib['T_cam_tof'],
                                         self.calib['K'], self.calib['dist'], model='pinhole')
        uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']
        fin = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
        u = np.round(np.where(fin, uv[:, 0], -1)).astype(int)
        v = np.round(np.where(fin, uv[:, 1], -1)).astype(int)
        inb = ok & fin & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
        if inb.sum() < 50:
            return
        vv, uu = v[inb], u[inb]
        self.acc.append({'s': ds[vv, uu], 't': dt[vv, uu], 'iv': 1.0 / z[inb]})
        print(f"  frame {len(self.acc)}: {inb.sum()} anchors", flush=True)
        if len(self.acc) >= self.a.frames:
            raise SystemExit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--backbone-engine', required=True)
    p.add_argument('--frames', type=int, default=5)
    a = p.parse_args()
    rclpy.init()
    n = TvS(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if n.acc:
            s = np.concatenate([r['s'] for r in n.acc])
            t = np.concatenate([r['t'] for r in n.acc])
            iv = np.concatenate([r['iv'] for r in n.acc])
            print(f"\n{len(s)} anchors over {len(n.acc)} frames, "
                  f"true depth {1/iv.max():.2f}-{1/iv.min():.2f} m\n")
            print(f"{'':<10}{'rho vs 1/z':>13}{'best MAE':>12}{'range slope':>14}")
            print('-' * 49)
            for nm, dsp in (('student', s), ('teacher', t)):
                rho, mae, al = score(dsp, iv)
                print(f"{nm:<10}{rho:>13.4f}{mae:>11.3f} m{al:>14.3f}")
            print(f"\ncorr(student, teacher) at anchors = {np.corrcoef(s, t)[0,1]:.4f}")
            print("range slope: 1.0 = depth spans the true range; "
                  "<1 = output compressed toward one distance")
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

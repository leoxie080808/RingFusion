#!/usr/bin/env python3
"""Where do the deployed milliseconds go? Profiles the FULL node path, not just the maths.

The deployed node runs at 7.2 Hz (139 ms/frame) with blend+ROI on, while `time_pipeline.py`
measured the same stages at 81.4 ms offline. Something cost ~58 ms on the robot that cost
nothing on logged frames, and `pipeline.run` alone could not show it -- because the missing
work is the part that is NOT pipeline.run:

  * rectification -- a 2 MP cv2.remap on the CPU, per frame, that the offline harness skips
    entirely (it reads already-rectified PNGs)
  * publishing -- /depth and /depth_var are 1640x1232 float32 = 8.1 MB EACH, plus the cloud.
    That is ~17 MB per frame of serialisation and transport the offline harness never does.

So this subscribes to the same topics, rectifies the same way, calls the same pipeline with
per-stage timing, and publishes the same three messages -- a faithful replica of
perception_node, instrumented. Run it INSTEAD of the perception node (it publishes to
/prof_* by default so it can also run alongside without colliding).
"""
import argparse
import array
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from ringfusion_msgs.msg import ToFFrame

from ringfusion_perception import pipeline, roi
from ringfusion_perception.rectify import FisheyeRectifier
from ringfusion_perception.cloud_util import xyz_to_pointcloud2
from ringfusion_perception.perception_node import load_calib

WARMUP = 8            # TensorRT autotunes on the first inferences; those are not steady state


class Prof(Node):
    def __init__(self, a):
        super().__init__('profile_node')
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
        self.calib['dist'] = np.zeros(4)
        from ringfusion_perception.backbone import TensorRTBackbone
        from ringfusion_perception.residual import ResidualRefiner
        self.backbone = TensorRTBackbone(a.backbone_engine)
        self.residual = ResidualRefiner(a.residual_engine) if a.residual_engine else None
        self.tracker = roi.PlaneTracker(refit_every=a.plane_refit_every)
        self.depth_pub = self.create_publisher(Image, a.prefix + 'depth', 5)
        self.var_pub = self.create_publisher(Image, a.prefix + 'depth_var', 5)
        self.cloud_pub = self.create_publisher(PointCloud2, a.prefix + 'cloud', 5)
        self.img = None
        self.rows = []
        self.n = 0
        self.create_subscription(Image, '/image', self.on_image, 5)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 5)
        print(f'profiling blend={a.blend} roi={a.roi_enable} '
              f'(warmup {WARMUP} frames)...', flush=True)

    def on_image(self, m):
        buf = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        self.img_raw = buf if m.encoding == 'rgb8' else buf[:, :, ::-1]

    def _float_image(self, arr, stamp):
        """Byte-for-byte the node's own packing -- see perception_node._float_image.

        The `array.array('B', ...)` is NOT cosmetic. Assigning raw bytes to msg.data makes
        rclpy validate the buffer element-by-element; a first version of this profiler did
        that and measured 2081 ms per publish, which implied 0.46 Hz against the node's
        actual 7.2 Hz. A profiler that does not replicate the node exactly measures itself.
        """
        d = np.ascontiguousarray(arr, dtype=np.float32)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'cam_0'
        msg.height, msg.width = d.shape
        msg.encoding = '32FC1'
        msg.is_bigendian = 0
        msg.step = d.shape[1] * 4
        msg.data = array.array('B', d.tobytes())
        return msg

    def on_tof(self, m):
        if getattr(self, 'img_raw', None) is None:
            return
        t = {}
        t0 = time.perf_counter()
        # --- rectification: the 2 MP CPU remap the offline harness never runs
        rgb = self.rect.rectify(self.img_raw)
        t['1_rectify'] = (time.perf_counter() - t0) * 1e3

        d = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols)
        valid = np.isfinite(d) & (d > 0)
        conf = (np.asarray(m.confidence, np.uint8).reshape(m.rows, m.cols)
                if len(m.confidence) == m.rows * m.cols else None)

        t1 = time.perf_counter()
        res = pipeline.run(rgb, d, valid, self.calib, self.backbone, self.residual,
                           confidence=conf, min_confidence=-1,
                           blend=self.a.blend, roi_enable=self.a.roi_enable,
                           plane_tracker=self.tracker, timings=t)
        t['PIPELINE_TOTAL'] = (time.perf_counter() - t1) * 1e3
        if not res['ok']:
            return

        # --- publishing: ~17 MB/frame the offline harness never serialises
        t2 = time.perf_counter()
        self.depth_pub.publish(self._float_image(res['metric'], m.header.stamp))
        if res.get('var') is not None:
            self.var_pub.publish(self._float_image(res['var'], m.header.stamp))
        self.cloud_pub.publish(xyz_to_pointcloud2(res['cloud'], 'cam_0', m.header.stamp))
        t['9_publish'] = (time.perf_counter() - t2) * 1e3
        t['FRAME_TOTAL'] = (time.perf_counter() - t0) * 1e3

        self.n += 1
        if self.n > WARMUP:
            self.rows.append(t)
            if len(self.rows) % 20 == 0:
                print(f'  {len(self.rows)} frames', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--backbone-engine', required=True)
    p.add_argument('--residual-engine', default='')
    p.add_argument('--blend', type=lambda s: s.lower() != 'false', default=True)
    p.add_argument('--roi-enable', type=lambda s: s.lower() != 'false', default=True)
    p.add_argument('--plane-refit-every', type=int, default=1)
    p.add_argument('--frames', type=int, default=80)
    p.add_argument('--prefix', default='/prof_')
    p.add_argument('--out', default='')
    a = p.parse_args()

    rclpy.init()
    n = Prof(a)
    t_end = time.time() + 180
    while rclpy.ok() and len(n.rows) < a.frames and time.time() < t_end:
        rclpy.spin_once(n, timeout_sec=0.1)

    if not n.rows:
        print('no frames profiled -- are /image and /tof publishing?')
        rclpy.shutdown()
        return

    keys = [k for k in n.rows[0] if k not in ('FRAME_TOTAL', 'PIPELINE_TOTAL')]
    keys.sort()
    tot = float(np.median([r['FRAME_TOTAL'] for r in n.rows]))
    print(f'\n=== per-stage cost, {len(n.rows)} frames, '
          f'blend={a.blend} roi={a.roi_enable} ===')
    print(f'{"stage":<20}{"median ms":>11}{"p90":>9}{"% frame":>10}')
    print('-' * 50)
    report = {}
    for k in keys:
        v = [r.get(k, 0.0) for r in n.rows]
        med = float(np.median(v))
        report[k] = {'median_ms': med, 'p90_ms': float(np.percentile(v, 90))}
        print(f'{k:<20}{med:>11.2f}{np.percentile(v, 90):>9.2f}{100*med/tot:>9.1f}%')
    pipe = float(np.median([r['PIPELINE_TOTAL'] for r in n.rows]))
    print('-' * 50)
    print(f'{"pipeline.run total":<20}{pipe:>11.2f}{"":>9}{100*pipe/tot:>9.1f}%')
    print(f'{"FRAME TOTAL":<20}{tot:>11.2f}{"":>9}{100.0:>9.1f}%')
    print(f'\nimplied rate: {1000/tot:.2f} Hz')
    print('offline pipeline.run reference (time_pipeline.py, 1640x1232, post-GPU-rewrites):')
    print('  70.7 ms blend+ROI on   |   50.9 ms both off')
    print(f'non-pipeline overhead here: {tot - pipe:.1f} ms '
          f'(rectify + publish, neither present offline)')

    if a.out:
        json.dump({'blend': a.blend, 'roi_enable': a.roi_enable, 'n': len(n.rows),
                   'frame_total_ms': tot, 'pipeline_total_ms': pipe, 'stages': report},
                  open(a.out, 'w'), indent=1)
        print(f'wrote {a.out}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()

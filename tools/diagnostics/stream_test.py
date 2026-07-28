#!/usr/bin/env python3
"""Record the live /depth stream for N seconds and measure TEMPORAL stability.

Single-frame metrics say nothing about flicker. This watches consecutive published frames
and reports how much the depth map jumps between them -- the thing a downstream consumer
(SLAM, obstacle avoidance) actually feels.
"""
import argparse
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class Stream(Node):
    def __init__(self, secs, out):
        super().__init__('stream_test')
        self.secs = secs
        self.out = out
        self.t0 = None
        self.frames = []
        self.times = []
        self.create_subscription(Image, '/depth', self.on_depth, 30)

    def on_depth(self, m):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        if now - self.t0 > self.secs:
            raise SystemExit(0)
        d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
        self.frames.append(d[::4, ::4].copy())      # 4x decimate: plenty for stability stats
        self.times.append(now)


def report(fr, ts, out):
    n = len(fr)
    dur = ts[-1] - ts[0] if n > 1 else 0.0
    rate = (n - 1) / dur if dur > 0 else float('nan')
    med = np.array([np.median(f[f > 0]) for f in fr])
    mx = np.array([f.max() for f in fr])
    p99 = np.array([np.percentile(f[f > 0], 99) for f in fr])
    # frame-to-frame change on pixels valid in both
    jit, rel = [], []
    for a, b in zip(fr[:-1], fr[1:]):
        m = (a > 0) & (b > 0)
        if m.sum():
            jit.append(float(np.mean(np.abs(b[m] - a[m]))))
            rel.append(float(np.mean(np.abs(b[m] - a[m]) / np.maximum(a[m], 1e-3))))
    jit, rel = np.array(jit), np.array(rel)
    res = {
        'frames': n, 'duration_s': round(dur, 2), 'rate_hz': round(rate, 2),
        'median_depth_mean_m': round(float(med.mean()), 4),
        'median_depth_std_m': round(float(med.std()), 4),
        'median_depth_range_m': [round(float(med.min()), 3), round(float(med.max()), 3)],
        'max_depth_mean_m': round(float(mx.mean()), 3),
        'max_depth_worst_m': round(float(mx.max()), 3),
        'p99_depth_mean_m': round(float(p99.mean()), 3),
        'p99_depth_worst_m': round(float(p99.max()), 3),
        'frame_to_frame_abs_m': round(float(jit.mean()), 4),
        'frame_to_frame_abs_worst_m': round(float(jit.max()), 4),
        'frame_to_frame_rel_pct': round(float(rel.mean() * 100), 2),
        'frames_hitting_clamp': int((mx >= 19.9).sum()),
    }
    json.dump(res, open(out, 'w'), indent=2)
    for k, v in res.items():
        print(f"  {k:<28} {v}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--secs', type=float, default=5.0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rclpy.init()
    n = Stream(a.secs, a.out)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if len(n.frames) > 1:
            report(n.frames, n.times, a.out)
        else:
            print('not enough frames')
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

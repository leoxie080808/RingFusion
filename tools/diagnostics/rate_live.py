#!/usr/bin/env python3
"""Measure the DEPLOYED publish rate and end-to-end latency of the running perception node.

This settles the one number in the READMEs that has never been observed: `time_pipeline.py`
times `pipeline.run()` with no ROS and no rectification, so the deployed rate has only ever
been extrapolated (81.4 ms + ~20 ms => ~9.9 Hz). This measures it -- and the extrapolation
was wrong: the robot returned 7.2 Hz, because the overhead is not a constant you can add on.
See "the deployed rate, measured" in ros2_ws/README.md.

Two numbers, and they answer different questions:

  RATE     how many depth maps per second come out. What throughput-bound consumers feel.
  LATENCY  how OLD each map is when it arrives, measured from the ToF stamp it carries.
           A robot avoiding an obstacle cares about this one, and it is not 1/rate: if
           input arrives faster than we process, maps queue up and latency grows without
           the rate changing at all.

Why /tof is measured too: /depth can never exceed its input rate, so a low /depth rate only
proves the pipeline is slow IF /tof was supplying frames faster than that. Without that
check, an input-starved run and a compute-bound run look identical. The driver publishes a
complete 32x32 map on every subframe (~16 Hz, not per-pair), so there should be ample supply.

Run it while the robot sits still -- this measures compute, not scene content. Nothing else
should be on the GPU: a background job silently doubled a DEPTHOR timing earlier in this
project, and this measurement is just as vulnerable.
"""
import argparse
import json
import statistics
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from ringfusion_msgs.msg import ToFFrame

# TensorRT engines are slow for the first few inferences (kernel autotuning, allocation), and
# the ROS graph takes a moment to settle. Those frames are not representative of steady state.
WARMUP_S = 3.0


def stamp_s(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class RateLive(Node):
    def __init__(self, a):
        super().__init__('rate_live')
        self.a = a
        self.t_start = time.time()
        self.rec = {k: [] for k in ('image', 'tof', 'depth', 'cloud')}
        self.lat = []                      # /depth age at arrival, seconds
        self.create_subscription(Image, a.image_topic, lambda m: self._hit('image'), 30)
        self.create_subscription(ToFFrame, a.tof_topic, lambda m: self._hit('tof'), 30)
        self.create_subscription(Image, a.depth_topic, self._on_depth, 30)
        self.create_subscription(PointCloud2, a.cloud_topic, lambda m: self._hit('cloud'), 30)

    def _warm(self):
        return (time.time() - self.t_start) < WARMUP_S

    def _hit(self, key):
        if not self._warm():
            self.rec[key].append(time.time())

    def _on_depth(self, m):
        if self._warm():
            return
        now = time.time()
        self.rec['depth'].append(now)
        # The node stamps /depth with the ToF message's own header, so this is the full age
        # of the data: driver -> ROS -> rectify -> pipeline -> publish. Same machine, so no
        # clock-sync caveat.
        age = now - stamp_s(m.header)
        if -1.0 < age < 10.0:              # ignore nonsense from an unset/zero stamp
            self.lat.append(age)


def summarise(ts):
    """Rate from the span (robust to a dropped sample), plus the inter-arrival spread."""
    if len(ts) < 3:
        return None
    span = ts[-1] - ts[0]
    d = np.diff(ts) * 1e3
    return {'n': len(ts), 'hz': (len(ts) - 1) / span if span > 0 else float('nan'),
            'median_ms': float(np.median(d)), 'p90_ms': float(np.percentile(d, 90)),
            'max_ms': float(d.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--secs', type=float, default=30.0,
                   help='measurement window AFTER a 3 s warmup')
    p.add_argument('--image-topic', default='/image')
    p.add_argument('--tof-topic', default='/tof')
    p.add_argument('--depth-topic', default='/depth')
    p.add_argument('--cloud-topic', default='/cloud')
    p.add_argument('--label', default='', help='e.g. "blend+roi on" -- goes into the JSON')
    p.add_argument('--expect-hz', type=float, default=9.9,
                   help='the extrapolated prediction this run is testing')
    p.add_argument('--out', default='')
    a = p.parse_args()

    rclpy.init()
    n = RateLive(a)
    print(f'warming up {WARMUP_S:.0f} s, then measuring {a.secs:.0f} s...', flush=True)
    end = time.time() + WARMUP_S + a.secs
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)

    res = {k: summarise(v) for k, v in n.rec.items()}
    print(f'\n=== deployed rate{" -- " + a.label if a.label else ""} ===')
    print(f'{"topic":<10}{"n":>6}{"Hz":>9}{"median":>10}{"p90":>10}{"max":>10}')
    print('-' * 55)
    for k in ('image', 'tof', 'depth', 'cloud'):
        r = res[k]
        if r is None:
            print(f'{k:<10}{"-- no messages --":>45}')
            continue
        print(f'{k:<10}{r["n"]:>6}{r["hz"]:>9.2f}{r["median_ms"]:>9.1f}m'
              f'{r["p90_ms"]:>9.1f}m{r["max_ms"]:>9.1f}m')

    d, t = res['depth'], res['tof']
    if d is None:
        print('\n!! no /depth at all -- is the perception node running, and are the engines '
              'loaded? Check the launch output for "anchoring failed".')
        rclpy.shutdown()
        return

    lat = None
    if n.lat:
        lat = {'median_ms': float(np.median(n.lat) * 1e3),
               'p90_ms': float(np.percentile(n.lat, 90) * 1e3),
               'max_ms': float(np.max(n.lat) * 1e3)}
        print(f'\n/depth age on arrival (how stale each map is):')
        print(f'  median {lat["median_ms"]:.1f} ms   p90 {lat["p90_ms"]:.1f} ms   '
              f'max {lat["max_ms"]:.1f} ms')
        if lat['median_ms'] > 1.8 * d['median_ms']:
            print(f'  -> latency is much larger than the {d["median_ms"]:.0f} ms publish '
                  f'period, so frames are QUEUING: input arrives faster than we process it.\n'
                  f'     Throughput looks fine while the data a consumer sees is stale.')

    print('\ninterpretation:')
    if t is not None:
        if t['hz'] < d['hz'] * 1.15:
            print(f'  !! /tof is only {t["hz"]:.1f} Hz against /depth {d["hz"]:.1f} Hz -- '
                  f'this run is INPUT-LIMITED.\n     It measures the ToF, not the pipeline. '
                  f'Re-run with the ToF streaming properly before drawing any conclusion.')
        else:
            print(f'  /tof supplies {t["hz"]:.1f} Hz against /depth {d["hz"]:.1f} Hz, so '
                  f'the pipeline is the constraint --\n     this measures COMPUTE, which is '
                  f'what we wanted.')
    print(f'  measured {d["hz"]:.2f} Hz vs {a.expect_hz:.1f} Hz extrapolated '
          f'({100 * (d["hz"] - a.expect_hz) / a.expect_hz:+.0f}%)')
    print(f'  offline pipeline.run() reference: 14.1 Hz (70.7 ms) at 1640x1232, '
          f'blend+ROI on, no ROS')
    print(f'  ToF complete-map rate to stay above: 8.3 Hz -- '
          f'{"PASS" if d["hz"] >= 8.3 else "FAIL"}')

    if a.out:
        json.dump({'label': a.label, 'secs': a.secs, 'expect_hz': a.expect_hz,
                   'topics': res, 'depth_latency': lat},
                  open(a.out, 'w'), indent=1)
        print(f'\nwrote {a.out}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()

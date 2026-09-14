#!/usr/bin/env python3
"""Where a published depth map's age actually goes. Passive: subscribes only.

The reviewer asked for the relationship among core latency, publication rate and data
age to be explained. We can currently state two of the three and not the bridge between
them: profile_node measures the pipeline, rate_live measures total age, and nothing
accounts for the gap. On the last verified capture that gap was over 300 ms -- larger
than everything profile_node measures -- so it cannot be left as rounding.

WHAT IT MEASURES, and how, without modifying the node
-----------------------------------------------------
perception_node is driven by `on_tof`, and stamps /depth with the ToF message's own
header (perception_node.py: `self.depth_pub.publish(self._float_image(..., msg.header.stamp))`).
So a depth message and its originating ToF message share a header stamp, and matching on
that stamp splits the age in two:

    stamp ......... tof arrives here ......... depth arrives here
      |<- sensor + transport ->|<- queue + compute + publish ->|
      |<--------------------- total age --------------------->|

  sensor_to_arrival  = t_tof_arrival  - stamp   sensor integration, USB, driver assembly
  node_latency       = t_depth_arrival - t_tof_arrival   queue wait + pipeline + publish
  total_age          = t_depth_arrival - stamp   the number rate_live reports

Pass --pipeline-ms (profile_node's `pipeline_total_ms` from the SAME build) and the
residual is attributed too:

  queue_and_overhead = node_latency - pipeline_ms

THE IMAGE IS OLDER THAN THE STAMP SAYS
--------------------------------------
`on_image` caches the newest raw frame and `on_tof` consumes whatever is cached. The
published stamp is the ToF stamp, so it describes the dToF half of the fusion only -- the
camera half is as old as the cache. At a 30 Hz camera against a 10 Hz pipeline that is
tens of milliseconds, and if a camera frame is dropped it is more. `image_staleness`
reconstructs it as (t_tof_arrival - t_newest_image_arrival_before_it). A consumer
treating the stamp as the age of the whole observation is wrong by this amount, which is
worth a sentence in V-A whatever its size.

TWO HONEST LIMITS, both of which make node_latency an UPPER bound
-----------------------------------------------------------------
1. Our arrival times are when THIS process is woken, so each includes one DDS hop plus
   Python scheduling that the node itself does not pay. Both arrivals carry it, so it
   partly cancels in node_latency but not exactly.
2. image_staleness is reconstructed from our subscriber's ordering, not the node's. Under
   the same executor these agree closely, but it is an approximation and is labelled one.

Neither is fixable from outside the node; reporting them beats implying precision we do
not have. If the gap turns out to matter, the node needs to publish its own timing.

    python3 tools/diagnostics/age_budget.py --secs 120 --pipeline-ms 120.5 \\
        --label "r31 deployed" --out docs/demo/benchmarks/age_budget_r31.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import envinfo                                                      # noqa: E402

import rclpy                                                        # noqa: E402
from rclpy.node import Node                                         # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # noqa: E402
from sensor_msgs.msg import Image                                   # noqa: E402
from ringfusion_msgs.msg import ToFFrame                            # noqa: E402


def stamp_key(h):
    """Exact integer key -- float seconds would collide or miss on nanosecond stamps."""
    return (int(h.stamp.sec), int(h.stamp.nanosec))


def stamp_s(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class AgeBudget(Node):
    def __init__(self, args):
        super().__init__('age_budget')
        # Best-effort, depth 50: this must not add backpressure to the node under test,
        # and a dropped sample here costs one row of statistics rather than perturbing
        # the thing being measured.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=50)
        self.tof_arrival = {}        # stamp key -> wall time the ToF frame reached us
        self.image_arrivals = []     # wall times of camera frames, in order
        self.rows = []
        self.n_unmatched = 0
        self.create_subscription(ToFFrame, args.tof_topic, self.on_tof, qos)
        self.create_subscription(Image, args.image_topic, self.on_image, qos)
        self.create_subscription(Image, args.depth_topic, self.on_depth, qos)

    def on_image(self, msg):
        self.image_arrivals.append(time.time())
        if len(self.image_arrivals) > 4000:
            del self.image_arrivals[:2000]

    def on_tof(self, msg):
        self.tof_arrival[stamp_key(msg.header)] = (time.time(), stamp_s(msg.header))
        if len(self.tof_arrival) > 4000:
            for k in list(self.tof_arrival)[:2000]:
                del self.tof_arrival[k]

    def on_depth(self, msg):
        now = time.time()
        k = stamp_key(msg.header)
        rec = self.tof_arrival.get(k)
        if rec is None:
            # The matching ToF frame was dropped before reaching us, or arrived before we
            # subscribed. Counted, not silently discarded -- a large count would mean the
            # split below is drawn from a biased subset.
            self.n_unmatched += 1
            return
        t_tof, t_stamp = rec
        # Newest camera frame that had arrived before the ToF frame did: what on_tof
        # would have found in its cache.
        prior = [t for t in self.image_arrivals if t <= t_tof]
        img_stale = (t_tof - prior[-1]) if prior else float('nan')
        self.rows.append({
            'stamp': t_stamp,
            'sensor_to_arrival_ms': (t_tof - t_stamp) * 1e3,
            'node_latency_ms': (now - t_tof) * 1e3,
            'total_age_ms': (now - t_stamp) * 1e3,
            'image_staleness_ms': img_stale * 1e3,
            't_depth': now,
        })


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float('nan')


def summarise(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not v.size:
        return {'n': 0}
    return {'n': int(v.size), 'median_ms': round(float(np.median(v)), 2),
            'mean_ms': round(float(v.mean()), 2), 'p05_ms': round(pct(v, 5), 2),
            'p95_ms': round(pct(v, 95), 2), 'min_ms': round(float(v.min()), 2),
            'max_ms': round(float(v.max()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secs', type=float, default=120.0)
    ap.add_argument('--image-topic', default='/image')
    ap.add_argument('--tof-topic', default='/tof')
    ap.add_argument('--depth-topic', default='/depth')
    ap.add_argument('--label', default='')
    # profile_node's pipeline_total_ms from the SAME build, to attribute the residual.
    ap.add_argument('--pipeline-ms', type=float, default=0.0,
                    help="profile_node pipeline_total_ms, same build, to split the residual")
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    env = envinfo.capture(note='age_budget')
    envinfo.warn_if_unlocked(env)

    rclpy.init()
    node = AgeBudget(a)
    print(f'listening {a.secs:.0f}s on {a.tof_topic}, {a.image_topic}, {a.depth_topic} ...')
    t_end = time.time() + a.secs
    while time.time() < t_end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)
    rows = list(node.rows)
    unmatched = node.n_unmatched
    node.destroy_node()
    rclpy.shutdown()

    if not rows:
        sys.exit('no matched depth/tof pairs -- is the node running and publishing?')

    budget = {k: summarise([r[k] for r in rows]) for k in
              ('sensor_to_arrival_ms', 'node_latency_ms', 'total_age_ms', 'image_staleness_ms')}

    t = sorted(r['t_depth'] for r in rows)
    periods = np.diff(t) * 1e3
    rate = {'n_depth': len(t), 'duration_s': round(t[-1] - t[0], 2),
            'hz': round((len(t) - 1) / (t[-1] - t[0]), 3) if t[-1] > t[0] else float('nan'),
            'period_median_ms': round(float(np.median(periods)), 2) if periods.size else None,
            'period_p95_ms': round(pct(periods, 95), 2) if periods.size else None}

    report = {
        'label': a.label or 'age budget, deployed node',
        'env': env,
        'matched_pairs': len(rows),
        'unmatched_depth_msgs': unmatched,
        'budget': budget,
        'rate': rate,
        'method': {
            'split': 'stamp -> tof arrival -> depth arrival, matched on the shared header stamp',
            'node_latency_is_upper_bound': ('includes one DDS hop and Python scheduling this '
                                            'process pays but the node does not'),
            'image_staleness': ('reconstructed from our own subscriber ordering; approximates '
                                'what on_tof found cached, it is not read from the node'),
        },
    }

    if a.pipeline_ms > 0:
        resid = budget['node_latency_ms'].get('median_ms', float('nan')) - a.pipeline_ms
        report['attribution'] = {
            'pipeline_total_ms': a.pipeline_ms,
            'queue_and_overhead_ms': round(resid, 2),
            'note': ('node_latency minus the pipeline. Covers queue wait, publish, '
                     'serialisation and the DDS hop. Negative means the pipeline figure '
                     'came from a different build or a different resolution.'),
        }

    print(f'\n  matched {len(rows)} pairs, {unmatched} unmatched\n')
    print(f'  {"component":<26}{"median":>10}{"p05":>10}{"p95":>10}')
    for k, v in budget.items():
        if v.get('n'):
            print(f'  {k[:-3]:<26}{v["median_ms"]:>10.1f}{v["p05_ms"]:>10.1f}{v["p95_ms"]:>10.1f}')
    print(f'\n  publish {rate["hz"]:.2f} Hz, period median {rate["period_median_ms"]:.1f} ms')
    if 'attribution' in report:
        print(f'  pipeline {a.pipeline_ms:.1f} ms  ->  queue+overhead '
              f'{report["attribution"]["queue_and_overhead_ms"]:.1f} ms')
    tot = budget['total_age_ms'].get('median_ms', float('nan'))
    per = rate['period_median_ms'] or float('nan')
    print(f'\n  data age {tot:.0f} ms against a {per:.0f} ms publish period '
          f'-- a map is {tot/per:.1f} publish periods old when it lands' if per else '')

    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()

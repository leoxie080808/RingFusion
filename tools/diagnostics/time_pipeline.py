#!/usr/bin/env python3
"""Time pipeline.run offline, at any resolution, with the new stages on and off.

Two things this answers without touching the robot:

1. A FAIR speed comparison against DEPTHOR. Measured on this Orin, network-only at
   480x640: DEPTHOR-Small 79.4 ms (12.6 Hz), DEPTHOR-Large 183.8 ms (5.4 Hz). Our 13.7 Hz
   figure is at 1640x1232 -- 6.5x the pixels -- so quoting 13.7 against their 7.84 it/s
   compares different workloads. Running our pipeline at THEIR resolution makes the claim
   defensible.

   (Their published-throughput figure of 7.84 it/s from evaluate.py includes h5
   dataloading. Use the network-only numbers above for model-vs-model.)

2. The cost of Stage 7c (blend) and 4b/7d (ROI plane + pixel mask). Both were added after
   the last live profile, and the blend runs a distanceTransform over the full frame while
   the ROI runs a RANSAC plane fit per frame. Perception was already the bottleneck at
   ~73 ms/frame against a 60.3 ms ToF cadence, so this has to be checked before deploying.

Timing note: a background GPU job silently inflated an earlier measurement of DEPTHOR by
2x (166 ms vs 79 ms clean). Check nothing else is on the GPU before trusting output here.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'training'))

from anchoring_bridge import calib_from_yaml                 # noqa: E402
from ringfusion_perception import pipeline, roi              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-dir', required=True)
    ap.add_argument('--tof-dir', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--backbone-engine', required=True)
    ap.add_argument('--residual-engine', default='')
    ap.add_argument('--sizes', nargs='+', default=['1232x1640', '480x640'],
                    help='HxW input sizes to time')
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import cv2
    from ringfusion_perception.backbone import TensorRTBackbone
    backbone = TensorRTBackbone(a.backbone_engine)
    residual = None
    if a.residual_engine:
        from ringfusion_perception.residual import ResidualRefiner
        residual = ResidualRefiner(a.residual_engine)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a.tof_dir)
                   if f.endswith('.npz'))[:a.iters + 5]
    raw = [(cv2.imread(os.path.join(a.rgb_dir, s + '.png')),
            np.load(os.path.join(a.tof_dir, s + '.npz'))['dist_m'].astype(np.float32))
           for s in stems]
    raw = [(im, d) for im, d in raw if im is not None]
    H0, W0 = raw[0][0].shape[:2]

    report = {}
    print(f'source {W0}x{H0}, {len(raw)} frames, iters={a.iters}\n')
    hdr = f"{'size':>12}{'blend':>7}{'roi':>5}{'median ms':>11}{'p90 ms':>9}{'Hz':>7}"
    print(hdr); print('-' * len(hdr))

    for size in a.sizes:
        h, w = (int(x) for x in size.lower().split('x'))
        # Scale K with the image so the geometry stays self-consistent at every size.
        calib = calib_from_yaml(a.calib, train_size=(h, w))
        frames = [(np.ascontiguousarray(cv2.resize(im, (w, h))[:, :, ::-1]), d)
                  for im, d in raw]
        for blend, roi_en in ((False, False), (True, False), (True, True)):
            tracker = roi.PlaneTracker() if roi_en else None
            valid = np.isfinite(frames[0][1])
            for _ in range(3):                       # warmup
                pipeline.run(frames[0][0], frames[0][1], np.isfinite(frames[0][1]),
                             calib, backbone, residual, blend=blend,
                             roi_enable=roi_en, plane_tracker=tracker)
            ts = []
            for i in range(a.iters):
                rgb, d = frames[i % len(frames)]
                valid = np.isfinite(d)
                t = time.perf_counter()
                r = pipeline.run(rgb, d, valid, calib, backbone, residual, blend=blend,
                                 roi_enable=roi_en, plane_tracker=tracker)
                ts.append((time.perf_counter() - t) * 1e3)
            med = statistics.median(ts)
            print(f'{w}x{h:<7}{str(blend):>7}{str(roi_en):>5}{med:>11.1f}'
                  f'{np.percentile(ts,90):>9.1f}{1000/med:>7.1f}')
            report[f'{w}x{h}_blend{int(blend)}_roi{int(roi_en)}'] = {
                'median_ms': med, 'p90_ms': float(np.percentile(ts, 90)),
                'hz': 1000 / med}

    print('\nreference, measured on this Orin at 480x640, network-only:')
    print('  DEPTHOR-Small 79.4 ms (12.6 Hz)   DEPTHOR-Large 183.8 ms (5.4 Hz)')
    if a.out:
        import json
        json.dump(report, open(a.out, 'w'), indent=1)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()

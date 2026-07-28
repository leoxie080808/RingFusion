#!/usr/bin/env python3
"""Retrofit an already-recorded 4-panel clip into a colour+depth sequence for the 3D viewer.

Normally you would take raw float32 metres straight off /depth. This exists only to reuse a
drive that was already recorded, by inverting the TURBO colourisation in the clip's depth
panel. That is lossless enough to be worth it: the panel used a FIXED 0.15-4.0 m range, so
the mapping is well defined, and a nearest-LUT inverse round-trips at ~8 mm mean error
through JPEG q88 -- far below the pipeline's own ~200 mm.

Two real limits inherited from the recording:
  * depth was clipped to 4.0 m, so anything further saturates and is dropped here
  * resolution is the panel's 420x315, not the full 1640x1232

Panel layout (from record_clip.py): [camera | tof | fused depth | top-down], each 420x315,
with a 26 px label bar drawn on top and caption text along the bottom -- both masked out.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

PW, PH = 420, 315
BAR_TOP, BAR_BOT = 26, 20          # label bar height, caption strip height
INVALID_BGR = np.array([18, 18, 18], np.int32)


def turbo_lut():
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                             cv2.COLORMAP_TURBO).reshape(-1, 3).astype(np.int32)


def invert_turbo(panel, lut, lo, hi):
    """BGR panel -> metric depth, 0 where invalid. int32 throughout: 255**2 overflows int16."""
    f = panel.reshape(-1, 3).astype(np.int32)
    d2 = ((f[:, None, :] - lut[None, :, :]) ** 2).sum(2)
    idx = np.argmin(d2, axis=1)
    best = d2[np.arange(d2.shape[0]), idx]
    z = lo + idx / 255.0 * (hi - lo)
    # reject pixels that are not really on the colour ramp (label text, the grey fill)
    bad = (best > 400) | (np.abs(f - INVALID_BGR).sum(1) < 30)
    z[bad] = 0.0
    return z.reshape(panel.shape[:2]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--calib', required=True)
    p.add_argument('--depth-panel', type=int, default=2, help='0-based panel index')
    p.add_argument('--cam-panel', type=int, default=0)
    p.add_argument('--lo', type=float, default=0.15, help='d_lo used when the clip was drawn')
    p.add_argument('--hi', type=float, default=4.0)
    p.add_argument('--stride', type=int, default=2, help='keep every Nth frame')
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    lut = turbo_lut()
    files = sorted(glob.glob(os.path.join(a.frames_dir, 'f*.jpg')))[::a.stride]
    if not files:
        raise SystemExit(f'no frames in {a.frames_dir}')

    # rectified intrinsics, rescaled to the panel resolution
    import sys
    sys.path.insert(0, '/home/leroi-ultio/RingFusion/ros2_ws/build/ringfusion_perception')
    from ringfusion_perception.rectify import FisheyeRectifier
    from ringfusion_perception.perception_node import load_calib
    raw = load_calib(a.calib)
    r = raw['rectify']
    rect = FisheyeRectifier(raw['K'], raw['dist'], raw['model'],
                            size_in=(raw['img_w'], raw['img_h']),
                            size_out=(r['width'], r['height']),
                            balance=r['balance'], fov_scale=r['fov_scale'])
    K = np.asarray(rect.K_rect, np.float64).ravel()
    if K.size == 9:
        m = K.reshape(3, 3)
        K = np.array([m[0, 0], m[1, 1], m[0, 2], m[1, 2]])
    sx, sy = PW / float(r['width']), PH / float(r['height'])

    n = 0
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        # panels sit side by side; the clip also has a progress bar row at the bottom
        dp = im[:PH, a.depth_panel * PW:(a.depth_panel + 1) * PW]
        cp = im[:PH, a.cam_panel * PW:(a.cam_panel + 1) * PW]
        if dp.shape[:2] != (PH, PW):
            continue
        z = invert_turbo(dp, lut, a.lo, a.hi)
        z[:BAR_TOP, :] = 0.0                  # label bar
        z[PH - BAR_BOT:, :] = 0.0             # caption strip
        q = np.zeros((PH, PW), np.uint8)
        ok = z > 0
        q[ok] = np.clip(1 + (z[ok] - a.lo) / (a.hi - a.lo) * 254.0, 1, 255).astype(np.uint8)
        out = np.hstack([cp, cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)])
        cv2.imwrite(f'{a.out_dir}/f{n:05d}.png', out)
        n += 1
        if n % 50 == 0:
            print(f'  {n} frames', flush=True)

    meta = {'w': PW, 'h': PH, 'frames': n, 'fps': 15.0 / a.stride,
            'zmin': a.lo, 'zmax': a.hi,
            'fx': K[0] * sx, 'fy': K[1] * sy, 'cx': K[2] * sx, 'cy': K[3] * sy,
            'source': 'retrofit from rendered clip (TURBO inverse, ~8 mm)'}
    json.dump(meta, open(os.path.join(a.out_dir, 'meta.json'), 'w'), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

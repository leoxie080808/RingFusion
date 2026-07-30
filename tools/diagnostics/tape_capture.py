#!/usr/bin/env python3
"""Capture independent ground truth: freeze a frame, click a marker, type its measured range.

This is the ONLY workstream that breaks the closed loop. Every other number in this repo is
scored against the ToF the pipeline anchors to, so none of them can detect a ToF bias and
none say anything about the 92.5% of the frame the ToF never covers.

Run it on the robot with the pipeline up. For each marked point:
  SPACE   freeze the live view (and latch depth/var/tof for that instant)
  click   on the marker centre -- a zoomed inset opens so the click can be placed precisely
  type    the measured range in metres, ENTER
  u       undo the last point        s  save        q  quit

The measurement you type is SLANT RANGE from the camera's optical centre to the marker. The
pipeline predicts optical-axis depth z. tape_eval.py does the z = r*cos(theta) conversion --
do NOT pre-convert here, and do not measure from the housing face (see --origin-offset).

Everything is written raw: float32 depth, float32 variance, the rgb, and the ToF map, so the
session can be re-scored later without re-measuring anything.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

FONT = cv2.FONT_HERSHEY_SIMPLEX
ZOOM = 8                      # inset magnification, so a 1-2 px click error is visible
ZOOM_HALF = 40                # half-size of the zoomed source region, in source pixels


class TapeCapture(Node):
    def __init__(self, a):
        super().__init__('tape_capture')
        self.a = a
        self.rgb = self.depth = self.var = self.tof = None
        self.create_subscription(Image, a.image_topic, self.on_img, 5)
        self.create_subscription(Image, a.depth_topic, self.on_depth, 5)
        self.create_subscription(Image, a.var_topic, self.on_var, 5)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 5)
        os.makedirs(a.dir, exist_ok=True)
        self.points = []
        self.frozen = None
        self.click = None
        self.n_saved = 0
        self._load_existing()

    # --- subscriptions ---
    def on_img(self, m):
        buf = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        self.rgb = buf[:, :, ::-1].copy() if m.encoding == 'rgb8' else buf.copy()

    def on_depth(self, m):
        self.depth = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()

    def on_var(self, m):
        self.var = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()

    def on_tof(self, m):
        self.tof = (np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols).copy()
                    if len(m.dist_m) == m.rows * m.cols else None)

    # --- persistence ---
    def _gt_path(self):
        return os.path.join(self.a.dir, 'tape_gt.json')

    def _load_existing(self):
        """Appending to an existing session is the normal case -- ~20 points is more than
        one sitting, and losing half of them to a restart would mean re-measuring."""
        p = self._gt_path()
        if os.path.exists(p):
            d = json.load(open(p))
            self.points = d.get('points', [])
            self.n_saved = max([q['id'] for q in self.points], default=-1) + 1
            print(f'resuming: {len(self.points)} points already in {p}')

    def save(self):
        json.dump({'captured': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'origin_offset_m': self.a.origin_offset,
                   'instrument': self.a.instrument,
                   'note': ('range_m is SLANT RANGE from the optical centre; '
                            'tape_eval.py converts to axis depth via z = r*cos(theta)'),
                   'points': self.points}, open(self._gt_path(), 'w'), indent=1)
        print(f'saved {len(self.points)} points -> {self._gt_path()}')

    # --- ui ---
    def on_mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and self.frozen is not None:
            self.click = (int(x), int(y))

    def run(self):
        cv2.namedWindow('tape', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('tape', self.on_mouse)
        print(__doc__)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.rgb is None:
                continue
            if self.frozen is None:
                view = self.rgb.copy()
                st = f'LIVE  depth:{"ok" if self.depth is not None else "--"}  ' \
                     f'tof:{"ok" if self.tof is not None else "--"}  ' \
                     f'points:{len(self.points)}   SPACE=freeze  s=save  q=quit'
                cv2.putText(view, st, (10, 30), FONT, 0.7, (0, 255, 0), 2)
            else:
                view = self._draw_frozen()
            cv2.imshow('tape', view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                self.save()
            if k == ord('u') and self.points:
                gone = self.points.pop()
                print(f'undid point {gone["id"]} ({gone["label"]})')
            if k == 27:                       # ESC unfreezes without recording
                self.frozen, self.click = None, None
            if k == 32 and self.frozen is None:
                if self.depth is None:
                    print('!! no /depth yet -- is the perception node running?')
                    continue
                self.frozen = {'rgb': self.rgb.copy(), 'depth': self.depth.copy(),
                               'var': None if self.var is None else self.var.copy(),
                               'tof': None if self.tof is None else self.tof.copy()}
                self.click = None
                print('frozen -- click the marker centre')
            if k == 13 and self.frozen is not None and self.click is not None:
                self._record()
        cv2.destroyAllWindows()
        self.save()

    def _draw_frozen(self):
        view = self.frozen['rgb'].copy()
        h, w = view.shape[:2]
        msg = 'FROZEN  click marker'
        if self.click:
            u, v = self.click
            cv2.drawMarker(view, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
            d = self.frozen['depth'][v, u]
            msg = f'({u},{v})  pipeline says {d:.3f} m   ENTER=accept  ESC=cancel'
            # Zoomed inset: at 1640x1232 on a small monitor a click can land several
            # pixels off, and a few px at 4 m can be a different surface entirely.
            x0, y0 = max(0, u - ZOOM_HALF), max(0, v - ZOOM_HALF)
            x1, y1 = min(w, u + ZOOM_HALF), min(h, v + ZOOM_HALF)
            crop = cv2.resize(self.frozen['rgb'][y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM,
                              interpolation=cv2.INTER_NEAREST)
            ch, cw = crop.shape[:2]
            cv2.drawMarker(crop, ((u - x0) * ZOOM, (v - y0) * ZOOM), (0, 0, 255),
                           cv2.MARKER_CROSS, 30, 1)
            ch, cw = min(ch, h // 2), min(cw, w // 2)
            view[0:ch, w - cw:w] = crop[:ch, :cw]
            cv2.rectangle(view, (w - cw, 0), (w - 1, ch), (0, 255, 255), 2)
        cv2.putText(view, msg, (10, 30), FONT, 0.7, (0, 255, 255), 2)
        return view

    def _record(self):
        u, v = self.click
        try:
            r = float(input(f'  slant range at ({u},{v}) in metres '
                            f'[{self.a.instrument}]: ').strip())
        except (ValueError, EOFError):
            print('  not a number -- point discarded')
            self.frozen, self.click = None, None
            return
        label = input('  label (wall / door frame / box edge / ...): ').strip() or '?'
        pid = self.n_saved
        stem = f'pt{pid:03d}'
        np.save(os.path.join(self.a.dir, stem + '_depth.npy'), self.frozen['depth'])
        cv2.imwrite(os.path.join(self.a.dir, stem + '_rgb.png'), self.frozen['rgb'])
        if self.frozen['var'] is not None:
            np.save(os.path.join(self.a.dir, stem + '_var.npy'), self.frozen['var'])
        if self.frozen['tof'] is not None:
            np.save(os.path.join(self.a.dir, stem + '_tof.npy'), self.frozen['tof'])
        self.points.append({'id': pid, 'stem': stem, 'u': u, 'v': v,
                            'range_m': r, 'label': label})
        self.n_saved += 1
        print(f'  recorded {stem}: ({u},{v}) r={r:.3f} m "{label}"  '
              f'[{len(self.points)} total]')
        self.save()                       # save after EVERY point, not at the end
        self.frozen, self.click = None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True, help='output dir (also holds tape_gt.json)')
    p.add_argument('--image-topic', default='/image')
    p.add_argument('--depth-topic', default='/depth')
    p.add_argument('--var-topic', default='/depth_var')
    # The tape origin is the optical centre, inside the lens barrel -- not the housing face
    # you can actually put a tape against. Measure that offset ONCE and record it here so
    # the correction is applied uniformly and is visible in the published JSON.
    p.add_argument('--origin-offset', type=float, default=0.0,
                   help='metres from the surface you measure FROM to the optical centre; '
                        'ADDED to every range by tape_eval.py')
    p.add_argument('--instrument', default='laser',
                   help='laser (+-1-3 mm) or tape (+-5 mm); goes into the uncertainty budget')
    a = p.parse_args()
    rclpy.init()
    try:
        TapeCapture(a).run()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

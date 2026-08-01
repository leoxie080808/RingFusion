#!/usr/bin/env python3
"""Capture independent ground truth: freeze a frame, click a marker, type its measured range.

NAMING, because "tape" is overloaded here:
  * "TAPE MEASURE" is the instrument -- the retractable ruler you measure distances with.
    "Tape ground truth" means "distances measured by hand", nothing more. A laser
    rangefinder does the same job (and this accepts --instrument laser|tape). Whatever you
    use NEVER appears in a captured frame: measure first, get it out of shot, then capture.
  * "MARKER" is the small sticker left ON the scene so you know which pixel a measurement
    belongs to. Masking or painter's tape works well for this -- it is visible, cheap and
    peels off. Markers DO stay in view; that is the entire point of them.

This is the ONLY workstream that breaks the closed loop. Every other number in this repo is
scored against the ToF the pipeline anchors to, so none of them can detect a ToF bias and
none say anything about the 92.5% of the frame the ToF never covers.

Run it on the robot with the pipeline up. For each marked point:
  SPACE   freeze the live view (and latch depth/var/tof for that instant)
  click   the FOUR CORNERS of the marker, in any order -- the centre is computed from where
          the diagonals cross, which is projectively exact however the marker is rotated or
          tilted. (One click still works if a marker is too small or too oblique to resolve
          corners; that click is then taken as the centre.)
  type    the measured range in metres, ENTER
  c       clear the current corners   u  undo the last point   s  save   q  quit
  ESC     back to the live view

Why corners rather than the centre: a square seen at an angle images as a general
quadrilateral whose centre is NOT the average of its corners -- averaging is off by a median
8.7% of the marker width and up to 1.9 widths on oblique views. The diagonal crossing is
exact. Corners are also easier to see than the middle: at 4 m the centre dot is sub-pixel
while the corners stay crisp.

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


def _quad_centre(pts):
    """Centre of a square marker from its four imaged corners.

    Take the intersection of the DIAGONALS, not the average of the corners. A square seen at
    an angle images as a general quadrilateral, and perspective does not preserve midpoints --
    averaging the corners is wrong by a median 8.7% of the marker width and, on the oblique
    views this session deliberately includes, by up to 1.9 widths (233 mm on a 12 cm marker).

    Projection does preserve lines and incidence, so the square's centre -- which is where its
    diagonals cross -- images exactly at the crossing of the imaged diagonals. Verified against
    400 random homographies: error 5.6e-16 of a marker width, i.e. float noise.

    Corners are sorted into cyclic order first, so they can be clicked in any order.
    """
    P = np.asarray(pts, float)
    c = P.mean(0)
    P = P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]
    (x1, y1), (x2, y2) = P[0], P[2]        # one diagonal
    (x3, y3), (x4, y4) = P[1], P[3]        # the other
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None                        # degenerate: three corners collinear
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return ((a * (x3 - x4) - (x1 - x2) * b) / den,
            (a * (y3 - y4) - (y1 - y2) * b) / den)


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
        self.clicks = []           # 1 click = that pixel; 4 clicks = the marker's corners
        self.n_saved = 0
        self.frame_id = 0          # which frozen frame a point was taken from
        self.pts_this_frame = 0
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
            # Continue frame numbering past whatever is already on disk, so a resumed
            # session cannot overwrite an earlier frame's arrays.
            self.frame_id = max([q.get('frame', 0) for q in self.points], default=0)
            print(f'resuming: {len(self.points)} points already in {p} '
                  f'(next frozen frame will be {self.frame_id + 1})')

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
        if ev == cv2.EVENT_LBUTTONDOWN and self.frozen is not None and len(self.clicks) < 4:
            self.clicks.append((int(x), int(y)))


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
            if k == ord('c'):                 # clear the current corner selection
                self.clicks = []
            if k == 27:                       # ESC unfreezes without recording
                self.frozen, self.clicks = None, []
            if k == 32 and self.frozen is None:
                if self.depth is None:
                    print('!! no /depth yet -- is the perception node running?')
                    continue
                self.frozen = {'rgb': self.rgb.copy(), 'depth': self.depth.copy(),
                               'var': None if self.var is None else self.var.copy(),
                               'tof': None if self.tof is None else self.tof.copy()}
                self.clicks = []
                self.frame_id += 1
                self.pts_this_frame = 0
                print(f'frozen (frame {self.frame_id}) -- click a marker. Stays frozen, so '
                      f'click/type each marker in turn. ESC = back to live.')
            if k == 13 and self.frozen is not None and self._centre() is not None:
                self._record()
        cv2.destroyAllWindows()
        self.save()

    def _draw_frozen(self):
        view = self.frozen['rgb'].copy()
        h, w = view.shape[:2]
        n = len(self.clicks)
        centre = self._centre()
        if n == 0:
            msg = (f'FROZEN frame {self.frame_id} - {self.pts_this_frame} pts - '
                   f'click the 4 CORNERS of a marker (or 1 click for its centre)')
        elif n < 4:
            msg = f'{n}/4 corners - c=clear  (or ENTER now to use a single click as the centre)'
        else:
            msg = 'centre from diagonals   ENTER=accept  c=clear  ESC=live'
        for i, (u, v) in enumerate(self.clicks):
            cv2.circle(view, (u, v), 7, (255, 200, 0), 2)
            cv2.putText(view, str(i + 1), (u + 9, v - 9), FONT, 0.6, (255, 200, 0), 2)
        if n == 4:
            P = np.asarray(self.clicks, float)
            c0 = P.mean(0)
            order = np.argsort(np.arctan2(P[:, 1] - c0[1], P[:, 0] - c0[0]))
            poly = P[order].astype(np.int32)
            cv2.polylines(view, [poly], True, (255, 200, 0), 2)
            # draw the diagonals whose crossing IS the answer, so the geometry is visible
            cv2.line(view, tuple(poly[0]), tuple(poly[2]), (0, 200, 255), 1)
            cv2.line(view, tuple(poly[1]), tuple(poly[3]), (0, 200, 255), 1)
        if centre is not None:
            cu, cv_ = int(round(centre[0])), int(round(centre[1]))
            cv2.drawMarker(view, (cu, cv_), (0, 0, 255), cv2.MARKER_CROSS, 34, 2)
            if 0 <= cv_ < h and 0 <= cu < w:
                d = self.frozen['depth'][cv_, cu]
                msg = f'centre ({cu},{cv_})  pipeline says {d:.3f} m   ' + \
                      ('ENTER=accept  c=clear' if n else 'ENTER=accept')
            # zoomed inset around the DERIVED centre -- what actually gets recorded
            x0, y0 = max(0, cu - ZOOM_HALF), max(0, cv_ - ZOOM_HALF)
            x1, y1 = min(w, cu + ZOOM_HALF), min(h, cv_ + ZOOM_HALF)
            if x1 > x0 and y1 > y0:
                crop = cv2.resize(self.frozen['rgb'][y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM,
                                  interpolation=cv2.INTER_NEAREST)
                cv2.drawMarker(crop, ((cu - x0) * ZOOM, (cv_ - y0) * ZOOM), (0, 0, 255),
                               cv2.MARKER_CROSS, 30, 1)
                ch, cw = min(crop.shape[0], h // 2), min(crop.shape[1], w // 2)
                view[0:ch, w - cw:w] = crop[:ch, :cw]
                cv2.rectangle(view, (w - cw, 0), (w - 1, ch), (0, 255, 255), 2)
        cv2.putText(view, msg, (10, 30), FONT, 0.7, (0, 255, 255), 2)
        return view

    def _centre(self):
        """The pixel that will be recorded: 4 corners -> diagonal crossing, 1 click -> itself."""
        if len(self.clicks) == 4:
            return _quad_centre(self.clicks)
        if len(self.clicks) == 1:
            return (float(self.clicks[0][0]), float(self.clicks[0][1]))
        return None

    def _record(self):
        c = self._centre()
        if c is None:
            print('  need 4 corners (or exactly 1 click) before ENTER')
            return
        # Store the sub-pixel centre AND the corners it came from, so a session can be
        # re-derived later if the centre rule is ever revised.
        uf, vf = float(c[0]), float(c[1])
        u, v = int(round(uf)), int(round(vf))
        try:
            r = float(input(f'  slant range at ({u},{v}) in metres '
                            f'[{self.a.instrument}]: ').strip())
        except (ValueError, EOFError):
            # Clear the selection but STAY frozen: a mistyped number should cost one
            # re-click, not the frozen frame and every marker still to be taken from it.
            print('  not a number -- selection cleared, still on this frame')
            self.clicks = []
            return
        label = input('  label (wall / door frame / box edge / ...): ').strip() or '?'
        pid = self.n_saved
        # Arrays are saved PER FROZEN FRAME, not per point. Every marker clicked on one
        # frozen frame shares that frame's depth/var/rgb, so writing a copy each time would
        # store ~16 MB of identical float32 per point -- 300+ MB for a 20-point session.
        # `stem` names the frame; several points legitimately share one.
        stem = f'frame{self.frame_id:03d}'
        dpath = os.path.join(self.a.dir, stem + '_depth.npy')
        if not os.path.exists(dpath):
            np.save(dpath, self.frozen['depth'])
            cv2.imwrite(os.path.join(self.a.dir, stem + '_rgb.png'), self.frozen['rgb'])
            if self.frozen['var'] is not None:
                np.save(os.path.join(self.a.dir, stem + '_var.npy'), self.frozen['var'])
            if self.frozen['tof'] is not None:
                np.save(os.path.join(self.a.dir, stem + '_tof.npy'), self.frozen['tof'])
        self.points.append({'id': pid, 'stem': stem, 'u': u, 'v': v,
                            'u_sub': uf, 'v_sub': vf,
                            'corners': [list(map(int, p)) for p in self.clicks]
                                       if len(self.clicks) == 4 else None,
                            'range_m': r, 'label': label, 'frame': self.frame_id})
        self.n_saved += 1
        print(f'  recorded {stem}: ({u},{v}) r={r:.3f} m "{label}"  '
              f'[{len(self.points)} total]')
        self.save()                       # save after EVERY point, not at the end
        # STAY FROZEN. The scene is required to be static for this whole exercise, so one
        # frozen frame can supply every marker in view -- click, type, click, type. Making
        # each point re-freeze would mean re-entering the live view ~20 times and risking a
        # frame where somebody has walked back in. ESC returns to live when you need to move
        # the robot or re-aim.
        self.clicks = []
        self.pts_this_frame += 1


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

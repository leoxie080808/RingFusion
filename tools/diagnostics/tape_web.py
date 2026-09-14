#!/usr/bin/env python3
"""Browser marker capture -- tape_capture.py's clicking UI, served over the network.

tape_capture.py opens an OpenCV window, so it needs a monitor, keyboard and mouse attached
to the Jetson. A tape session is ~20 markers plus plane groups, each measured at the marker
and then clicked at the machine: that is twenty-odd round trips across the room, and every
one of them is a chance to nudge the robot in a session whose whole premise is that nothing
moves. This serves the same UI to a phone, so the measuring and the clicking happen in the
same place.

IDENTICAL OUTPUT, DELIBERATELY. The centre rule (`_quad_centre`) and the cone classifier
(`cone_class`) are IMPORTED from tape_capture rather than reimplemented, and the on-disk
layout is byte-for-byte the same contract: one `tape_gt.json`, arrays saved per FROZEN FRAME
as `frame<NNN>_{depth,var,tof}.npy` + `_rgb.png`, points carrying
`id/stem/u/v/u_sub/v_sub/corners/range_m/label/frame/cone/role`. So tape_repeats.py,
tape_eval.py, plane_eval.py and tape_stats_r31.py all consume this with no changes, and a
session may be started in one tool and finished in the other.

WHY THE CENTRE RULE IS IMPORTED AND NOT COPIED. A square viewed obliquely images as a
general quadrilateral, and perspective does not preserve midpoints -- averaging four corners
is wrong by a median 8.7% of the marker width and by up to 1.9 widths on the oblique views
this session deliberately includes. The diagonals' crossing is exact. Two copies of that
rule would be two chances to get it wrong.

    python3 tools/diagnostics/tape_web.py \\
        --dir docs/demo/benchmarks/tape_r31 \\
        --calib ros2_ws/src/ringfusion_bringup/config/calibration.yaml \\
        --instrument laser --role heldout

then open http://<jetson-ip>:8081/ on a phone or laptop.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tape_capture import _quad_centre, cone_class                 # noqa: E402
from marker_view import cone_rect, dashed_rect                    # noqa: E402
from ringfusion_perception import geometry as geo                 # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
ZOOM, ZOOM_HALF = 6, 28

# What a usable session has to contain. Angular composition (in/edge/out of the dToF cone)
# and RANGE composition are independent requirements and both get missed: a set can sit
# entirely inside the cone, or span the cone nicely while every marker is at 2 m. Progress
# against both is shown live because a gap discovered after the scene is dismantled cannot
# be filled -- the whole point of the session is that nothing moved between repeats.
CONE_TARGET = {'in': 8, 'edge': 3, 'out': 4}
BAND_TARGET = {'near': 4, 'mid': 4, 'far': 4}
PLANE_MIN = 3          # a plane is defined by 3; the 4th is the only check that it was flat


def band_of(r, near_m, far_m):
    return 'near' if r < near_m else ('mid' if r < far_m else 'far')


class Session:
    """All shared state. Every field is touched by both the ROS thread and HTTP threads,
    so every access goes through `lock` -- a half-written frozen frame handed to a click
    would record a pixel from one image against depth from another."""

    def __init__(self, a, calib):
        self.a, self.calib = a, calib
        self.lock = threading.Lock()
        self.rgb = self.depth = self.var = self.tof = None
        self.frozen = None
        self.clicks = []
        self.points = []
        self.n_saved = 0
        self.frame_id = 0
        self.msg = 'waiting for /image and /depth'
        self.show_tof = True
        os.makedirs(a.dir, exist_ok=True)
        self._load_existing()

    # --- persistence: same contract as tape_capture --------------------------------
    def _gt_path(self):
        return os.path.join(self.a.dir, 'tape_gt.json')

    def _load_existing(self):
        p = self._gt_path()
        if os.path.exists(p):
            d = json.load(open(p))
            self.points = d.get('points', [])
            self.n_saved = max([q['id'] for q in self.points], default=-1) + 1
            self.frame_id = max([q.get('frame', 0) for q in self.points], default=0)
            print(f'resuming: {len(self.points)} points already in {p} '
                  f'(next frozen frame will be {self.frame_id + 1})')

    def save(self):
        json.dump({'captured': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'origin_offset_m': self.a.origin_offset,
                   'instrument': self.a.instrument,
                   'role': self.a.role,
                   'edge_deg': self.a.edge_deg,
                   'note': ('range_m is SLANT RANGE from the optical centre; '
                            'tape_eval.py converts to axis depth via z = r*cos(theta)'),
                   'points': self.points}, open(self._gt_path(), 'w'), indent=1)

    # --- derived ---------------------------------------------------------------------
    def centre(self):
        if len(self.clicks) == 4:
            return _quad_centre(self.clicks)
        if len(self.clicks) == 1:
            return (float(self.clicks[0][0]), float(self.clicks[0][1]))
        return None

    def tally(self):
        t = {}
        for q in self.points:
            t[str(q.get('cone'))] = t.get(str(q.get('cone')), 0) + 1
        return t

    def bands(self):
        t = {}
        for q in self.points:
            b = band_of(float(q.get('range_m', 0.0)), self.a.near_m, self.a.far_m)
            t[b] = t.get(b, 0) + 1
        return t

    def progress(self):
        """Remaining counts against both composition axes, plus under-sized plane groups."""
        cone, band = self.tally(), self.bands()
        need = {f'cone {k}': max(0, v - cone.get(k, 0)) for k, v in CONE_TARGET.items()}
        need.update({f'range {k}': max(0, v - band.get(k, 0)) for k, v in BAND_TARGET.items()})
        short = {k.split(':', 1)[1]: PLANE_MIN - v
                 for k, v in self.plane_groups().items() if v < PLANE_MIN}
        return need, short

    def plane_groups(self):
        """Plane groups need >=3 points and are better with 4+, so the count is surfaced
        live -- discovering a 2-point group after the scene is dismantled is unrecoverable."""
        g = {}
        for q in self.points:
            lb = str(q.get('label', ''))
            if lb.startswith('plane:'):
                g[lb] = g.get(lb, 0) + 1
        return g

    # --- actions ---------------------------------------------------------------------
    def freeze(self):
        with self.lock:
            if self.depth is None:
                return False, 'no /depth yet -- is the perception node running?'
            self.frozen = {'rgb': self.rgb.copy(), 'depth': self.depth.copy(),
                           'var': None if self.var is None else self.var.copy(),
                           'tof': None if self.tof is None else self.tof.copy()}
            self.clicks = []
            self.frame_id += 1
            return True, f'frozen frame {self.frame_id} -- tap a marker'

    def record(self, rng, label):
        with self.lock:
            if self.frozen is None:
                return False, 'not frozen'
            c = self.centre()
            if c is None:
                return False, 'tap 1 point (centre) or exactly 4 corners first'
            uf, vf = float(c[0]), float(c[1])
            u, v = int(round(uf)), int(round(vf))
            # Arrays are written once per FROZEN FRAME, not per point: every marker taken
            # from one frozen frame shares its depth/var/rgb, so a copy per point would be
            # ~16 MB of identical float32 each and 300+ MB for a 20-point session.
            stem = f'frame{self.frame_id:03d}'
            dpath = os.path.join(self.a.dir, stem + '_depth.npy')
            if not os.path.exists(dpath):
                np.save(dpath, self.frozen['depth'])
                cv2.imwrite(os.path.join(self.a.dir, stem + '_rgb.png'), self.frozen['rgb'])
                if self.frozen['var'] is not None:
                    np.save(os.path.join(self.a.dir, stem + '_var.npy'), self.frozen['var'])
                if self.frozen['tof'] is not None:
                    np.save(os.path.join(self.a.dir, stem + '_tof.npy'), self.frozen['tof'])
            cone = cone_class(u, v, self.calib, self.a.edge_deg) if self.calib else None
            self.points.append({'id': self.n_saved, 'stem': stem, 'u': u, 'v': v,
                                'u_sub': uf, 'v_sub': vf,
                                'corners': [list(map(int, p)) for p in self.clicks]
                                           if len(self.clicks) == 4 else None,
                                'range_m': float(rng), 'label': label or '?',
                                'frame': self.frame_id,
                                'cone': cone, 'role': self.a.role})
            self.n_saved += 1
            self.clicks = []
            self.save()             # after EVERY point, never at the end
            d = self.frozen['depth'][v, u] if (0 <= v < self.frozen['depth'].shape[0]
                                               and 0 <= u < self.frozen['depth'].shape[1]) else float('nan')
            print(f'  recorded {stem}: ({u},{v}) r={rng:.3f} m "{label}" cone={cone} '
                  f'pipeline={d:.3f} m  [{len(self.points)} total]')
            return True, f'recorded #{self.n_saved - 1} "{label}" cone={cone}'

    # --- rendering -------------------------------------------------------------------
    def render(self):
        """The frame the browser sees. Frozen frame if frozen, else live."""
        with self.lock:
            if self.frozen is not None:
                base, live = self.frozen['rgb'], False
                depth = self.frozen['depth']
            elif self.rgb is not None:
                base, live = self.rgb, True
                depth = self.depth
            else:
                return None
            view = base.copy()
            clicks = list(self.clicks)
            centre = self.centre()
            fid = self.frame_id
            tof = None if self.frozen is None else self.frozen['tof']
            if tof is None:
                tof = self.tof
            pts = [dict(q) for q in self.points]
            show_tof = self.show_tof
        h, w = view.shape[:2]

        # dToF returns, tinted by depth. This is the only live view of the scene's RANGE
        # spread, which is half the composition requirement and the half that is invisible
        # in a plain camera image -- a scene can look varied and still be all at 2 m.
        if show_tof and tof is not None and self.calib:
            valid = np.isfinite(tof) & (tof > 0.05) & (tof < 10.0)
            if valid.any():
                rows_, cols_ = tof.shape
                proj = geo.project_zone_to_pixel(
                    tof, valid, cols_, rows_, self.calib['fov_h'], self.calib['fov_v'],
                    self.calib['T_cam_tof'], self.calib['K'], self.calib['dist'],
                    model=self.calib['model'])
                uv, zc, ok = proj['uv'], proj['z_cam'], proj['valid']
                d = tof.reshape(-1)
                keep = (ok & valid.reshape(-1) & np.isfinite(uv[:, 0])
                        & np.isfinite(uv[:, 1]) & (zc > 0))
                uu = np.round(uv[keep, 0]).astype(int)
                vv_ = np.round(uv[keep, 1]).astype(int)
                dk = d[keep]
                inb = (uu >= 0) & (uu < w) & (vv_ >= 0) & (vv_ < h)
                uu, vv_, dk = uu[inb], vv_[inb], dk[inb]
                lo, hi = self.a.near_m * 0.25, self.a.far_m * 1.6
                t = np.clip((dk - lo) / max(hi - lo, 1e-6), 0, 1)
                cmap = cv2.applyColorMap((t * 255).astype(np.uint8),
                                         cv2.COLORMAP_TURBO).reshape(-1, 3)
                for i in range(len(uu)):
                    cv2.circle(view, (int(uu[i]), int(vv_[i])), 3,
                               tuple(int(c) for c in cmap[i]), -1)

        if self.calib:
            hh = float(self.calib['fov_h']) / 2.0
            hv = float(self.calib['fov_v']) / 2.0
            e = float(self.a.edge_deg)
            cv2.rectangle(view, cone_rect(self.calib, hh, hv)[:2],
                          cone_rect(self.calib, hh, hv)[2:], (0, 255, 255), 2)
            dashed_rect(view, *cone_rect(self.calib, hh + e, hv + e), (0, 200, 255))
            dashed_rect(view, *cone_rect(self.calib, hh - e, hv - e), (0, 200, 255))

        # Points already taken, so the operator can see coverage rather than remember it.
        CONE_COL = {'in': (80, 220, 80), 'edge': (0, 200, 255), 'out': (80, 80, 255)}
        for q in pts:
            col = CONE_COL.get(str(q.get('cone')), (200, 200, 200))
            cu, cvv = int(q['u']), int(q['v'])
            cv2.circle(view, (cu, cvv), 13, col, 2)
            cv2.circle(view, (cu, cvv), 2, col, -1)
            cv2.putText(view, f"{q['id']}", (cu + 15, cvv - 10), FONT, 0.55, col, 2)

        for i, (u, v) in enumerate(clicks):
            cv2.circle(view, (u, v), 9, (255, 200, 0), 2)
            cv2.putText(view, str(i + 1), (u + 11, v - 11), FONT, 0.7, (255, 200, 0), 2)
        if len(clicks) == 4:
            P = np.asarray(clicks, float)
            c0 = P.mean(0)
            poly = P[np.argsort(np.arctan2(P[:, 1] - c0[1], P[:, 0] - c0[0]))].astype(np.int32)
            cv2.polylines(view, [poly], True, (255, 200, 0), 2)
            # the diagonals whose crossing IS the recorded centre, drawn so it is visible
            cv2.line(view, tuple(poly[0]), tuple(poly[2]), (0, 200, 255), 1)
            cv2.line(view, tuple(poly[1]), tuple(poly[3]), (0, 200, 255), 1)
        if centre is not None:
            cu, cvv = int(round(centre[0])), int(round(centre[1]))
            cv2.drawMarker(view, (cu, cvv), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
            x0, y0 = max(0, cu - ZOOM_HALF), max(0, cvv - ZOOM_HALF)
            x1, y1 = min(w, cu + ZOOM_HALF), min(h, cvv + ZOOM_HALF)
            if x1 > x0 and y1 > y0:
                crop = cv2.resize(base[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM,
                                  interpolation=cv2.INTER_NEAREST)
                cv2.drawMarker(crop, ((cu - x0) * ZOOM, (cvv - y0) * ZOOM), (0, 0, 255),
                               cv2.MARKER_CROSS, 34, 2)
                ch, cw = min(crop.shape[0], h // 2), min(crop.shape[1], w // 2)
                view[0:ch, w - cw:w] = crop[:ch, :cw]
                cv2.rectangle(view, (w - cw, 0), (w - 1, ch), (0, 255, 255), 2)

        bar = ('LIVE  tap Freeze to start' if live
               else f'FROZEN frame {fid}  |  {len(clicks)} corner(s) tapped')
        cv2.rectangle(view, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(view, bar, (10, 28), FONT, 0.8, (0, 255, 0) if live else (0, 255, 255), 2)
        sc = float(self.a.scale)
        if sc != 1.0:
            view = cv2.resize(view, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        return view


class Ros(Node):
    def __init__(self, sess, a):
        super().__init__('tape_web')
        self.s = sess
        self.create_subscription(Image, a.image_topic, self.on_img, 5)
        self.create_subscription(Image, a.depth_topic, self.on_depth, 5)
        self.create_subscription(Image, a.var_topic, self.on_var, 5)
        self.create_subscription(ToFFrame, '/tof', self.on_tof, 5)

    def on_img(self, m):
        b = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        with self.s.lock:
            self.s.rgb = b[:, :, ::-1].copy() if m.encoding == 'rgb8' else b.copy()

    def on_depth(self, m):
        with self.s.lock:
            self.s.depth = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()

    def on_var(self, m):
        with self.s.lock:
            self.s.var = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()

    def on_tof(self, m):
        if len(m.dist_m) == m.rows * m.cols:
            with self.s.lock:
                self.s.tof = np.asarray(m.dist_m, np.float32).reshape(m.rows, m.cols).copy()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>tape capture</title><style>
:root{color-scheme:dark}
body{margin:0;background:#111;color:#eee;font:14px/1.4 system-ui,sans-serif}
#wrap{position:relative;width:100%}
img{width:100%;display:block;touch-action:manipulation}
.bar{display:flex;gap:6px;padding:8px;flex-wrap:wrap;background:#1a1a1a;position:sticky;top:0;z-index:5}
button{flex:1 1 auto;min-width:74px;padding:12px 8px;font-size:15px;border:0;border-radius:7px;
 background:#2d6cdf;color:#fff;font-weight:600}
button.sec{background:#333}button.warn{background:#a33}
form{display:flex;gap:6px;padding:8px;flex-wrap:wrap;background:#1a1a1a}
input{flex:1 1 110px;padding:12px;font-size:16px;border-radius:7px;border:1px solid #444;
 background:#222;color:#eee}
#st{padding:8px 10px;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:13px}
.ok{color:#6d6}.bad{color:#f77}
#prog{display:flex;gap:5px;padding:6px 8px;background:#161616;flex-wrap:wrap}
.chip{flex:1 1 84px;padding:6px 8px;border-radius:6px;background:#242424;font-size:12px;
 font-family:ui-monospace,monospace;border-left:4px solid #555}
.chip b{display:block;font-size:15px;margin-top:2px}
.done{border-left-color:#6d6;color:#9f9}.todo{border-left-color:#c84;color:#fc9}
</style></head><body>
<div class="bar">
 <button onclick="post('/freeze')">Freeze</button>
 <button class="sec" onclick="post('/unfreeze')">Live</button>
 <button class="sec" onclick="post('/clear')">Clear taps</button>
 <button class="warn" onclick="if(confirm('Undo last point?'))post('/undo')">Undo pt</button>
 <button class="sec" onclick="post('/tof')">dToF dots</button>
</div>
<div id="prog"></div>
<div id="wrap"><img id="v" src="/frame.jpg" onclick="tap(event)"></div>
<form onsubmit="rec(event)">
 <input id="r" type="number" step="0.001" inputmode="decimal" placeholder="range m" required>
 <input id="l" type="text" placeholder="label (or plane:floor)" required>
 <button type="submit">Record</button>
</form>
<div id="st">loading…</div>
<script>
const v=document.getElementById('v'),st=document.getElementById('st');
let busy=false;
function refresh(){ if(!busy) v.src='/frame.jpg?t='+Date.now(); }
function tap(e){
  const b=v.getBoundingClientRect();
  // The served image is scaled; map the tap back to FULL-RES pixel coords before sending,
  // because the recorded u,v must index the full-resolution depth map.
  const u=(e.clientX-b.left)/b.width, w=(e.clientY-b.top)/b.height;
  fetch('/click',{method:'POST',body:JSON.stringify({fu:u,fv:w})})
    .then(()=>{ v.src='/frame.jpg?t='+Date.now(); state(); });
}
function post(p){ fetch(p,{method:'POST'}).then(state); }
function rec(e){
  e.preventDefault();
  const r=document.getElementById('r'), l=document.getElementById('l');
  busy=true;
  fetch('/record',{method:'POST',body:JSON.stringify({range_m:parseFloat(r.value),label:l.value})})
    .then(x=>x.json()).then(d=>{ busy=false; if(d.ok){r.value='';} state(); });
}
function state(){
  fetch('/state').then(x=>x.json()).then(d=>{
    let h='';
    const mk=(lbl,got,want)=>{const ok=got>=want;
      h+='<div class="chip '+(ok?'done':'todo')+'">'+lbl+'<b>'+got+'/'+want+'</b></div>';};
    mk('cone in',d.cone.in[0],d.cone.in[1]);
    mk('cone edge',d.cone.edge[0],d.cone.edge[1]);
    mk('cone out',d.cone.out[0],d.cone.out[1]);
    mk('&lt;'+d.band_edges[0]+'m',d.band.near[0],d.band.near[1]);
    mk(d.band_edges[0]+'-'+d.band_edges[1]+'m',d.band.mid[0],d.band.mid[1]);
    mk('&gt;'+d.band_edges[1]+'m',d.band.far[0],d.band.far[1]);
    document.getElementById('prog').innerHTML=h;
    st.innerHTML='<span class="'+(d.ready?'ok':'bad')+'">'+d.hint+'</span>\n'+
      d.msg+'\npoints '+d.n+'   taps '+d.clicks+
      (d.depth!=null?'   pipeline '+d.depth.toFixed(3)+' m':'')+
      '\nstill need: '+d.need+
      '\nplanes: '+d.planes+(d.plane_short?'  ('+d.plane_short+')':'');
    v.src='/frame.jpg?t='+Date.now();
  });
}
setInterval(()=>{ if(!busy) state(); }, 1200);
state();
</script></body></html>"""


def make_handler(sess):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass                       # the console belongs to the capture log

        def _send(self, code, ctype, body):
            # A browser cancels the in-flight frame request every time it asks for a new
            # one, so a broken pipe here is routine, not an error. Without this the log
            # fills with tracebacks and a real failure becomes impossible to spot.
            try:
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def handle_one_request(self):
            try:
                BaseHTTPRequestHandler.handle_one_request(self)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def do_GET(self):
            p = self.path.split('?')[0]
            if p == '/':
                return self._send(200, 'text/html; charset=utf-8', PAGE.encode())
            if p == '/frame.jpg':
                img = sess.render()
                if img is None:
                    return self._send(503, 'text/plain', b'no frame yet')
                ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return self._send(200, 'image/jpeg', buf.tobytes())
            if p == '/state':
                with sess.lock:
                    c = sess.centre()
                    d = None
                    if c is not None and sess.frozen is not None:
                        u, v = int(round(c[0])), int(round(c[1]))
                        dm = sess.frozen['depth']
                        if 0 <= v < dm.shape[0] and 0 <= u < dm.shape[1]:
                            d = float(dm[v, u])
                            if not np.isfinite(d):
                                d = None
                    nc = len(sess.clicks)
                    if sess.frozen is None:
                        hint = 'LIVE - tap Freeze to begin'
                    elif nc == 0:
                        hint = 'tap 1x for centre, or 4x for corners'
                    elif nc == 1:
                        hint = 'READY - 1 tap = centre. Record, or tap 3 more for corners'
                    elif nc == 4:
                        hint = 'READY - 4 corners, centre from diagonals'
                    else:
                        hint = f'{nc} taps is not usable - need 1 or 4. Tap {4 - nc} more, or Clear'
                    need, short = sess.progress()
                    cone, band = sess.tally(), sess.bands()
                    body = {'ok': True, 'msg': sess.msg, 'n': len(sess.points),
                            'clicks': nc, 'hint': hint,
                            'ready': nc in (1, 4) and sess.frozen is not None, 'depth': d,
                            'show_tof': sess.show_tof,
                            'cone': {k: [cone.get(k, 0), v] for k, v in CONE_TARGET.items()},
                            'band': {k: [band.get(k, 0), v] for k, v in BAND_TARGET.items()},
                            'band_edges': [sess.a.near_m, sess.a.far_m],
                            'need': ', '.join(f'{k} +{v}' for k, v in need.items() if v) or 'all targets met',
                            'planes': ', '.join(f'{k.split(":",1)[1]}={v}'
                                                for k, v in sorted(sess.plane_groups().items())) or '-',
                            'plane_short': ', '.join(f'{k} needs +{v}' for k, v in short.items()) or ''}
                return self._send(200, 'application/json', json.dumps(body).encode())
            return self._send(404, 'text/plain', b'no')

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            raw = self.rfile.read(n) if n else b'{}'
            try:
                body = json.loads(raw or b'{}')
            except ValueError:
                body = {}
            p = self.path.split('?')[0]
            ok, msg = True, sess.msg

            if p == '/freeze':
                ok, msg = sess.freeze()
            elif p == '/unfreeze':
                with sess.lock:
                    sess.frozen, sess.clicks = None, []
                msg = 'live'
            elif p == '/tof':
                with sess.lock:
                    sess.show_tof = not sess.show_tof
                msg = f'dToF dots {"on" if sess.show_tof else "off"}'
            elif p == '/clear':
                with sess.lock:
                    sess.clicks = []
                msg = 'taps cleared'
            elif p == '/undo':
                with sess.lock:
                    if sess.points:
                        g = sess.points.pop()
                        sess.n_saved = max(0, sess.n_saved - 1)
                        sess.save()
                        msg = f'undid #{g["id"]} "{g["label"]}"'
                    else:
                        ok, msg = False, 'nothing to undo'
            elif p == '/click':
                with sess.lock:
                    if sess.frozen is None:
                        ok, msg = False, 'freeze first'
                    elif len(sess.clicks) >= 4:
                        ok, msg = False, '4 corners already -- Record or Clear'
                    else:
                        h, w = sess.frozen['rgb'].shape[:2]
                        u = int(round(float(body.get('fu', 0)) * w))
                        vv = int(round(float(body.get('fv', 0)) * h))
                        u, vv = max(0, min(w - 1, u)), max(0, min(h - 1, vv))
                        sess.clicks.append((u, vv))
                        msg = f'tap {len(sess.clicks)} at ({u},{vv})'
            elif p == '/record':
                try:
                    rng = float(body.get('range_m'))
                except (TypeError, ValueError):
                    ok, msg = False, 'range must be a number'
                else:
                    ok, msg = sess.record(rng, str(body.get('label', '')).strip())
            else:
                return self._send(404, 'text/plain', b'no')

            sess.msg = msg
            return self._send(200, 'application/json',
                              json.dumps({'ok': ok, 'msg': msg}).encode())
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    p.add_argument('--calib', default='')
    p.add_argument('--port', type=int, default=8081)
    p.add_argument('--scale', type=float, default=0.75,
                   help='serve at this fraction of full res; taps are mapped back to full res')
    p.add_argument('--origin-offset', type=float, default=0.0)
    p.add_argument('--instrument', default='laser')
    p.add_argument('--edge-deg', type=float, default=3.0)
    # Range bands. The session needs markers spanning ~0.3-4.0 m; these two edges cut that
    # span into roughly equal thirds so "near/mid/far" is a real spread rather than a label.
    p.add_argument('--near-m', type=float, default=1.3, help='near/mid range-band edge')
    p.add_argument('--far-m', type=float, default=2.6, help='mid/far range-band edge')
    p.add_argument('--role', default='heldout', choices=['heldout', 'calibration'])
    p.add_argument('--image-topic', default='/image')
    p.add_argument('--depth-topic', default='/depth')
    p.add_argument('--var-topic', default='/depth_var')
    a = p.parse_args()

    calib = None
    if a.calib:
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, os.path.join(_repo, 'training'))
        from anchoring_bridge import calib_from_yaml
        calib = calib_from_yaml(a.calib, train_size=(1232, 1640))

    sess = Session(a, calib)
    rclpy.init()
    node = Ros(sess, a)

    srv = ThreadingHTTPServer(('0.0.0.0', a.port), make_handler(sess))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ip = os.popen('hostname -I').read().split()
    print(f'\n  open  http://{ip[0] if ip else "<jetson-ip>"}:{a.port}/  on a phone or laptop')
    print(f'  writing to {os.path.abspath(a.dir)}\n')

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    sess.save()
    print(f'saved {len(sess.points)} points -> {sess._gt_path()}')
    srv.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

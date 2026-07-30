#!/usr/bin/env python3
"""Score RingFusion on ZJU-L5 -- the first OPEN-LOOP evaluation we have.

Every other number in this project is scored against the ToF we anchor to, so it cannot
detect a ToF bias and says nothing about pixels the ToF never sees. ZJU-L5 ships DENSE
ground truth from a RealSense 435i on a calibrated rig with the sparse sensor, so here the
thing being measured and the thing measuring it are different devices.

WHY THE CLOSED-FORM PATH TRANSFERS AND NETWORK B DOES NOT
---------------------------------------------------------
The affine fit (anchoring.solve_robust) has NO learned parameters -- it re-solves per frame
from whatever anchors it is given -- so it transfers to a different sensor by construction.
That is the publishable claim here. Network B was trained on 32x32 anchor geometry at our
intrinsics and is out of domain on an 8x8 grid; it is reported for completeness and should
not be read as a like-for-like number.

WHAT IS DIFFERENT ABOUT THIS SENSOR (state these next to any result)
  * 8x8 = 64 zones, median 41 valid per frame, vs our 32x32 with ~830 valid.
  * Each zone's iFoV is ~62x62 px, so the zones TILE ~50% of the 480x640 frame.
    Ours cover 7.5%. "Outside coverage" here is a frame-edge band, not our situation.
  * Zone depths span ~0.33-3.24 m: closer-range indoor scenes than our logs.
  * Our Network A backbone was distilled on OUR rectified fisheye at 1640x1232 and is
    being fed RealSense RGB at 480x640 -- a real domain shift, and rho(disp, 1/z) at the
    anchors is printed so it can be judged rather than assumed.

The projection is NOT re-derived here: the dataset gives each zone's pixel rectangle in
`fr`, so the ToF->camera mapping is theirs, already applied. That removes the single
largest risk in porting to a new rig (a mirrored or transposed grid -- the exact bug that
cost this project a week on its own sensor).

Sample fields (see deltar/vis_data.py):
    rgb        (480,640,3) uint8
    depth      (480,640)   float32  RealSense dense GT, 0 and >=65.5 are invalid
    fr         (64,4)      int32    per-zone [sy,sx,ey,ex] pixel rect, may go negative
    hist_data  (64,2)      float64  [:,0] = zone depth (m)
    mask       (64,)       bool     zone validity
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_perception'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ringfusion_perception import anchoring as anc          # noqa: E402
from ringfusion_perception.blend import blend_depth, apply_scene_cap   # noqa: E402
from ringfusion_perception.residual import MAX_DEPTH_M       # noqa: E402
import metrics as M                                         # noqa: E402

GT_MIN, GT_MAX = 0.1, 20.0        # RealSense sentinel is 65.535; 0 means no return
METHODS = ('B1_nearest', 'B2_bilinear', 'B4c_affine_cl', 'B5_ringfusion',
           'B6_blend_over_D0', 'B6_blend_over_net')


def zone_anchors(fr, hist, mask, h, w):
    """-> (u, v, z) at the centre of each valid zone's pixel rect, clipped to the image."""
    u, v, z = [], [], []
    for i in np.flatnonzero(mask):
        sy, sx, ey, ex = fr[i]
        sy, sx = max(0, int(sy)), max(0, int(sx))
        ey, ex = min(h, int(ey)), min(w, int(ex))
        if ey <= sy or ex <= sx:
            continue                       # zone lies entirely off-image
        d = float(hist[i, 0])
        if not np.isfinite(d) or d <= 0:
            continue
        v.append((sy + ey) // 2)
        u.append((sx + ex) // 2)
        z.append(d)
    return np.array(u, int), np.array(v, int), np.array(z, np.float64)


def coverage_mask(fr, mask, h, w):
    """Union of the valid zones' pixel rects -- where the ToF actually measured."""
    cm = np.zeros((h, w), bool)
    for i in np.flatnonzero(mask):
        sy, sx, ey, ex = fr[i]
        sy, sx = max(0, int(sy)), max(0, int(sx))
        ey, ex = min(h, int(ey)), min(w, int(ex))
        if ey > sy and ex > sx:
            cm[sy:ey, sx:ex] = True
    return cm


def sparse_fill(u, v, z, h, w, method):
    """Dense depth from the zone samples alone -- no camera. 'nearest' is a Voronoi fill,
    'linear' interpolates and leaves NaN outside the convex hull (reported as coverage)."""
    from scipy.interpolate import griddata
    gy, gx = np.mgrid[0:h, 0:w]
    pts = np.stack([v, u], 1).astype(np.float64)
    try:
        out = griddata(pts, z, (gy, gx), method=method)
    except Exception:
        return np.full((h, w), np.nan)
    return out.astype(np.float32)


class _Teacher:
    """Depth Anything V2 via transformers, matching training/cache_teacher.py exactly:
    `predicted_depth` is disparity-like inverse depth (larger = closer), bilinearly
    resized to the source image size. Same convention the student was distilled on, so
    the affine fit downstream is unchanged."""

    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        self.torch = torch
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.dev).eval()

    def infer(self, rgb):
        import torch.nn.functional as F
        with self.torch.no_grad():
            inp = self.proc(images=rgb, return_tensors='pt').to(self.dev)
            pred = self.model(**inp).predicted_depth
            pred = F.interpolate(pred.unsqueeze(1).float(), size=rgb.shape[:2],
                                 mode='bilinear', align_corners=False)
        return np.clip(pred[0, 0].cpu().numpy(), 1e-3, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dir containing ZJUL5/ and data.json')
    ap.add_argument('--backbone-engine', default='')
    # THE CONTROL THAT MATTERS. "The closed-form path transfers by construction" is only
    # true of the FIT -- the pipeline still contains Network A, which is learned and does
    # not transfer. student_v3 was distilled on our rectified fisheye at 1640x1232; on
    # RealSense RGB it scores rho 0.519 against 0.917 on our own sensor. Running the
    # domain-general Depth Anything V2 teacher in its place separates "our anchoring math
    # does not transfer" from "our small distilled student does not generalise".
    ap.add_argument('--teacher', default='',
                    help="e.g. depth-anything/Depth-Anything-V2-Large-hf; replaces the engine")
    ap.add_argument('--residual-engine', default='')
    ap.add_argument('--split', default='test', choices=['test', 'train', 'all'])
    ap.add_argument('--limit', type=int, default=0)
    # No intrinsics ship with the dataset. The blend only uses fx to turn its angular
    # ramp into pixels, so an approximate value shifts the ramp slightly and nothing else.
    # 462 ~ RealSense D435 RGB at 640x480 (~69 deg HFOV).
    ap.add_argument('--blend-fx', type=float, default=462.0)
    # What the blend mixes the ToF WITH. 'd0' = the closed-form depth (the right choice
    # here: Network B is out of domain on 8x8 and blending over it just inherits its
    # error -- 2.083 m vs ~0.33 m). 'net' deliberately measures that propagation.
    ap.add_argument('--blend-over', choices=['d0', 'net'], default='d0')
    # Scene-bounded far-field cap. MUST be swept on --split train, never on test.
    # Our own logs cannot tune this: they are scored only at ToF zone pixels inside the
    # cone, while the blow-ups live in regions with no zones -- after the 20 m clamp our
    # zone-pixel bias is -0.290 m (UNDER-reading), so a cap has nothing to act on there.
    # ZJU-L5's dense GT is the only evaluation that can see the tail.
    ap.add_argument('--sweep-k', type=float, nargs='*', default=[])
    # Bracket the affine -> scale-only spectrum. b sets the ceiling 1/b; ridging b toward
    # 0 lifts it, and b_prior=inf is exact scale-only (no ceiling). Sweep on TRAIN.
    # Predictions to check, stated before running: bias -> 0, RMSE falls, far-field sigma
    # grows (D^4 inherits the ceiling), near-field medAE degrades slightly. If near-field
    # does NOT degrade, the prior is not doing anything.
    ap.add_argument('--sweep-bprior', type=float, nargs='*', default=[],
                    help='e.g. --sweep-bprior 0.01 0.1 1 inf')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import h5py
    if not (a.backbone_engine or a.teacher):
        sys.exit('need --backbone-engine or --teacher')

    meta = json.load(open(os.path.join(a.root, 'ZJUL5', 'data.json')))
    names = ([m['filename'] for m in meta[a.split]] if a.split != 'all'
             else [m['filename'] for k in meta for m in meta[k]])
    if a.limit:
        names = names[:a.limit]
    print(f'{len(names)} samples ({a.split} split)')

    if a.teacher:
        backbone = _Teacher(a.teacher)
        print(f'backbone: TEACHER {a.teacher}')
    else:
        from ringfusion_perception.backbone import TensorRTBackbone
        backbone = TensorRTBackbone(a.backbone_engine)
        print(f'backbone: engine {os.path.basename(a.backbone_engine)}')
    residual = None
    if a.residual_engine:
        from ringfusion_perception.residual import ResidualRefiner
        residual = ResidualRefiner(a.residual_engine)

    # Three regions: everywhere, inside the ToF footprint, and outside it. The OUTSIDE
    # column is the one no ToF-scored protocol can produce -- independent GT where the
    # sparse sensor measured nothing.
    methods = (list(METHODS) + [f'B4k{k:g}' for k in a.sweep_k]
               + [f'B4bp{b:g}' for b in a.sweep_bprior])
    acc = {r: {m: {'p': [], 'g': []} for m in methods} for r in ('all', 'in', 'out')}
    rhos, n_anch, skipped = [], [], 0

    for n, rel in enumerate(names):
        p = os.path.join(a.root, 'ZJUL5', rel)
        if not os.path.exists(p):
            skipped += 1
            continue
        with h5py.File(p, 'r') as f:
            rgb = f['rgb'][:]
            gt = f['depth'][:].astype(np.float64)
            fr, hist, mask = f['fr'][:], f['hist_data'][:], f['mask'][:]
        h, w = gt.shape
        u, v, z = zone_anchors(fr, hist, mask, h, w)
        if u.size < 8:
            skipped += 1
            continue
        n_anch.append(u.size)

        disp = backbone.infer(np.ascontiguousarray(rgb))
        disp_at = disp[v, u].astype(np.float64)
        # Projection/domain sanity: if the backbone's disparity does not track true
        # inverse depth at the anchors, nothing downstream is meaningful.
        if u.size >= 4 and np.std(disp_at) > 0:
            rhos.append(float(np.corrcoef(disp_at, 1.0 / z)[0, 1]))

        inv = 1.0 / z
        wts = np.ones_like(z)
        fit = anc.solve_robust(disp_at, inv, wts, iters=1)
        if fit is None:
            skipped += 1
            continue
        a_, b_ = fit
        D0 = np.clip(anc.to_metric_depth(disp, a_, b_), None, MAX_DEPTH_M)

        preds = {
            'B1_nearest': sparse_fill(u, v, z, h, w, 'nearest'),
            'B2_bilinear': sparse_fill(u, v, z, h, w, 'linear'),
            'B4c_affine_cl': D0,
        }
        ad = np.zeros((h, w), np.float32); ad[v, u] = z.astype(np.float32)
        am = np.zeros((h, w), np.float32); am[v, u] = 1.0
        D_net = D0
        if residual is not None:
            D, _ = residual.refine(np.ascontiguousarray(rgb), D0, disp, ad, am, a_, b_)
            D = np.clip(D, None, MAX_DEPTH_M)
            preds['B5_ringfusion'] = D
            if a.blend_over == 'net':
                D_net = D
        # Blend over D0 by DEFAULT here, not over the residual. Network B is out of domain
        # on an 8x8 grid (measured: MAE 2.825 m, delta<1.25 = 0.063), and blending over a
        # broken source just propagates it -- B6 scored 2.083 m when fed the residual and
        # ~0.33 m when fed D0. Blending cannot repair a bad network, it only mixes; use
        # --blend-over net to measure that propagation deliberately.
        preds['B6_blend_over_D0' if a.blend_over == 'd0' else 'B6_blend_over_net'] = \
            blend_depth(D_net, ad, am, fx=a.blend_fx)[0]
        for kk in a.sweep_k:
            preds[f'B4k{kk:g}'] = apply_scene_cap(D0, ad, am, k=kk)[0]
        for bp in a.sweep_bprior:
            fp = anc.solve_robust(disp_at, inv, wts, iters=1, b_prior=bp)
            if fp is None:
                continue
            preds[f'B4bp{bp:g}'] = np.clip(anc.to_metric_depth(disp, *fp), None, MAX_DEPTH_M)

        gv = (gt > GT_MIN) & (gt < GT_MAX)
        cm = coverage_mask(fr, mask, h, w)
        for region, sel in (('all', gv), ('in', gv & cm), ('out', gv & ~cm)):
            if not sel.any():
                continue
            for k, P in preds.items():
                acc[region][k]['p'].append(np.asarray(P, np.float64)[sel])
                acc[region][k]['g'].append(gt[sel])
        if (n + 1) % 100 == 0:
            print(f'  {n+1}/{len(names)}', flush=True)

    print(f'\nskipped {skipped}; median {np.median(n_anch):.0f} anchors/frame')
    if rhos:
        print(f'rho(disparity, 1/z) at anchors: median {np.median(rhos):.3f}  '
              f'(our own sensor after the projection fix: 0.917)')
        if np.median(rhos) < 0.5:
            print('  ** LOW -- backbone domain shift or projection issue; '
                  'treat every number below as suspect **')

    report = {'split': a.split, 'n': len(names), 'rho_median': float(np.median(rhos)) if rhos else None,
              'regions': {}}
    for region, label in (('all', 'ALL pixels'),
                          ('in', 'INSIDE ToF footprint (~50% of frame)'),
                          ('out', 'OUTSIDE ToF footprint -- independent GT, no ToF')):
        rows = []
        for k in methods:
            if not acc[region][k]['p']:
                continue
            mm = M.depth_metrics(np.concatenate(acc[region][k]['p']),
                                 np.concatenate(acc[region][k]['g']))
            rows.append((k, mm))
        if not rows:
            continue
        print(f'\n=== {label} ===')
        print(M.format_table(rows))
        report['regions'][region] = {k: v for k, v in rows}

    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()

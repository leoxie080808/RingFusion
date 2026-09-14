#!/usr/bin/env python3
"""Dense accuracy against a TAPE-ANCHORED plane, including outside the dToF cone.

The reviewer asked for accuracy inside and outside the dToF coverage area. Inside is
answerable from withheld zones. Outside is not: every evaluation target in the withheld-
zone protocol IS a dToF zone, and zones exist only inside the cone. The tape reference
covers outside, but with four points and a 30x interval on the median it can validate and
cannot measure.

A plane fixes that. Three tape-measured points define one exactly, and every pixel inside
the region it spans then carries ground truth -- ~10^5 of them per capture instead of four,
in the same AbsRel / RMSE / delta-1 form as the rest of the paper, and available wherever
a flat surface can be put, which includes the periphery the cone never reaches.

TWO RULES, AND THE RESULT IS WORTHLESS WITHOUT EITHER
  1. The plane comes from TAPE, never from a dToF fit. roi.fit_ground_plane exists and
     would be easier, but it fits the plane through the same measurements the pipeline
     anchors to, so scoring against it measures self-consistency -- the exact circularity
     this workstream exists to escape. Here the plane is fitted to backprojected tape
     ranges and the dToF is not consulted.
  2. The region is declared, not inferred. Deciding which pixels are floor by asking
     whether their predicted depth looks floor-like grades the prediction against itself.
     The operator marks the region by measuring its corners, and only pixels inside the
     resulting hull are scored.

CAPTURE PROTOCOL. Use tape_capture.py as normal. For each planar target, click and measure
at least three points on it -- corners are easiest to hit precisely -- and give them all
the same label prefixed `plane:`, e.g. `plane:floor_left`, `plane:board_2m`. This tool
groups by that label, fits a plane per group, and scores the region its points enclose.
Four or more points per group is better than three: the extra ones become a planarity
residual, which is the only check on whether the surface really was flat and the ranges
really were right.

    python3 tools/diagnostics/plane_eval.py --dir <tape capture dir> \
        --calib ros2_ws/src/ringfusion_bringup/config/calibration.yaml \
        --out docs/demo/benchmarks/plane_eval_r31.json
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anchoring_bridge import calib_from_yaml                      # noqa: E402
from bootstrap import frame_bootstrap                             # noqa: E402
import envinfo                                                    # noqa: E402

PLANE_PREFIX = 'plane:'


def rays(u, v, K):
    """Unit ray direction per pixel, camera frame. u, v may be arrays."""
    fx, fy, cx, cy = [float(x) for x in np.asarray(K, float).ravel()[:4]]
    d = np.stack([(np.asarray(u, float) - cx) / fx,
                  (np.asarray(v, float) - cy) / fy,
                  np.ones_like(np.asarray(u, float))], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def fit_plane(P):
    """Least-squares plane through 3D points -> (n, c) with n.P + c = 0, |n| = 1."""
    P = np.asarray(P, float)
    centroid = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - centroid)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    return n, float(-n @ centroid)


def cone_rect(calib):
    fx, fy, cx, cy = [float(x) for x in np.asarray(calib['K'], float).ravel()[:4]]
    dx = fx * np.tan(np.deg2rad(float(calib['fov_h']) / 2.0))
    dy = fy * np.tan(np.deg2rad(float(calib['fov_v']) / 2.0))
    return cx - dx, cy - dy, cx + dx, cy + dy


def metrics(pred, gt):
    e = pred - gt
    ae = np.abs(e)
    return {
        'n_px': int(len(e)),
        'medae': round(float(np.median(ae)), 4),
        'mae': round(float(ae.mean()), 4),
        'rmse': round(float(np.sqrt((e ** 2).mean())), 4),
        'absrel': round(float((ae / np.maximum(gt, 1e-6)).mean()), 4),
        'bias': round(float(e.mean()), 4),
        'd1': round(float((np.maximum(pred / np.maximum(gt, 1e-6),
                                      gt / np.maximum(pred, 1e-6)) < 1.25).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='dir written by tape_capture.py')
    ap.add_argument('--calib', required=True)
    ap.add_argument('--gt', default='')
    ap.add_argument('--max-planarity-mm', type=float, default=25.0,
                    help='reject a group whose tape points do not lie on a plane to within this')
    ap.add_argument('--erode-px', type=int, default=6,
                    help='shrink each region by this much so marker tape and hull edges '
                         'do not contribute')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import cv2

    gt_path = a.gt or os.path.join(a.dir, 'tape_gt.json')
    meta = json.load(open(gt_path))
    pts = meta['points']
    origin = float(meta.get('origin_offset_m', 0.0))
    calib = calib_from_yaml(a.calib, train_size=(1232, 1640))
    K = calib['K']

    groups = {}
    for p in pts:
        lab = str(p.get('label', ''))
        if lab.startswith(PLANE_PREFIX):
            groups.setdefault(lab[len(PLANE_PREFIX):].strip(), []).append(p)
    if not groups:
        sys.exit(f'no points labelled "{PLANE_PREFIX}<name>" in {gt_path}. See the capture '
                 f'protocol in this file\'s docstring.')

    print(f'{len(groups)} plane group(s) in {os.path.basename(a.dir.rstrip("/"))}')
    x0c, y0c, x1c, y1c = cone_rect(calib)
    report = {
        'label': 'r31 dense accuracy against tape-anchored planes',
        'env': envinfo.capture(note='plane_eval'),
        'capture_dir': os.path.abspath(a.dir),
        'origin_offset_m': origin,
        'cone_rect_px': [round(v, 1) for v in (x0c, y0c, x1c, y1c)],
        'planes': {}, 'pooled': {},
    }
    pooled = {'all': [], 'in': [], 'out': []}
    pooled_groups = {'all': [], 'in': [], 'out': []}

    for name, gp in sorted(groups.items()):
        if len(gp) < 3:
            print(f'  {name}: only {len(gp)} points, need 3 -- skipped')
            report['planes'][name] = {'skipped': 'fewer than 3 tape points'}
            continue
        uv = np.array([[p['u'], p['v']] for p in gp], float)
        r = np.array([float(p['range_m']) + origin for p in gp], float)
        P = rays(uv[:, 0], uv[:, 1], K) * r[:, None]          # tape ranges only
        n, c = fit_plane(P)
        resid = np.abs(P @ n + c)
        planarity_mm = float(resid.max() * 1000.0)
        if planarity_mm > a.max_planarity_mm:
            print(f'  {name}: tape points are {planarity_mm:.0f} mm off a plane '
                  f'(limit {a.max_planarity_mm:.0f}) -- skipped, check the ranges')
            report['planes'][name] = {'skipped': 'not planar',
                                      'planarity_mm': round(planarity_mm, 1)}
            continue

        stem = gp[0]['stem']
        dpath = os.path.join(a.dir, stem + '_depth.npy')
        if not os.path.exists(dpath):
            print(f'  {name}: no {stem}_depth.npy -- skipped')
            continue
        depth = np.load(dpath).astype(np.float64)
        h, w = depth.shape

        # Region: the hull the measured points enclose, eroded so the marker tape itself
        # and the hull boundary do not contribute.
        mask = np.zeros((h, w), np.uint8)
        hull = cv2.convexHull(uv.astype(np.int32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 1)
        if a.erode_px > 0:
            k = np.ones((a.erode_px * 2 + 1,) * 2, np.uint8)
            mask = cv2.erode(mask, k)
        vs, us = np.nonzero(mask)
        if len(us) < 500:
            print(f'  {name}: region is only {len(us)} px after erosion -- skipped')
            continue

        # Ground truth per pixel: where its ray pierces the tape-fitted plane.
        d = rays(us, vs, K)
        denom = d @ n
        ok = np.abs(denom) > 1e-6
        t = np.where(ok, -c / np.where(ok, denom, 1.0), np.nan)
        gt_z = t * d[:, 2]                                    # axial depth, as /depth is
        pred = depth[vs, us]
        good = ok & np.isfinite(gt_z) & (gt_z > 0.05) & (gt_z < 20.0) & np.isfinite(pred) & (pred > 0)
        us, vs, gt_z, pred = us[good], vs[good], gt_z[good], pred[good]
        if len(us) < 500:
            print(f'  {name}: {len(us)} usable px -- skipped')
            continue

        inside = (us >= x0c) & (us <= x1c) & (vs >= y0c) & (vs <= y1c)
        ent = {
            'n_tape_points': len(gp),
            'planarity_mm': round(planarity_mm, 2),
            'plane_normal': [round(float(x), 5) for x in n],
            'plane_offset_m': round(c, 5),
            'tape_range_span_m': [round(float(r.min()), 3), round(float(r.max()), 3)],
            'stem': stem,
            'frac_inside_cone': round(float(inside.mean()), 4),
            'all': metrics(pred, gt_z),
        }
        if inside.sum() >= 200:
            ent['inside_cone'] = metrics(pred[inside], gt_z[inside])
        if (~inside).sum() >= 200:
            ent['outside_cone'] = metrics(pred[~inside], gt_z[~inside])
        report['planes'][name] = ent

        pooled['all'].append((pred, gt_z))
        pooled_groups['all'].append(np.abs(pred - gt_z))
        if inside.sum():
            pooled['in'].append((pred[inside], gt_z[inside]))
            pooled_groups['in'].append(np.abs(pred[inside] - gt_z[inside]))
        if (~inside).sum():
            pooled['out'].append((pred[~inside], gt_z[~inside]))
            pooled_groups['out'].append(np.abs(pred[~inside] - gt_z[~inside]))

        ic = ent.get('inside_cone', {}).get('medae')
        oc = ent.get('outside_cone', {}).get('medae')
        print(f"  {name:<18} {len(us):>7} px  planarity {planarity_mm:5.1f} mm  "
              f"medAE all {ent['all']['medae']:.4f}"
              + (f"  in {ic:.4f}" if ic is not None else '')
              + (f"  out {oc:.4f}" if oc is not None else ''))

    # Pooled, with a bootstrap over PLANE GROUPS -- pixels within one plane are massively
    # correlated (one plane, one surface), so resampling pixels would give an interval
    # tighter than the number of independent surfaces could ever support.
    for key, label in (('all', 'all regions'), ('in', 'inside the cone'), ('out', 'outside the cone')):
        if not pooled[key]:
            continue
        pr = np.concatenate([p for p, _ in pooled[key]])
        gz = np.concatenate([g for _, g in pooled[key]])
        ent = {'label': label, 'n_regions': len(pooled[key]), **metrics(pr, gz)}
        if len(pooled_groups[key]) >= 2:
            ent['medae_ci'] = frame_bootstrap(pooled_groups[key], np.median, B=2000)
            ent['ci_note'] = ('resampled over plane regions, not pixels -- pixels on one '
                              'plane are not independent observations')
        report['pooled'][key] = ent

    print()
    for key in ('all', 'in', 'out'):
        e = report['pooled'].get(key)
        if not e:
            continue
        ci = e.get('medae_ci')
        extra = f"  [{ci['lo']:.4f}, {ci['hi']:.4f}]" if ci else '  (interval needs >=2 regions)'
        print(f"  pooled {e['label']:<18} {e['n_px']:>8} px over {e['n_regions']} region(s)  "
              f"medAE {e['medae']:.4f}{extra}  AbsRel {e['absrel']:.4f}  d1 {e['d1']:.4f}")

    if 'out' not in report['pooled']:
        print('\n  NOTE: no region fell outside the cone. The out-of-coverage measurement is '
              '\n  the entire point of this tool -- place at least one plane target beyond '
              '\n  the cone boundary (above it is easiest; vertical FOV is the binding limit).')

    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()

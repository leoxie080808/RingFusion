#!/usr/bin/env python3
"""Trivial baselines: does the neural stack beat naive interpolation of the ToF?

Every depth number this project reports is scored against held-out ToF zones, with no
reference point -- a reader cannot tell whether 0.199 m is excellent or poor. This scores
six methods on the SAME anchors at the SAME held-out zones, from the zero-information
floor up to the full pipeline.

  B0 const     median of the anchor depths, everywhere        (0 params, no camera)
  B1 nearest   each held-out zone takes its nearest anchor     (0 params, no camera)
  B2 bilinear  linear interp over anchors in zone space        (0 params, no camera)
  B3 medscale  mono disparity x one global scale               (1 param)
  B4 affine    the deployed closed-form fit, Network B off     (2 params)
  B5 ringfusion   B4 + the residual refiner                    (~0.46M params)

TWO HOLD-OUT PROTOCOLS, and the difference between them is the point
--------------------------------------------------------------------
'random' is what training and every published number here have used: 25% of zones held
out at random. Measured over 300 real logs, 99.6% of those held-out zones have an anchor
within ONE zone -- a median of 1.7 cm away in world space. That is not depth estimation,
it is interpolating between adjacent samples, and B1/B2 will score well on it for reasons
that say nothing about deployment.

'center' mirrors the real geometry instead: anchor on a central 16x16 island, predict
everything outside it. The ToF covers 7.5% of the frame and the pipeline extrapolates
outward, so this is the same problem shape at smaller angles -- median 8.4 deg from the
nearest anchor, ~30 cm at 2 m depth. B2 cannot extrapolate outside its convex hull at all
and will show it as collapsed COVERAGE, not as bad MAE; that is a real result, so it is
reported rather than papered over with a nearest-neighbour fallback.

The headline output is not a single MAE but ERROR vs ANGULAR DISTANCE from the nearest
anchor -- the decay curve, which can be extrapolated toward the frame periphery and then
checked independently with tape.

Runs offline on the logged pairs; no robot motion needed, but the TensorRT engines mean
it must run on the Orin.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anchoring_bridge import build_residual_inputs, calib_from_yaml   # noqa: E402
from ringfusion_perception import geometry as geo                     # noqa: E402
from ringfusion_perception.blend import blend_depth                   # noqa: E402
import metrics as M                                                   # noqa: E402

MIN_RANGE, MAX_RANGE = 0.15, 6.5      # same gate as build_real_supervision
ANG_EDGES = [0.0, 3.0, 6.0, 10.0, 15.0, 30.0]
METHODS = ('B0_const', 'B1_nearest', 'B2_bilinear', 'B3_medscale',
           'B4_affine', 'B4c_affine_cl', 'B5_ringfusion', 'B6_blend')

# B4 and B5 do NOT share a far-field policy: anchoring.to_metric_depth clamps inverse
# depth at min_disp=1e-4, so B4 can emit 10,000 m, while ResidualRefiner.refine caps at
# MAX_DEPTH_M=20. Comparing their MEANS therefore compares clamp policy, not model
# quality -- it made B4 look like 18 m MAE against B5's 0.33 m. B4c applies B5's own cap
# to the closed-form output so the two are finally comparable; the B4-vs-B4c gap is the
# size of the artefact.
CLAMP_M = 20.0


def split_zones(valid, protocol, rng, frac=0.25, island=16):
    """-> (anchor_mask, holdout_mask), both (rows,cols) bool subsets of `valid`."""
    rows, cols = valid.shape
    if protocol == 'random':
        idx = np.flatnonzero(valid.ravel())
        rng.shuffle(idx)
        hold = np.zeros(valid.size, bool)
        hold[idx[:max(1, int(round(idx.size * frac)))]] = True
        hold = hold.reshape(valid.shape)
        return valid & ~hold, hold
    if protocol == 'center':
        rr, cc = np.mgrid[0:rows, 0:cols]
        isl = ((np.abs(rr - (rows - 1) / 2.0) < island / 2.0) &
               (np.abs(cc - (cols - 1) / 2.0) < island / 2.0))
        return valid & isl, valid & ~isl
    if protocol == 'insample':
        # NOT a hold-out: fits and scores on the SAME zones. This is exactly what
        # moving_ab.py and sigma_cal.py do on-robot (both splat the anchor set that
        # drove solve_robust, then score there), so it reproduces the published
        # 0.294 m / 0.199 m and quantifies how optimistic they are.
        return valid.copy(), valid.copy()
    raise ValueError(protocol)


def project(dist, mask, calib, h, w):
    """Project the zones in `mask` to pixels. -> (u, v, z, flat_zone_idx), in-bounds only."""
    rows, cols = dist.shape
    p = geo.project_zone_to_pixel(dist, mask, cols, rows, calib['fov_h'], calib['fov_v'],
                                  calib['T_cam_tof'], calib['K'], calib['dist'],
                                  model=calib['model'])
    uv, z, ok = p['uv'], p['z_cam'], p['valid']
    fin = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
    u = np.round(np.where(fin, uv[:, 0], -1)).astype(int)
    v = np.round(np.where(fin, uv[:, 1], -1)).astype(int)
    inb = ok & fin & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    idx = np.flatnonzero(inb)
    return u[idx], v[idx], z[idx], idx


def angular_coords(flat_idx, cols, pitch_h, pitch_v):
    r, c = np.divmod(flat_idx, cols)
    return np.stack([c * pitch_h, r * pitch_v], 1)      # degrees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-dir', required=True)
    ap.add_argument('--tof-dir', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--backbone-engine', required=True)
    ap.add_argument('--residual-engine', default='')
    ap.add_argument('--protocol', nargs='+', default=['random', 'center'])
    ap.add_argument('--island', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='0 = all frames')
    # B5 is a TRAINED net and these 1234 pairs are its training set, so scoring it on all
    # of them is contaminated. train_residual.py splits with
    # random_split(..., manual_seed(0)) at --val-frac 0.05, which is reproducible -- pass
    # the recovered val stems here to score B5 on frames it never saw. B0-B4 are
    # unaffected (no learned parameters; B3/B4 fit per frame).
    ap.add_argument('--stems-file', default='', help='newline-separated stems to restrict to')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    # A 2-param fit needs spread in disparity. The 'center' island concentrates anchors,
    # which can degrade conditioning -- if it does, B4/B5 would look bad for a reason
    # unrelated to extrapolation. Drop those frames and report how many.
    ap.add_argument('--max-cond', type=float, default=1e8)
    # Crossover between nearest-zone ToF and the network sits near 3 deg; ramp around it.
    ap.add_argument('--blend-near', type=float, default=2.0)
    ap.add_argument('--blend-far', type=float, default=5.0)
    a = ap.parse_args()

    import cv2
    from ringfusion_perception.backbone import TensorRTBackbone
    from ringfusion_perception import anchoring as anc

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a.tof_dir) if f.endswith('.npz'))
    if a.stems_file:
        keep = {s.strip() for s in open(a.stems_file) if s.strip()}
        stems = [s for s in stems if s in keep]
    if a.limit:
        stems = stems[:a.limit]
    if not stems:
        sys.exit(f'no .npz under {a.tof_dir}')

    first = cv2.imread(os.path.join(a.rgb_dir, stems[0] + '.png'))
    if first is None:
        sys.exit(f'cannot read {stems[0]}.png under {a.rgb_dir}')
    h, w = first.shape[:2]
    # Both engines take any input size and return at SOURCE resolution, so running at the
    # stored rectified size reproduces the deployed pipeline rather than training res.
    calib = calib_from_yaml(a.calib, train_size=(h, w))
    print(f'{len(stems)} frames at {w}x{h}  |  ToF fov {calib["fov_h"]}x{calib["fov_v"]} deg')

    backbone = TensorRTBackbone(a.backbone_engine)
    residual = None
    if a.residual_engine:
        from ringfusion_perception.residual import ResidualRefiner
        residual = ResidualRefiner(a.residual_engine)

    acc = {p: {m: {'pred': [], 'gt': [], 'ang': []} for m in METHODS} for p in a.protocol}
    skipped = {p: {'nosplit': 0, 'nofit': 0, 'illcond': 0} for p in a.protocol}
    t0 = time.time()

    for n, stem in enumerate(stems):
        img = cv2.imread(os.path.join(a.rgb_dir, stem + '.png'))
        if img is None:
            continue
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        d = np.load(os.path.join(a.tof_dir, stem + '.npz'))['dist_m'].astype(np.float32)
        rows, cols = d.shape
        ph, pv = calib['fov_h'] / cols, calib['fov_v'] / rows
        valid = np.isfinite(d) & (d >= MIN_RANGE) & (d <= MAX_RANGE)

        disp = backbone.infer(rgb)          # one backbone pass shared by every protocol

        for proto in a.protocol:
            rng = np.random.default_rng(a.seed + n)     # same split for every method
            am_z, hm_z = split_zones(valid, proto, rng, island=a.island)
            if am_z.sum() < 32 or hm_z.sum() < 16:
                skipped[proto]['nosplit'] += 1
                continue

            ua, va, za, ia = project(d, am_z, calib, h, w)
            uh, vh, zh, ih = project(d, hm_z, calib, h, w)
            if ua.size < 32 or uh.size < 16:
                skipped[proto]['nosplit'] += 1
                continue

            disp_a = disp[va, ua]
            X = np.stack([disp_a, np.ones_like(disp_a)], 1)
            if np.linalg.cond(X.T @ X) > a.max_cond:
                skipped[proto]['illcond'] += 1
                continue

            info = build_residual_inputs(disp, d, am_z, calib)
            if info is None:
                skipped[proto]['nofit'] += 1
                continue

            pa = angular_coords(ia, cols, ph, pv)
            phd = angular_coords(ih, cols, ph, pv)
            dmat = np.linalg.norm(phd[:, None, :] - pa[None, :, :], axis=2)
            ang = dmat.min(1)
            nearest = za[dmat.argmin(1)]

            pred = {
                'B0_const': np.full(zh.shape, float(np.median(za))),
                'B1_nearest': nearest,
                'B2_bilinear': _bilinear(pa, za, phd),
                'B3_medscale': _medscale(disp_a, za, disp[vh, uh]),
                'B4_affine': info['D0'][vh, uh],
                'B4c_affine_cl': np.clip(info['D0'][vh, uh], None, CLAMP_M),
            }
            D_net = info['D0']
            if residual is not None:
                D, _ = residual.refine(rgb, info['D0'], disp, info['anchor_depth'],
                                       info['anchor_mask'], info['a'], info['b'])
                pred['B5_ringfusion'] = D[vh, uh]
                D_net = D
            # B6: neither source wins everywhere (see blend.py) -- ToF near the anchors,
            # network far from them, smoothstep between.
            Kv = np.asarray(calib['K'], np.float64).ravel()
            D_bl, _ = blend_depth(D_net, info['anchor_depth'], info['anchor_mask'],
                                  fx=float(Kv[0]), near_deg=a.blend_near,
                                  far_deg=a.blend_far)
            pred['B6_blend'] = D_bl[vh, uh]

            for m, p in pred.items():
                acc[proto][m]['pred'].append(np.asarray(p, np.float64))
                acc[proto][m]['gt'].append(zh.astype(np.float64))
                acc[proto][m]['ang'].append(ang)

        if (n + 1) % 100 == 0:
            print(f'  {n+1}/{len(stems)}  ({(time.time()-t0)/(n+1)*1e3:.0f} ms/frame)', flush=True)

    report = {'frames': len(stems), 'size': [h, w], 'island': a.island, 'protocols': {}}
    for proto in a.protocol:
        rows_out = []
        print(f'\n{"="*len(M.HEADER)}\nPROTOCOL: {proto}'
              f'   (skipped: {skipped[proto]})\n{"="*len(M.HEADER)}')
        per_method = {}
        for m in METHODS:
            if not acc[proto][m]['pred']:
                continue
            p = np.concatenate(acc[proto][m]['pred'])
            g = np.concatenate(acc[proto][m]['gt'])
            x = np.concatenate(acc[proto][m]['ang'])
            mm = M.depth_metrics(p, g)
            rows_out.append((m, mm))
            per_method[m] = {'overall': mm, 'by_angle': [
                {'lo': lo, 'hi': hi, 'n': nn, **({} if r is None else r)}
                for lo, hi, nn, r in M.binned(p, g, x, ANG_EDGES)]}
        print(M.format_table(rows_out))

        # Bins holding a handful of points are noise, not signal -- under 'random' the
        # far bins hold n=228/9/1 -- so print n and blank anything under MIN_BIN_N.
        MIN_BIN_N = 500
        print(f'\n  median AE (m) by angular distance from nearest anchor'
              f'   [bins with n<{MIN_BIN_N} suppressed]:')
        hdr = ''.join(f'{lo:.0f}-{hi:.0f}deg'.rjust(11)
                      for lo, hi in zip(ANG_EDGES[:-1], ANG_EDGES[1:]))
        counts = ''.join(str(b['n']).rjust(11) for b in per_method[METHODS[0]]['by_angle'])
        print(f'  {"method":<14}{hdr}\n  {"n =":<14}{counts}')
        for m in METHODS:
            if m not in per_method:
                continue
            cells = ''
            for b in per_method[m]['by_angle']:
                v = b.get('medae', float('nan'))
                cells += (f'{v:.3f}' if (np.isfinite(v) and b['n'] >= MIN_BIN_N)
                          else '  -  ').rjust(11)
            print(f'  {m:<14}{cells}')
        report['protocols'][proto] = {'skipped': skipped[proto], 'methods': per_method}

    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'\nwrote {a.out}')


def _bilinear(anchor_ang, anchor_z, hold_ang):
    """Linear interp over anchors in ANGULAR zone space. NaN outside the convex hull --
    deliberately not backfilled, so the inability to extrapolate shows up as coverage."""
    from scipy.interpolate import griddata
    try:
        return griddata(anchor_ang, anchor_z, hold_ang, method='linear')
    except Exception:
        return np.full(hold_ang.shape[0], np.nan)


def _medscale(disp_a, z_a, disp_h):
    """Mono + ONE global scale, using the median scaling protocol standard in the
    monocular-depth literature: take the network's relative depth 1/disp and multiply by
    median(gt)/median(pred) over the anchors.

    Fitting a scale in the INVERSE domain instead (1/z = s*disp, forcing the shift to 0)
    is not equivalent and is not a fair baseline: the backbone's disparity carries an
    arbitrary offset, so dropping the shift term is catastrophic rather than merely naive
    -- it scored 26 m MAE, which measures the strawman and not the idea."""
    pa = 1.0 / np.clip(disp_a, 1e-6, None)
    s = float(np.median(z_a) / max(np.median(pa), 1e-9))
    if not np.isfinite(s) or s <= 0:
        return np.full(disp_h.shape, np.nan)
    return s / np.clip(disp_h, 1e-6, None)


if __name__ == '__main__':
    main()

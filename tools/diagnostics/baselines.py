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
from ringfusion_perception import roi                                  # noqa: E402
from bootstrap import frame_bootstrap, paired_diff                    # noqa: E402
from ringfusion_perception.blend import blend_depth, apply_scene_cap                   # noqa: E402
import metrics as M                                                   # noqa: E402
import envinfo                                                        # noqa: E402

MIN_RANGE, MAX_RANGE = 0.15, 6.5      # same gate as build_real_supervision
ANG_EDGES = [0.0, 3.0, 6.0, 10.0, 15.0, 30.0]
METHODS = ('B0_const', 'B1_nearest', 'B2_bilinear', 'B3_medscale',
           'W0_uniform_norobust', 'W1_rangep1', 'W2_rangep2', 'W3_roi', 'W0_uniform',
           'B4_affine', 'B4c_affine_cl')

# The W* rows exist because the paper's Method section and the deployed code disagree
# about the anchor weighting. Section IV-C says "we use w_i ~ z_i"; anchoring.py has
# RANGE_WEIGHT_P = 0.0 and pipeline.run folds in the geometric ROI gate from roi.py
# instead. All five run on the SAME anchors and the SAME held-out zones, differing only
# in w_i, so the column is a clean weighting ablation rather than five separate configs:
#
#   W0_uniform_norobust  w = 1, no Huber pass      <- the table's reference row
#   W1_rangep1           w = z      (what the paper claims)
#   W2_rangep2           w = z^2    (what error propagation predicts)
#   W3_roi               geometric ROI gate        (what actually ships)
#   W0_uniform           w = 1, one Huber pass     <- isolates the robust pass alone
#
# W0_uniform should reproduce B4c_affine_cl; they take different code paths to the same
# fit, so a gap between them is a bug and the report prints it as a consistency check.

# B4 and B5 do NOT share a far-field policy: anchoring.to_metric_depth clamps inverse
# depth at min_disp=1e-4, so B4 can emit 10,000 m, while ResidualRefiner.refine caps at
# MAX_DEPTH_M=20. Comparing their MEANS therefore compares clamp policy, not model
# quality -- it made B4 look like 18 m MAE against B5's 0.33 m. B4c applies B5's own cap
# to the closed-form output so the two are finally comparable; the B4-vs-B4c gap is the
# size of the artefact.
CLAMP_M = 20.0


def zone_step(dist, valid):
    """Per-zone |depth - median of its valid 8-neighbours|, in metres. 0 where isolated.

    A zone whose reading differs sharply from its neighbours is straddling a depth edge:
    the surface it sees is not the surface the zones around it see. That is exactly the
    case a nearest-anchor copy gets wrong and the one the 'random'/'center' protocols
    cannot expose, because both score at zone centres where the neighbourhood is smooth.
    """
    rows, cols = dist.shape
    step = np.zeros((rows, cols), np.float64)
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            r0, r1 = max(0, r - 1), min(rows, r + 2)
            c0, c1 = max(0, c - 1), min(cols, c + 2)
            nb = dist[r0:r1, c0:c1][valid[r0:r1, c0:c1]]
            nb = nb[nb != dist[r, c]] if nb.size > 1 else nb
            if nb.size:
                step[r, c] = abs(float(dist[r, c]) - float(np.median(nb)))
    return step


def split_zones(valid, protocol, rng, frac=0.25, island=16, dist=None, edge_thresh=0.15):
    """-> (anchor_mask, holdout_mask), both (rows,cols) bool subsets of `valid`."""
    rows, cols = valid.shape
    if protocol == 'edge':
        # Hold out the zones sitting ON a depth step, anchor on the smooth remainder.
        # Requires a real step (edge_thresh) rather than just taking the top quartile,
        # so a frame with no discontinuity is SKIPPED rather than contributing a
        # quartile of smooth zones relabelled as edges. That keeps the protocol
        # measuring what it claims to measure, at the cost of scoring fewer frames.
        if dist is None:
            raise ValueError("protocol 'edge' needs the depth map")
        step = zone_step(dist, valid)
        cand = valid & (step > edge_thresh)
        n_cand = int(cand.sum())
        if n_cand == 0:
            return np.zeros_like(valid), np.zeros_like(valid)
        cap = max(1, int(round(int(valid.sum()) * frac)))
        if n_cand > cap:                      # keep the sharpest steps
            flat = np.flatnonzero(cand.ravel())
            keep = flat[np.argsort(-step.ravel()[flat])[:cap]]
            cand = np.zeros(valid.size, bool)
            cand[keep] = True
            cand = cand.reshape(valid.shape)
        return valid & ~cand, cand
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


def parse_engines(specs):
    """['NAME=PATH' | 'PATH'] -> ordered {name: path}.

    More than one refiner can be scored in a SINGLE run because the reviewer's objection
    to Table V is that its rows came from different runs on different splits: the analytic
    output appears as 0.055 m in the table and 0.064 m in the scattered-hold-out paragraph.
    Loading every engine at once means all of them see the same frames, the same per-frame
    anchor/hold-out split and the same backbone pass, so the rows are directly comparable
    by construction rather than by hoping two runs matched.

    A bare path keeps the old single-engine behaviour and the old row names.
    """
    out = {}
    for spec in specs or []:
        if not spec:
            continue
        name, sep, path = spec.partition('=')
        if not sep:
            name, path = 'ringfusion', name
        if name in out:
            sys.exit(f'--residual-engine: duplicate name {name!r}')
        if not os.path.exists(path):
            sys.exit(f'--residual-engine: no such engine {path!r}')
        out[name] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-dir', required=True)
    ap.add_argument('--tof-dir', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--backbone-engine', required=True)
    # Repeatable. 'NAME=PATH' scores that engine as its own row pair; a bare PATH keeps
    # the legacy names. The supervision-geometry comparison the table needs is
    #   --residual-engine v7=... scattered=residual_v3_best_fp16.engine \
    #                            island=residual_v4_best_fp16.engine
    # with --deployed-engine v7, which puts the scattered and island rows beside the
    # deployed one on a single split.
    ap.add_argument('--residual-engine', nargs='*', default=[],
                    help="repeatable; 'NAME=PATH' or a bare PATH")
    # The deployed engine keeps the canonical B5_ringfusion / B6_blend row names so the
    # numbers already cited elsewhere stay findable; every other engine is suffixed.
    ap.add_argument('--deployed-engine', default='',
                    help='name of the engine that ships (default: the first one given)')
    ap.add_argument('--protocol', nargs='+', default=['random', 'center'],
                    choices=['random', 'center', 'edge', 'insample'])
    # 'edge' holds out zones straddling a depth step. 0.15 m is above the sensor's own
    # ~0.01 m tape-verified accuracy by more than an order of magnitude, so a zone over
    # this is a real surface change rather than measurement noise.
    ap.add_argument('--edge-thresh', type=float, default=0.15,
                    help='metres of local depth step that defines an edge zone')
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
    # Frame-level, not pixel-level: errors inside a frame are strongly correlated, so
    # resampling pixels would return an interval far too tight to be honest. 0 = skip.
    ap.add_argument('--bootstrap', type=int, default=1000,
                    help='bootstrap resamples over FRAMES for the overall metrics')
    # A 2-param fit needs spread in disparity. The 'center' island concentrates anchors,
    # which can degrade conditioning -- if it does, B4/B5 would look bad for a reason
    # unrelated to extrapolation. Drop those frames and report how many.
    ap.add_argument('--max-cond', type=float, default=1e8)
    # Crossover between nearest-zone ToF and the network sits near 3 deg; ramp around it.
    ap.add_argument('--blend-near', type=float, default=2.0)
    ap.add_argument('--blend-far', type=float, default=5.0)
    # Scene-bounded far-field cap (blend.scene_cap). Sweep k HERE, on our own logs -- never
    # on a test split we then report. Prediction to falsify: RMSE should fall sharply while
    # Rel and delta1 barely move, because the cap should only touch extreme pixels. If
    # delta1 drops materially, k is too tight and it is mangling normal pixels.
    ap.add_argument('--sweep-k', type=float, nargs='*', default=[],
                    help='e.g. --sweep-k 1.25 1.5 2 3 4')
    # Two rows scored on the SAME frames are paired data, and overlapping marginal CIs are
    # not the test for them -- frame-to-frame difficulty is common to both arms, so the
    # marginal intervals are inflated by variation the comparison should difference out.
    # Each pair here is re-tested with one shared frame draw applied to both arms.
    ap.add_argument('--paired', nargs='*', default=[],
                    help="'A:B' row pairs, e.g. B5_scattered:B5_island")
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

    eng_paths = parse_engines(a.residual_engine)
    deployed = a.deployed_engine or (next(iter(eng_paths)) if eng_paths else '')
    if deployed and deployed not in eng_paths:
        sys.exit(f'--deployed-engine {deployed!r} is not one of {list(eng_paths)}')

    def b5(name):
        return 'B5_ringfusion' if name == deployed else f'B5_{name}'

    def b6(name):
        return 'B6_blend' if name == deployed else f'B6_{name}'

    refiners = {}
    if eng_paths:
        from ringfusion_perception.residual import ResidualRefiner
        for nm, pth in eng_paths.items():
            print(f'  refiner {nm:<12} {os.path.basename(pth)}'
                  f'{"   <- deployed" if nm == deployed else ""}')
            refiners[nm] = ResidualRefiner(pth)

    methods = list(METHODS) + ['B6_analytic']
    for nm in refiners:
        methods += [b5(nm), b6(nm)]
    methods += [f'B4k{k:g}' for k in a.sweep_k]
    acc = {p: {m: {'pred': [], 'gt': [], 'ang': []} for m in methods} for p in a.protocol}
    skipped = {p: {'nosplit': 0, 'nofit': 0, 'illcond': 0} for p in a.protocol}
    # The robust pass DOWNWEIGHTS rather than rejects, so both numbers are kept: the
    # fraction past the Huber threshold and the weight mass actually removed.
    robust = {p: {'downweighted': [], 'mass_removed': [], 'n_anchors': []} for p in a.protocol}
    no_plane = {p: 0 for p in a.protocol}
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
            am_z, hm_z = split_zones(valid, proto, rng, island=a.island,
                                     dist=d, edge_thresh=a.edge_thresh)
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

            # --- weighting ablation: same anchors, same targets, different w_i --------
            inv_at = 1.0 / za
            disp_h = disp[vh, uh]
            ones = np.ones_like(inv_at)
            variants = {
                'W0_uniform_norobust': (ones, 0),
                'W0_uniform': (ones, 1),
                'W1_rangep1': (anc.range_weights(inv_at, ones, p=1.0), 1),
                'W2_rangep2': (anc.range_weights(inv_at, ones, p=2.0), 1),
            }
            # The deployed gate is geometric, so it needs 3D points and a ground plane --
            # the same construction pipeline.run does at stage 4b. A frame whose floor
            # is not visible gets no plane and simply contributes no W3 row, rather than
            # silently falling back to uniform and diluting the comparison.
            pts_a = roi.backproject(ua, va, za, calib['K'])
            plane = roi.fit_ground_plane(pts_a)
            if plane is not None:
                variants['W3_roi'] = (roi.roi_weights(pts_a, plane, ones), 1)
            else:
                no_plane[proto] += 1

            for wname, (wts, it) in variants.items():
                rinfo = {} if it else None
                wfit = anc.solve_robust(disp_a, inv_at, wts, iters=it, info=rinfo)
                if wfit is None:
                    continue
                aw, bw = wfit
                pred[wname] = np.clip(anc.to_metric_depth(disp_h, aw, bw), None, CLAMP_M)
                if rinfo and wname == 'W0_uniform':
                    robust[proto]['downweighted'].append(rinfo['downweighted_frac'])
                    robust[proto]['mass_removed'].append(rinfo['weight_mass_removed_frac'])
                    robust[proto]['n_anchors'].append(rinfo['n_anchors'])
            # B6: neither source wins everywhere (see blend.py) -- ToF near the anchors,
            # network far from them, smoothstep between.
            Kv = np.asarray(calib['K'], np.float64).ravel()

            def _blend(D_src):
                D_bl, _ = blend_depth(D_src, info['anchor_depth'], info['anchor_mask'],
                                      fx=float(Kv[0]), near_deg=a.blend_near,
                                      far_deg=a.blend_far)
                return D_bl[vh, uh]

            # Arbitration over the ANALYTIC map, no refiner in the path. This is the
            # topology Table VI actually printed while being labelled as deployed, so it
            # gets its own row: the deployed row below blends over the refiner instead,
            # and the two can no longer be mistaken for one another.
            # Stage 7b (far-field clamp) runs BEFORE 7c (blend) in pipeline.py, so every
            # arbitration row blends over CLAMPED depth. Blending the raw D0 instead lets
            # the unclamped far-field artefact through and destroys MAE (10.9 m vs 0.15 m
            # on 'center') while barely moving the median -- exactly the medAE/MAE
            # signature that caught the missing clamp originally.
            pred['B6_analytic'] = _blend(np.clip(info['D0'], None, CLAMP_M))
            for nm, ref in refiners.items():
                D, _ = ref.refine(rgb, info['D0'], disp, info['anchor_depth'],
                                  info['anchor_mask'], info['a'], info['b'])
                D = np.clip(D, None, CLAMP_M)
                pred[b5(nm)] = D[vh, uh]
                pred[b6(nm)] = _blend(D)
            # Scene-bounded cap applied to the closed-form output, one row per k.
            for kk in a.sweep_k:
                Dk, _, _ = apply_scene_cap(info['D0'], info['anchor_depth'],
                                           info['anchor_mask'], k=kk)
                pred[f'B4k{kk:g}'] = Dk[vh, uh]

            for m, p in pred.items():
                acc[proto][m]['pred'].append(np.asarray(p, np.float64))
                acc[proto][m]['gt'].append(zh.astype(np.float64))
                acc[proto][m]['ang'].append(ang)

        if (n + 1) % 100 == 0:
            print(f'  {n+1}/{len(stems)}  ({(time.time()-t0)/(n+1)*1e3:.0f} ms/frame)', flush=True)

    report = {
        'label': 'r31 component ablation -- one run, one split, every row',
        'env': envinfo.capture([a.backbone_engine] + list(eng_paths.values()),
                               note='baselines'),
        'frames': len(stems), 'size': [h, w], 'island': a.island,
        'config': {
            'rgb_dir': os.path.abspath(a.rgb_dir), 'tof_dir': os.path.abspath(a.tof_dir),
            'calib': os.path.abspath(a.calib),
            'stems_file': os.path.abspath(a.stems_file) if a.stems_file else '',
            'protocols': list(a.protocol), 'seed': a.seed, 'bootstrap': a.bootstrap,
            'edge_thresh': a.edge_thresh, 'island': a.island, 'max_cond': a.max_cond,
            'blend_deg': [a.blend_near, a.blend_far], 'clamp_m': CLAMP_M,
            'range_gate_m': [MIN_RANGE, MAX_RANGE],
        },
        'refiners': {nm: {'path': os.path.abspath(pth), 'b5_row': b5(nm),
                          'b6_row': b6(nm), 'deployed': nm == deployed}
                     for nm, pth in eng_paths.items()},
        'one_split_note': (
            'Every row comes from ONE run over ONE split: the same frames, the same '
            'per-frame anchor/hold-out zone split, and one shared backbone pass, with all '
            'refiners resident at once. Rows are therefore comparable to each other '
            'directly. B6_analytic is arbitration with NO refiner; the deployed row '
            'blends over the refiner named in "refiners".'),
        'protocols': {},
    }
    for proto in a.protocol:
        rows_out = []
        print(f'\n{"="*len(M.HEADER)}\nPROTOCOL: {proto}'
              f'   (skipped: {skipped[proto]})\n{"="*len(M.HEADER)}')
        per_method = {}
        for m in methods:
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
            if a.bootstrap:
                # One group per frame. NaNs come from B2 outside its convex hull, where
                # the method genuinely has no prediction -- dropping them keeps the
                # interval consistent with the point estimate instead of propagating NaN.
                grp = []
                for pp, gg in zip(acc[proto][m]['pred'], acc[proto][m]['gt']):
                    e = np.abs(np.asarray(pp) - np.asarray(gg))
                    e = e[np.isfinite(e)]
                    if e.size:
                        grp.append(e)
                per_method[m]['ci'] = {
                    'medae': frame_bootstrap(grp, np.median, B=a.bootstrap, seed=a.seed),
                    'mae': frame_bootstrap(grp, np.mean, B=a.bootstrap, seed=a.seed)}
        print(M.format_table(rows_out))

        # Bins holding a handful of points are noise, not signal -- under 'random' the
        # far bins hold n=228/9/1 -- so print n and blank anything under MIN_BIN_N.
        MIN_BIN_N = 500
        print(f'\n  median AE (m) by angular distance from nearest anchor'
              f'   [bins with n<{MIN_BIN_N} suppressed]:')
        hdr = ''.join(f'{lo:.0f}-{hi:.0f}deg'.rjust(11)
                      for lo, hi in zip(ANG_EDGES[:-1], ANG_EDGES[1:]))
        counts = ''.join(str(b['n']).rjust(11) for b in per_method[methods[0]]['by_angle'])
        print(f'  {"method":<14}{hdr}\n  {"n =":<14}{counts}')
        for m in methods:
            if m not in per_method:
                continue
            cells = ''
            for b in per_method[m]['by_angle']:
                v = b.get('medae', float('nan'))
                cells += (f'{v:.3f}' if (np.isfinite(v) and b['n'] >= MIN_BIN_N)
                          else '  -  ').rjust(11)
            print(f'  {m:<14}{cells}')
        # --- robust pass: what it actually does to the anchors ----------------------
        rb = robust[proto]
        rb_summary = None
        if rb['downweighted']:
            rb_summary = {
                'frames': len(rb['downweighted']),
                'median_anchors': float(np.median(rb['n_anchors'])),
                'downweighted_frac_median': float(np.median(rb['downweighted'])),
                'downweighted_frac_mean': float(np.mean(rb['downweighted'])),
                'weight_mass_removed_median': float(np.median(rb['mass_removed'])),
                'note': ('Huber DOWNWEIGHTS, it does not reject. downweighted_frac is '
                         'the share of anchors past the threshold; weight_mass_removed '
                         'is how much total weight the pass took out. The paper says '
                         '"removes X% of anchors", which names neither.')}
            print('\n  robust pass: %.1f%% of anchors downweighted (median over %d '
                  'frames, median %.0f anchors), removing %.1f%% of weight mass'
                  % (rb_summary['downweighted_frac_median'] * 100, rb_summary['frames'],
                     rb_summary['median_anchors'],
                     rb_summary['weight_mass_removed_median'] * 100))
        if no_plane[proto]:
            print('  no ground plane on %d frames -- W3_roi omitted there'
                  % no_plane[proto])

        # W0_uniform and B4c_affine_cl are the same fit reached by two code paths, so a
        # gap between them is a bug in one of them rather than a result.
        if 'W0_uniform' in per_method and 'B4c_affine_cl' in per_method:
            d0 = per_method['W0_uniform']['overall'].get('medae', float('nan'))
            d1 = per_method['B4c_affine_cl']['overall'].get('medae', float('nan'))
            gap = abs(d0 - d1)
            print('  consistency check  W0_uniform %.4f vs B4c_affine_cl %.4f   gap %.4f m%s'
                  % (d0, d1, gap, '  <-- SHOULD BE ~0' if gap > 5e-3 else '  OK'))

        # --- paired A/B on a shared frame draw ---------------------------------------
        paired_out = {}
        for spec in a.paired:
            ra, sep, rb = spec.partition(':')
            if not sep or ra not in acc[proto] or rb not in acc[proto]:
                print(f'  paired {spec}: unknown row, skipped'); continue

            def _err(row):
                out = []
                for pp, gg in zip(acc[proto][row]['pred'], acc[proto][row]['gt']):
                    e = np.abs(np.asarray(pp, float) - np.asarray(gg, float))
                    out.append(e[np.isfinite(e)])
                return out

            ga, gb = _err(ra), _err(rb)
            # A row absent on some frames (W3_roi with no ground plane) would silently
            # pair frame i of one arm with frame j of the other, so refuse instead.
            if len(ga) != len(gb):
                print(f'  paired {spec}: {len(ga)} vs {len(gb)} frames, not paired -- skipped')
                paired_out[spec] = {'error': f'frame counts differ: {len(ga)} vs {len(gb)}'}
                continue
            pm = paired_diff(ga, gb, np.median, B=a.bootstrap or 1000, seed=a.seed)
            pa_ = paired_diff(ga, gb, np.mean, B=a.bootstrap or 1000, seed=a.seed)
            paired_out[spec] = {'medae': pm, 'mae': pa_}
            print('  paired  %-34s medAE %+.4f m  [%+.4f, %+.4f]  p=%.3f  (n=%d frames)'
                  % (f'{rb} - {ra}', pm['point'], pm['lo'], pm['hi'],
                     pm['p_two_sided'], pm['n_groups']))

        report['protocols'][proto] = {'skipped': skipped[proto], 'methods': per_method,
                                      'robust_pass': rb_summary,
                                      'paired': paired_out,
                                      'frames_without_ground_plane': no_plane[proto]}

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

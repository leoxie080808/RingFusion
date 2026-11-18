#!/usr/bin/env python3
"""One held-out frame as a picture, for the ICRAE accuracy slide.

Rebuilds the deployed pipeline on a logged frame exactly as baselines.py scores it
(withheld-zone protocol, deployed backbone + refiner + arbitration), then draws:
  camera     the frame, with the KEPT zones as dots and the withheld ones as rings
  nearest    dense "copy the nearest kept zone" map (the B1 baseline, as an image)
  ours       the deployed published depth (refiner + arbitration, clamped)
  err_*      |error| at every withheld zone, as coloured discs on a dimmed camera frame,
             for nearest-zone and for ours, on ONE shared colour scale
Per-frame median errors at the withheld zones are written to summary.json.

Frame choice (--pick, comma separated), from the stems in --stems-file:
  median  the frame whose deployed median error is closest to the split's median
  gap     the frame where ours beats nearest-zone by the most (median error)
Or pass --stem to render one frame by name.

Defaults point at the paper's held-out split, engines and data in the repo checkout.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_perception'))
sys.path.insert(0, _HERE)

from baselines import split_zones, project, angular_coords, MIN_RANGE, MAX_RANGE, CLAMP_M   # noqa: E402
from anchoring_bridge import build_residual_inputs, calib_from_yaml                          # noqa: E402
from ringfusion_perception.blend import blend_depth                                          # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def colorize(d, valid, lo, hi, cmap=cv2.COLORMAP_TURBO):
    n = np.clip((np.nan_to_num(d) - lo) / max(1e-6, hi - lo), 0, 1)
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    c[~valid] = (18, 18, 18)
    return c


def caption(img, txt, sub=''):
    img = img.copy()
    s = img.shape[1] / 640.0
    bar = int(30 * s)
    cv2.rectangle(img, (0, 0), (img.shape[1], bar), (0, 0, 0), -1)
    cv2.putText(img, txt, (int(10 * s), int(21 * s)), FONT, 0.62 * s, (255, 255, 255), max(1, int(s)), cv2.LINE_AA)
    if sub:
        cv2.rectangle(img, (0, img.shape[0] - bar), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, sub, (int(10 * s), img.shape[0] - int(9 * s)), FONT, 0.55 * s,
                    (90, 230, 255), max(1, int(s)), cv2.LINE_AA)
    return img


def nearest_fill(u, v, z, h, w):
    from scipy.interpolate import griddata
    gy, gx = np.mgrid[0:h, 0:w]
    return griddata(np.stack([v, u], 1).astype(np.float64), z, (gy, gx), method='nearest').astype(np.float32)


def error_discs(base_bgr, uh, vh, err, emax, radius):
    out = (base_bgr * 0.35).astype(np.uint8)
    n = np.clip(err / emax, 0, 1)
    cols = cv2.applyColorMap((n * 255).astype(np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0]
    for x, y, c in zip(uh, vh, cols):
        cv2.circle(out, (int(x), int(y)), radius, tuple(int(k) for k in c), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), radius, (170, 170, 170), 1, cv2.LINE_AA)   # keeps ~0 error visible
    return out


def run_frame(stem, a, calib, backbone, refiner):
    img = cv2.imread(os.path.join(a.rgb_dir, stem + '.png'))
    if img is None:
        return None
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = img.shape[:2]
    d = np.load(os.path.join(a.tof_dir, stem + '.npz'))['dist_m'].astype(np.float32)
    rows, cols = d.shape
    ph, pv = calib['fov_h'] / cols, calib['fov_v'] / rows
    valid = np.isfinite(d) & (d >= MIN_RANGE) & (d <= MAX_RANGE)
    disp = backbone.infer(rgb)
    rng = np.random.default_rng(a.seed + a.stem_index.get(stem, 0))   # same split as baselines.py
    am_z, hm_z = split_zones(valid, a.protocol, rng, island=a.island)
    if am_z.sum() < 32 or hm_z.sum() < 16:
        return None
    ua, va, za, ia = project(d, am_z, calib, h, w)
    uh, vh, zh, ih = project(d, hm_z, calib, h, w)
    if ua.size < 32 or uh.size < 16:
        return None
    info = build_residual_inputs(disp, d, am_z, calib)
    if info is None:
        return None
    D, _ = refiner.refine(rgb, info['D0'], disp, info['anchor_depth'], info['anchor_mask'],
                          info['a'], info['b'])
    D = np.clip(D, None, CLAMP_M)
    Kv = np.asarray(calib['K'], np.float64).ravel()
    D_pub, _ = blend_depth(D, info['anchor_depth'], info['anchor_mask'], fx=float(Kv[0]),
                           near_deg=a.blend_near, far_deg=a.blend_far)
    pa = angular_coords(ia, cols, ph, pv)
    phd = angular_coords(ih, cols, ph, pv)
    near_z = za[np.linalg.norm(phd[:, None, :] - pa[None, :, :], axis=2).argmin(1)]   # B1, angular
    e_near = np.abs(near_z - zh)
    e_ours = np.abs(D_pub[vh, uh] - zh)
    return dict(stem=stem, rgb=rgb, bgr=img, D_pub=D_pub, ua=ua, va=va, za=za, uh=uh, vh=vh, zh=zh,
                e_near=e_near, e_ours=e_ours, med_near=float(np.median(e_near)),
                med_ours=float(np.median(e_ours)), n_kept=int(ua.size), n_held=int(uh.size))


def summary_panel(shape, r, a):
    """Bottom-left cell of the sheet: counts, the two medians, and the error colour scale."""
    h, w = shape[:2]
    out = np.zeros(shape, np.uint8)
    s = w / 640.0
    def put(txt, y, size=0.62, col=(255, 255, 255)):
        cv2.putText(out, txt, (int(24 * s), int(y * s)), FONT, size * s, col, max(1, int(s)), cv2.LINE_AA)
    put(f'frame {r["stem"]}', 52, 0.55, (170, 170, 170))
    put(f'{r["n_kept"]} zones kept (input)', 96)
    put(f'{r["n_held"]} zones withheld (scored)', 132)
    put('median |error| on withheld zones', 190, 0.55, (170, 170, 170))
    put(f'copy nearest   {100 * r["med_near"]:5.1f} cm', 230, 0.7)
    put(f'ours           {100 * r["med_ours"]:5.1f} cm', 270, 0.7, (90, 230, 255))
    x0, x1, y0 = int(24 * s), w - int(24 * s), int(330 * s)
    ramp = cv2.applyColorMap(np.linspace(0, 255, x1 - x0).astype(np.uint8)[None, :], cv2.COLORMAP_INFERNO)
    out[y0:y0 + int(16 * s), x0:x1] = np.repeat(ramp, int(16 * s), axis=0)
    cv2.putText(out, '0', (x0, y0 + int(40 * s)), FONT, 0.5 * s, (200, 200, 200), max(1, int(s)), cv2.LINE_AA)
    cv2.putText(out, f'{a.emax:g} m', (x1 - int(40 * s), y0 + int(40 * s)), FONT, 0.5 * s, (200, 200, 200),
                max(1, int(s)), cv2.LINE_AA)
    return out


def render(r, tag, a):
    h, w = r['bgr'].shape[:2]
    rad = max(4, int(round(w / 160)))
    cam = r['bgr'].copy()
    for x, y in zip(r['ua'], r['va']):
        cv2.circle(cam, (int(x), int(y)), max(2, rad // 2), (255, 255, 255), -1, cv2.LINE_AA)
    for x, y in zip(r['uh'], r['vh']):
        cv2.circle(cam, (int(x), int(y)), max(2, rad // 2), (200, 200, 200), 1, cv2.LINE_AA)
    near = nearest_fill(r['ua'], r['va'], r['za'], h, w)
    imgs = {
        'camera': cam,
        'nearest': colorize(near, np.isfinite(near), a.d_lo, a.d_hi),
        'ours': colorize(r['D_pub'], np.isfinite(r['D_pub']) & (r['D_pub'] > 0), a.d_lo, a.d_hi),
        'err_nearest': error_discs(r['bgr'], r['uh'], r['vh'], r['e_near'], a.emax, rad),
        'err_ours': error_discs(r['bgr'], r['uh'], r['vh'], r['e_ours'], a.emax, rad),
    }
    stem = os.path.join(a.out, f'{tag}_{r["stem"]}')
    for k, v in imgs.items():
        cv2.imwrite(f'{stem}_{k}.png', v)
    S = (640, int(640 * h / w))
    sm = {k: cv2.resize(v, S, interpolation=cv2.INTER_AREA) for k, v in imgs.items()}
    top = np.hstack([caption(sm['camera'], 'Camera (dots kept, rings withheld)'),
                     caption(sm['nearest'], f'Copy nearest kept zone ({a.d_lo:g}-{a.d_hi:g} m)'),
                     caption(sm['ours'], 'Ours, as published')])
    bot = np.hstack([summary_panel(sm['camera'].shape, r, a),
                     caption(sm['err_nearest'], f'|error|, nearest (0-{a.emax:g} m)',
                             f'median {100 * r["med_near"]:.1f} cm'),
                     caption(sm['err_ours'], f'|error|, ours (0-{a.emax:g} m)',
                             f'median {100 * r["med_ours"]:.1f} cm')])
    cv2.imwrite(f'{stem}_sheet.png', np.vstack([top, bot]))
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-dir', default=os.path.join(_REPO, 'ros2_ws', 'data', 'real', 'rgb'))
    ap.add_argument('--tof-dir', default=os.path.join(_REPO, 'ros2_ws', 'data', 'real', 'tof'))
    ap.add_argument('--calib', default=os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_bringup',
                                                    'config', 'calibration.yaml'))
    ap.add_argument('--backbone-engine', default=os.path.join(_REPO, 'student_v4_heldout_fp16.engine'))
    ap.add_argument('--residual-engine', default=os.path.join(_REPO, 'residual_v7_fov73_fp16.engine'))
    ap.add_argument('--stems-file', default=os.path.join(_REPO, 'docs', 'demo', 'benchmarks', 'val_stems.txt'))
    ap.add_argument('--protocol', default='center', choices=['center', 'random'])
    ap.add_argument('--island', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--blend-near', type=float, default=2.0)
    ap.add_argument('--blend-far', type=float, default=5.0)
    ap.add_argument('--pick', default='median,gap')
    ap.add_argument('--stem', default='', help='render this one frame instead of --pick')
    ap.add_argument('--d-lo', type=float, default=0.15)
    ap.add_argument('--d-hi', type=float, default=5.0)
    ap.add_argument('--emax', type=float, default=0.5, help='error colour scale top (m)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from ringfusion_perception.backbone import TensorRTBackbone
    from ringfusion_perception.residual import ResidualRefiner

    # baselines.py seeds each frame's split with seed + its position in the SORTED stem list
    # (after the stems-file filter), so rebuild that order to reproduce its exact split.
    all_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a.tof_dir) if f.endswith('.npz'))
    keep = {s.strip() for s in open(a.stems_file) if s.strip()} if a.stems_file else set(all_stems)
    stems = [s for s in all_stems if s in keep]
    a.stem_index = {s: i for i, s in enumerate(stems)}
    if not stems:
        sys.exit('no stems found')

    first = cv2.imread(os.path.join(a.rgb_dir, stems[0] + '.png'))
    h, w = first.shape[:2]
    calib = calib_from_yaml(a.calib, train_size=(h, w))
    backbone = TensorRTBackbone(a.backbone_engine)
    refiner = ResidualRefiner(a.residual_engine)

    todo = [a.stem] if a.stem else stems
    results = []
    for s in todo:
        r = run_frame(s, a, calib, backbone, refiner)
        if r is not None:
            results.append(r)
    if not results:
        sys.exit('no frame produced a valid split')
    med_ours = np.array([r['med_ours'] for r in results])
    med_near = np.array([r['med_near'] for r in results])
    print(f'{len(results)} frames, {a.protocol} protocol. Median of per-frame medians: '
          f'ours {100 * np.median(med_ours):.1f} cm, nearest {100 * np.median(med_near):.1f} cm')

    if a.stem:
        chosen = [('frame', 0)]
    else:
        chosen, used = [], set()
        for p in [x.strip() for x in a.pick.split(',') if x.strip()]:
            if p == 'median':
                order = np.argsort(np.abs(med_ours - np.median(med_ours)))
            elif p == 'gap':
                order = np.argsort(-(med_near - med_ours))
            else:
                sys.exit(f'unknown pick {p!r}; use median, gap')
            i = next(int(j) for j in order if int(j) not in used)
            used.add(i)
            chosen.append((p, i))

    summary = {'protocol': a.protocol, 'frames_scored': len(results),
               'median_of_frame_medians_m': {'ours': float(np.median(med_ours)),
                                             'nearest': float(np.median(med_near))},
               'picks': []}
    for tag, i in chosen:
        r = results[i]
        stem = render(r, tag, a)
        print(f'{tag:6s} {r["stem"]}  kept {r["n_kept"]}  withheld {r["n_held"]}  '
              f'median error: nearest {100 * r["med_near"]:.1f} cm, ours {100 * r["med_ours"]:.1f} cm')
        summary['picks'].append({'tag': tag, 'stem': r['stem'], 'file': os.path.basename(stem),
                                 'kept': r['n_kept'], 'withheld': r['n_held'],
                                 'median_err_nearest_m': r['med_near'], 'median_err_ours_m': r['med_ours']})
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'wrote {len(chosen)} frames and summary.json to {a.out}')


if __name__ == '__main__':
    main()

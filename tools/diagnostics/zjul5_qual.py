#!/usr/bin/env python3
"""ZJU-L5 side-by-side for the ICRAE talk: our closed-form fit vs DEPTHOR-Small, as images.

Runs both methods on the ZJU-L5 test split exactly the way the paper's numbers were made:
  * ours   = zjul5_eval.py's B4c_affine_cl path: Depth Anything V2 ViT-S teacher, zone
             anchors, anc.solve_robust(..., iters=1) with unit weights, clipped at
             MAX_DEPTH_M (the "Ours, affine, ViT-S" row, AbsRel 0.094)
  * DEPTHOR-Small = depthor_regions.py's path: their released weights and their loader
Then picks three test images and saves, for each, the RGB (footprint outlined), ground
truth, both predictions and both absolute-error maps on SHARED colour scales.

Picks (--pick, comma separated):
  match   an image where we are at or below our median AbsRel and closest to DEPTHOR's
  median  the image whose AbsRel (ours) is the median of the split
  far     the image with the largest share of ground truth beyond --far-m (shows the ceiling)
Or pass --indices to choose test-split indices yourself.

Writes to --out: <tag>_<idx>_{rgb,gt,ours,depthor,err_ours,err_depthor,sheet}.png,
<tag>_<idx>.npz (all arrays, for re-rendering), and summary.json (per-image metrics plus
pooled AbsRel over everything scanned, as a check against the paper: 0.094 and 0.079).

Needs the DEPTHOR checkout (--depthor-root) and the ZJU-L5 folder that holds ZJUL5/data.json.
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

from ringfusion_perception import anchoring as anc                      # noqa: E402
from ringfusion_perception.residual import MAX_DEPTH_M                  # noqa: E402
from zjul5_eval import zone_anchors, coverage_mask, GT_MIN, GT_MAX, _Teacher   # noqa: E402
import metrics as M                                                     # noqa: E402


# ---------------------------------------------------------------- rendering
def depth_rgb(d, valid, vmax, vmin=0.1):
    n = np.clip((np.nan_to_num(d) - vmin) / (vmax - vmin), 0, 1)
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[~valid] = (0, 0, 0)
    return c


def err_rgb(e, valid, emax):
    n = np.clip(np.nan_to_num(e) / emax, 0, 1)
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    c[~valid] = (0, 0, 0)
    return c


def outline(rgb_bgr, cm, color=(255, 255, 255)):
    out = rgb_bgr.copy()
    cnts, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
    return out


def caption(img, txt, sub=''):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.rectangle(img, (0, img.shape[0] - 28), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, sub, (10, img.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (90, 230, 255), 1, cv2.LINE_AA)
    return img


def render(out_dir, tag, idx, name, rgb, gt, ours, dep, cm, m_ours, m_dep, vmax, emax):
    gv = (gt > GT_MIN) & (gt < GT_MAX)
    bgr = rgb[:, :, ::-1].copy()
    imgs = {
        'rgb': outline(bgr, cm),
        'gt': depth_rgb(gt, gv, vmax),
        'ours': depth_rgb(ours, np.isfinite(ours) & (ours > 0), vmax),
        'depthor': depth_rgb(dep, np.isfinite(dep) & (dep > 0), vmax),
        'err_ours': err_rgb(np.abs(ours - gt), gv, emax),
        'err_depthor': err_rgb(np.abs(dep - gt), gv, emax),
    }
    stem = os.path.join(out_dir, f'{tag}_{idx:03d}')
    for k, v in imgs.items():
        cv2.imwrite(f'{stem}_{k}.png', v)
    sheet_top = np.hstack([caption(imgs['rgb'], 'RGB (white = dToF footprint)'),
                           caption(imgs['ours'], 'Ours (closed-form fit)', f'AbsRel {m_ours["absrel"]:.3f}'),
                           caption(imgs['err_ours'], f'|error|, ours (0-{emax:g} m)')])
    sheet_bot = np.hstack([caption(imgs['gt'], f'Ground truth (0-{vmax:g} m)'),
                           caption(imgs['depthor'], 'DEPTHOR-Small', f'AbsRel {m_dep["absrel"]:.3f}'),
                           caption(imgs['err_depthor'], f'|error|, DEPTHOR (0-{emax:g} m)')])
    cv2.imwrite(f'{stem}_sheet.png', np.vstack([sheet_top, sheet_bot]))
    np.savez_compressed(f'{stem}.npz', rgb=rgb, gt=gt, ours=ours.astype(np.float32),
                        depthor=dep.astype(np.float32), footprint=cm, name=name)
    return stem


# ---------------------------------------------------------------- picking
def pick_indices(stats, picks, far_m):
    ours = np.array([s['all_ours']['absrel'] for s in stats])
    dep = np.array([s['all_depthor']['absrel'] for s in stats])
    far = np.array([s['far_frac'] for s in stats])
    med = np.median(ours)
    chosen, used = [], set()
    for p in picks:
        if p == 'median':
            order = np.argsort(np.abs(ours - med))
        elif p == 'match':
            cand = np.where(ours <= med)[0]
            order = cand[np.argsort(np.abs(ours[cand] - dep[cand]))]
        elif p == 'far':
            order = np.lexsort((-(ours - dep), -far))      # most far GT, then largest gap
        else:
            sys.exit(f'unknown pick {p!r}; use match, median, far')
        i = next(int(j) for j in order if int(j) not in used)
        used.add(i)
        chosen.append((p, i))
    return chosen


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zjul5-root', required=True, help='dir containing ZJUL5/ and its data.json')
    ap.add_argument('--depthor-root', default=os.path.expanduser('~/external/Depthor'))
    ap.add_argument('--config', default='configs/test_zju_small.txt')
    ap.add_argument('--teacher', default='depth-anything/Depth-Anything-V2-Small-hf',
                    help='the ViT-S backbone behind the paper row (same weights as DEPTHOR-Small)')
    ap.add_argument('--pick', default='match,median,far')
    ap.add_argument('--indices', type=int, nargs='*', default=[],
                    help='test-split indices to render instead of --pick')
    ap.add_argument('--limit', type=int, default=0, help='scan only the first N test images (0 = all)')
    ap.add_argument('--far-m', type=float, default=5.0)
    ap.add_argument('--vmax', type=float, default=10.0, help='depth colour scale top (m)')
    ap.add_argument('--emax', type=float, default=2.0, help='error colour scale top (m)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    a.zjul5_root = os.path.abspath(a.zjul5_root)
    os.makedirs(a.out, exist_ok=True)

    import h5py
    import torch

    teacher = _Teacher(a.teacher)                    # load before chdir (HF cache paths)
    print(f'ours: closed-form fit on {a.teacher}')

    os.chdir(a.depthor_root)                         # their configs need cwd == depthor root
    sys.path.insert(0, a.depthor_root)
    sys.argv = ['x', a.config]
    from src.config import args as dargs
    from src.dataloader.zju import ZJU
    from src.utils.model_io import load_weights
    Depthor = __import__('src.models.depthor_s', fromlist=['Depthor']).Depthor
    model = Depthor(n_bins=dargs.n_bins, min_val=dargs.min_depth,
                    max_val=dargs.max_depth, norm='linear').cuda()
    model = load_weights(model, dargs.weight_path)
    model.set_extra_param(device='cuda')
    model.eval()
    print('DEPTHOR-Small loaded')

    names = [m['filename'] for m in
             json.load(open(os.path.join(a.zjul5_root, 'ZJUL5', 'data.json')))['test']]
    loader = ZJU(dargs, 'online_eval').data
    want = set(a.indices)

    stats, keep = [], {}
    pooled = {'ours': {'p': [], 'g': []}, 'depthor': {'p': [], 'g': []}}
    for i, (batch, rel) in enumerate(zip(loader, names)):
        if a.limit and i >= a.limit:
            break
        with h5py.File(os.path.join(a.zjul5_root, 'ZJUL5', rel), 'r') as f:
            rgb = f['rgb'][:]
            gt = f['depth'][:].astype(np.float64)
            fr, hist, mask = f['fr'][:], f['hist_data'][:], f['mask'][:]
        h, w = gt.shape

        # DEPTHOR-Small, exactly as depthor_regions.py
        with torch.no_grad():
            _, pred = model({k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()})
        dep = pred[0, 0].float().cpu().numpy().astype(np.float64)
        if dep.shape != gt.shape:
            sys.exit(f'DEPTHOR pred {dep.shape} != gt {gt.shape}')

        # ours, exactly as zjul5_eval.py B4c_affine_cl
        u, v, z = zone_anchors(fr, hist, mask, h, w)
        if u.size < 8:
            continue
        disp = teacher.infer(np.ascontiguousarray(rgb))
        fit = anc.solve_robust(disp[v, u].astype(np.float64), 1.0 / z, np.ones_like(z), iters=1)
        if fit is None:
            continue
        ours = np.clip(anc.to_metric_depth(disp, *fit), None, MAX_DEPTH_M).astype(np.float64)

        gv = (gt > GT_MIN) & (gt < GT_MAX)
        if not gv.any():
            continue
        cm = coverage_mask(fr, mask, h, w)
        s = {'index': i, 'name': rel, 'anchors': int(u.size),
             'far_frac': float(np.mean(gt[gv] > a.far_m))}
        for region, sel in (('all', gv), ('in', gv & cm), ('out', gv & ~cm)):
            if sel.any():
                s[f'{region}_ours'] = M.depth_metrics(ours[sel], gt[sel])
                s[f'{region}_depthor'] = M.depth_metrics(dep[sel], gt[sel])
        stats.append(s)
        pooled['ours']['p'].append(ours[gv]); pooled['ours']['g'].append(gt[gv])
        pooled['depthor']['p'].append(dep[gv]); pooled['depthor']['g'].append(gt[gv])
        if want:
            if i in want:
                keep[i] = (rel, rgb, gt, ours, dep, cm)
        else:
            keep[i] = (rel, rgb, gt.astype(np.float32), ours.astype(np.float16),
                       dep.astype(np.float16), cm)
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(names)}', flush=True)

    pool = {k: M.depth_metrics(np.concatenate(v['p']), np.concatenate(v['g'])) for k, v in pooled.items()}
    print(f'\nscanned {len(stats)} images. Pooled AbsRel: ours {pool["ours"]["absrel"]:.3f} '
          f'(paper 0.094), DEPTHOR-Small {pool["depthor"]["absrel"]:.3f} (paper 0.079 published)')
    if not a.limit and abs(pool['ours']['absrel'] - 0.094) > 0.003:
        print('  ** ours does not reproduce the paper row; check the teacher id before using these images **')

    by_index = {s['index']: s for s in stats}
    if want:
        chosen = [('index', i) for i in a.indices if i in by_index]
    else:
        chosen = pick_indices(stats, [p.strip() for p in a.pick.split(',') if p.strip()], a.far_m)

    summary = {'teacher': a.teacher, 'scanned': len(stats), 'pooled': pool, 'picks': []}
    for tag, i in chosen:
        rel, rgb, gt, ours, dep, cm = keep[i]
        st = by_index[i]
        stem = render(a.out, tag, i, rel, rgb, np.asarray(gt, np.float64), np.asarray(ours, np.float64),
                      np.asarray(dep, np.float64), cm, st['all_ours'], st['all_depthor'], a.vmax, a.emax)
        print(f'{tag:7s} idx {i:3d}  {rel}  AbsRel ours {st["all_ours"]["absrel"]:.3f}  '
              f'DEPTHOR {st["all_depthor"]["absrel"]:.3f}  far>{a.far_m:g}m {100 * st["far_frac"]:.0f}%')
        summary['picks'].append({'tag': tag, 'stem': os.path.basename(stem), **st})
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    print(f'wrote {len(chosen)} image sets and summary.json to {a.out}')


if __name__ == '__main__':
    main()

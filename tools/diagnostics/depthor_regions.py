#!/usr/bin/env python3
"""Score DEPTHOR with OUR metric code and OUR region masks, so the split is comparable.

The published DEPTHOR numbers are over all valid-GT pixels. Ours are reported split by
whether a pixel falls inside the ToF footprint. Putting our inside-footprint row next to
their whole-frame row would be comparing different evaluation sets -- the objection that
makes "restrict to the ToF region and our metrics improve" inadmissible on its own.

Since their weights are public we can remove the objection instead of caveating it: run
their model, then score it through tools/diagnostics/metrics.py using the same coverage
mask, the same GT gate and the same regions as zjul5_eval.py. Identical code on both sides.

Their evaluate.py cannot help here -- its prediction-saving block is commented out, and its
compute_errors returns a key called `mae` that is actually AbsRel. So this drives their model
directly and ignores their metric code entirely.

Two scope notes:
  * Their dataloader emits sparse_depth as 64 SINGLE PIXELS (nonzero fraction 1e-4), not
    filled zone rectangles -- the same sparse-point form we splat. The ~50%-of-frame
    footprint used to define "inside" comes from the dataset's `fr` rectangles and is an
    evaluation region, not an input format.
  * We apply OUR gate (0.1-20 m), not their max_depth_eval=10, so both sides match our own
    numbers. The all-pixels row is printed for cross-checking against their published 0.079
    Rel / 0.371 RMSE; a gap there means the gate or the region logic differs, not the model.

Needs the DEPTHOR checkout on sys.path -- see --depthor-root.
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_perception'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as M                                              # noqa: E402
from zjul5_eval import coverage_mask, GT_MIN, GT_MAX             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depthor-root', default=os.path.expanduser('~/external/Depthor'))
    ap.add_argument('--config', default='configs/test_zju_small.txt')
    ap.add_argument('--variant', default='small', choices=['small', 'large'])
    ap.add_argument('--zjul5-root', required=True, help='dir containing ZJUL5/')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    # Resolve --out BEFORE the chdir below: their configs need cwd == depthor root, which
    # would otherwise silently relocate a relative output path (and only fail after the
    # whole 527-sample run had already completed).
    if a.out:
        a.out = os.path.abspath(a.out)
    a.zjul5_root = os.path.abspath(a.zjul5_root)

    import h5py
    import torch
    os.chdir(a.depthor_root)                 # their configs use relative imports/paths
    sys.path.insert(0, a.depthor_root)
    sys.argv = ['x', a.config]
    from src.config import args as dargs
    from src.dataloader.zju import ZJU
    from src.utils.model_io import load_weights

    mod = 'src.models.depthor_s' if a.variant == 'small' else 'src.models.depthor'
    Depthor = __import__(mod, fromlist=['Depthor']).Depthor
    model = Depthor(n_bins=dargs.n_bins, min_val=dargs.min_depth,
                    max_val=dargs.max_depth, norm='linear').cuda()
    model = load_weights(model, dargs.weight_path)
    model.set_extra_param(device='cuda')
    model.eval()

    # Their loader walks data.json['test'] in order, so the same list gives us the matching
    # h5 for each batch -- that is where `fr` (the zone rectangles) lives.
    names = [m['filename'] for m in
             json.load(open(os.path.join(a.zjul5_root, 'ZJUL5', 'data.json')))['test']]
    loader = ZJU(dargs, 'online_eval').data

    acc = {r: {'p': [], 'g': []} for r in ('all', 'in', 'out')}
    n = 0
    for batch, rel in zip(loader, names):
        if a.limit and n >= a.limit:
            break
        with torch.no_grad():
            _, pred = model({k: (v.cuda() if torch.is_tensor(v) else v)
                             for k, v in batch.items()})
        P = pred[0, 0].float().cpu().numpy()
        G = batch['depth'][0, 0].numpy()
        with h5py.File(os.path.join(a.zjul5_root, 'ZJUL5', rel), 'r') as f:
            fr, mask = f['fr'][:], f['mask'][:]
        h, w = G.shape
        if P.shape != G.shape:
            sys.exit(f'pred {P.shape} != gt {G.shape}; resolution assumption broken')
        cm = coverage_mask(fr, mask, h, w)
        gv = (G > GT_MIN) & (G < GT_MAX)
        for region, sel in (('all', gv), ('in', gv & cm), ('out', gv & ~cm)):
            if sel.any():
                acc[region]['p'].append(P[sel].astype(np.float64))
                acc[region]['g'].append(G[sel].astype(np.float64))
        n += 1
        if n % 100 == 0:
            print(f'  {n}/{len(names)}', flush=True)

    print(f'\nDEPTHOR-{a.variant}, {n} samples, scored with OUR metrics + OUR regions')
    rows, report = [], {}
    for region, label in (('all', 'ALL valid-GT pixels'),
                          ('in', 'INSIDE ToF footprint'),
                          ('out', 'OUTSIDE ToF footprint')):
        if not acc[region]['p']:
            continue
        mm = M.depth_metrics(np.concatenate(acc[region]['p']),
                             np.concatenate(acc[region]['g']))
        rows.append((label, mm))
        report[region] = mm
    print(M.format_table(rows))
    print('\ncross-check: their published all-pixel figures are Rel 0.079 / RMSE 0.371')
    print('(we gate 0.1-20 m, they gate to 10 m, so a small difference is expected)')

    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'variant': a.variant, 'n': n, 'regions': report}, f, indent=1)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()

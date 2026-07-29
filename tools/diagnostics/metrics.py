"""The one definition of every depth metric we report. Import this, don't re-derive.

Until now the harnesses reported MAE only, which is scale-dependent and appears in no
depth-estimation paper as a primary number. AbsRel and delta<1.25 are what the literature
uses, so without them nothing here can be placed next to a published result.

IMPORTANT: matching the metric FORM is necessary but not sufficient for comparability.
AbsRel measured on our lab is not comparable to AbsRel on NYUv2 or ZJU-L5 -- only the same
metric on the same data is. These functions make the comparison possible; they do not make
it valid on their own.

Pure numpy, no ROS/CUDA, so offline harnesses and training can both use it.
"""
import numpy as np

# Literature-standard accuracy thresholds: fraction of points whose ratio to ground
# truth is under 1.25, 1.25^2, 1.25^3.
DELTAS = (1.25, 1.25 ** 2, 1.25 ** 3)

KEYS = ('n_eval', 'coverage', 'mae', 'medae', 'p95', 'rmse', 'absrel', 'bias', 'd1', 'd2', 'd3')


def depth_metrics(pred, gt, min_depth=1e-3):
    """Metrics for one set of predictions against ground truth at the SAME points.

    pred, gt: 1-D metric depth in metres. pred may contain NaN/inf -- those points are
    counted against `coverage` and excluded from every accuracy metric, which is the
    only honest way to score a method that declines to predict (e.g. bilinear
    interpolation outside the convex hull of its anchors).

    Returns a dict with the keys in KEYS. Accuracy metrics are NaN when nothing is valid.
    """
    pred = np.asarray(pred, np.float64).ravel()
    gt = np.asarray(gt, np.float64).ravel()
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} vs gt {gt.shape}")

    n_eval = int(gt.size)
    ok = np.isfinite(pred) & np.isfinite(gt) & (pred > min_depth) & (gt > min_depth)
    out = {'n_eval': n_eval, 'coverage': float(ok.mean()) if n_eval else 0.0}
    if not ok.any():
        out.update({k: float('nan') for k in KEYS[2:]})
        return out

    p, g = pred[ok], gt[ok]
    e = p - g
    ratio = np.maximum(p / g, g / p)
    out['mae'] = float(np.abs(e).mean())
    # Report medae and p95 next to mae ALWAYS. Depth error here is heavy-tailed: a
    # method can have a 0.067 m median and a 7.98 m mean, because a small far-field
    # minority blows up. Quoting mae alone would call that method broken; quoting median
    # alone would hide a real failure. The gap between them IS the diagnostic.
    out['medae'] = float(np.median(np.abs(e)))
    out['p95'] = float(np.percentile(np.abs(e), 95))
    out['rmse'] = float(np.sqrt((e ** 2).mean()))
    out['absrel'] = float((np.abs(e) / g).mean())
    # Signed mean error. A large |bias| relative to MAE means a systematic scale/offset
    # problem rather than noise -- this is the shape the 2026-07-28 mirror bug had.
    out['bias'] = float(e.mean())
    for i, t in enumerate(DELTAS, start=1):
        out[f'd{i}'] = float((ratio < t).mean())
    return out


HEADER = (f"{'method':<14}{'n':>8}{'cov':>6}{'MAE':>10}{'medAE':>10}{'p95':>10}"
          f"{'AbsRel':>9}{'d<1.25':>9}{'bias':>10}")


def format_row(name, m):
    def f(k, w, p, suf=''):
        v = m.get(k, float('nan'))
        return ('nan' if not np.isfinite(v) else f'{v:.{p}f}{suf}').rjust(w)
    return (f"{name:<14}{m['n_eval']:>8d}{f('coverage',6,2)}{f('mae',9,3)}m"
            f"{f('medae',9,3)}m{f('p95',9,3)}m{f('absrel',9,3)}{f('d1',9,3)}"
            f"{f('bias',9,3)}m")


def format_table(rows):
    """rows: sequence of (name, metrics_dict)."""
    out = [HEADER, '-' * len(HEADER)]
    out += [format_row(n, m) for n, m in rows]
    return '\n'.join(out)


def binned(pred, gt, x, edges):
    """Metrics sliced by a per-point covariate -- used for error vs. angular distance
    from the nearest anchor, which is the curve that actually characterises
    extrapolation behaviour. Returns [(lo, hi, n, metrics), ...]."""
    pred, gt, x = map(lambda v: np.asarray(v).ravel(), (pred, gt, x))
    res = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        res.append((float(lo), float(hi), int(sel.sum()),
                    depth_metrics(pred[sel], gt[sel]) if sel.any() else None))
    return res

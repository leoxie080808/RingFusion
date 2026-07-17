"""Closed-form ToF metric anchoring (ported from the RingFusion ROS workspace,
already unit-tested there). Fits inv_depth_meas ~ a*disparity + b by weighted
least squares over the projected ToF anchors, then makes the whole mono-depth
map metric. Also returns the analytic covariance for per-pixel uncertainty."""
import numpy as np


def solve_scale_shift(disp, inv_depth, weights, eps=1e-9):
    w = np.asarray(weights, float)
    d = np.asarray(disp, float)
    s = np.asarray(inv_depth, float)
    if w.size < 2 or w.sum() <= eps:
        return None
    Sw = w.sum()
    Swd = np.sum(w * d)
    Sws = np.sum(w * s)
    Swdd = np.sum(w * d * d)
    Swds = np.sum(w * d * s)
    den = Sw * Swdd - Swd * Swd
    if abs(den) < eps:
        return None
    a = (Sw * Swds - Swd * Sws) / den
    b = (Sws - a * Swd) / Sw
    return float(a), float(b)


def solve_robust(disp, inv_depth, weights, iters=1, c=1.345):
    res = solve_scale_shift(disp, inv_depth, weights)
    if res is None:
        return None
    d = np.asarray(disp, float); s = np.asarray(inv_depth, float)
    w = np.asarray(weights, float).copy()
    for _ in range(max(0, iters)):
        a, b = res
        r = s - (a * d + b)
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (c * scale)
        hub = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
        r2 = solve_scale_shift(d, s, w * hub)
        if r2 is None:
            break
        res = r2
    return res


def covariance(disp, inv_depth, weights, a, b):
    """2x2 covariance of (a,b) from the same weighted sums."""
    w = np.asarray(weights, float); d = np.asarray(disp, float); s = np.asarray(inv_depth, float)
    n = np.count_nonzero(w > 0)
    if n <= 2:
        return None
    r = s - (a * d + b)
    sigma2 = (n / (n - 2.0)) * np.sum(w * r * r) / np.sum(w)
    Sw = w.sum(); Swd = np.sum(w * d); Swdd = np.sum(w * d * d)
    N = np.array([[Swdd, Swd], [Swd, Sw]])
    try:
        return sigma2 * np.linalg.inv(N)
    except np.linalg.LinAlgError:
        return None


def to_metric_depth(disp, a, b, min_disp=1e-4):
    inv = a * np.asarray(disp, float) + b
    inv = np.clip(inv, min_disp, None)
    return (1.0 / inv).astype(np.float32)

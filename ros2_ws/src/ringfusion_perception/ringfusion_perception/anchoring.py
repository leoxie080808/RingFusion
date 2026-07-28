"""Closed-form ToF metric anchoring (ported from the RingFusion ROS workspace,
already unit-tested there). Fits inv_depth_meas ~ a*disparity + b by weighted
least squares over the projected ToF anchors, then makes the whole mono-depth
map metric. Also returns the analytic covariance for per-pixel uncertainty."""
import numpy as np


# Range-exponent weighting (w = z**p). Superseded by the geometric ROI in roi.py: p=1
# fixed the far-field under-read but *cost* near-field accuracy, which is the wrong
# trade once the far field is out of scope. Kept at 0 (off) -- see range_weights.
RANGE_WEIGHT_P = 0.0


def range_weights(inv_depth, weights, p=RANGE_WEIGHT_P):
    """Weight anchors by z**p to stop the fit under-reading the far field.

    The fit is least squares on s = 1/z, so a near anchor at 0.3 m contributes s = 3.3
    while a far one at 2.4 m contributes s = 0.42 -- the far anchor carries ~8x less
    leverage. Measured on-robot that shows as a systematic far-field under-read
    (A/ToF ~0.65 on the farthest quartile of anchors) while the near field is exact.

    Theory says p=2: an inverse-depth residual ds maps to relative depth error
    ds * z, so w = z^2 makes the objective exactly relative depth error. **Measured,
    p=2 is much worse** -- it overshoots into a far OVER-read and wrecks the near
    field, because the affine model is misspecified (A's disparity is not exactly
    affine in 1/z), so re-weighting trades one regime for the other rather than
    fitting both. Absorbing that misspecification is Network B's job, not the fit's.

    Swept on the two 2026-07-25 captures (n=995 and n=703 anchors), median relative
    error / near ratio / far ratio:

        p     scene 1                     scene 2
        0.0   23.8% / 1.03 / 0.64         10.9% / 0.99 / 0.66     <- previous behaviour
        0.5   24.1% / 1.04 / 0.71         12.5% / 0.99 / 0.78
        1.0   21.9% / 1.05 / 0.76         11.1% / 1.00 / 0.96     <- chosen
        1.5   22.3% / 1.09 / 0.81         20.3% / 1.05 / 1.16
        2.0   24.0% / 1.17 / 0.85         41.0% / 1.32 / 1.27

    p=1 lifts the far ratio 0.64 -> 0.76 and 0.66 -> 0.96 while leaving the near field
    accurate. **It is nonetheless disabled (p=0).** Re-scored on only the anchors a robot
    can drive to (<= 2.5 m), p=1 is the WORST of the options -- 9.7% / 10.4% against 7.5%
    / 7.8% for uniform on scenes 2 and 3 -- because it buys far-field accuracy by
    spending near-field accuracy, and the far field is out of scope. The geometric ROI in
    roi.py replaces it and gets 6.0% / 6.1%. Kept here, off, so the sweep is not re-run
    from scratch if the scope ever changes.

    Normalised to mean 1 so `weights` keeps its original scale -- covariance() consumes
    the same effective weights and stays comparably scaled.
    """
    s = np.asarray(inv_depth, float)
    w = np.asarray(weights, float)
    if p == 0.0:
        return w
    zp = (1.0 / np.maximum(s, 1e-9)) ** p
    m = zp.mean()
    return w * (zp / m if m > 0 else 1.0)


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


def solve_robust(disp, inv_depth, weights, iters=1, c=1.345, range_weight=True):
    """range_weight applies the z**p term (a no-op at the current p=0).

    The region-of-interest weighting is NOT applied here -- it is geometric (floor plane
    + reach + height, see roi.py) and needs pixel coords and K, which this module does
    not have. The caller folds it into `weights` before calling; pipeline.run does this.
    covariance() must be passed the SAME weights or the reported uncertainty will not
    correspond to the fit that was actually solved."""
    if range_weight:
        weights = range_weights(inv_depth, weights)
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


def covariance(disp, inv_depth, weights, a, b, range_weight=True):
    """2x2 covariance of (a,b) from the same weighted sums.

    range_weight must match the value passed to solve_robust -- the covariance is only
    the covariance *of that fit* if it uses the same effective weights."""
    if range_weight:
        weights = range_weights(inv_depth, weights)
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
    # fp32, not fp64 -- this runs over the full 2 MP disparity every frame.
    inv = np.float32(a) * np.asarray(disp, np.float32) + np.float32(b)
    inv = np.clip(inv, min_disp, None)
    return (1.0 / inv).astype(np.float32)

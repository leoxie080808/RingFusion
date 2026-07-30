"""Region of interest -- which pixels the depth map actually exists to serve.

A ground robot navigates the local traversable area. The far wall, the shelving above
it and the posters on it are not what the depth map is for, but they drag a
two-parameter global fit around and they consume Network B's limited capacity. Measured
on-robot: scoring only the anchors a robot can drive to, an ROI-weighted fit cuts median
relative error from 7.5% to 6.0%.

The ROI is defined **geometrically**, from the ground plane:

    height  = signed distance above the fitted floor plane
    reach   = horizontal distance from the camera, measured in the floor plane
    inside  = (reach <= reach_max) AND (height <= height_max)

Why geometric rather than appearance-based: the obvious alternative is to find the VEX
field's black/white perimeter barcode and cut there. A prototype detector tracked the
strip in open view but locked onto trophy shelving and wall posters elsewhere -- other
high-contrast repeating structure scores the same. Geometry has no such failure mode, it
needs no barcode, it keeps game pieces sitting on the field (they are low and near), and
it works off the VEX field entirely. The trade is that it does not know the literal wall:
open floor at 3 m and a wall at 3 m are treated alike, which for navigation is the right
answer anyway.

The plane is fitted from the **ToF anchors**, which carry true measured depth. That
matters: it makes the ROI independent of the affine fit it is used to weight, so there is
no circularity in feeding ROI weights back into solve_robust.
"""
import numpy as np

REACH_MAX_M = 3.0        # how far along the floor we care about
HEIGHT_MAX_M = 0.6       # above this is the room, not the field (VEX perimeter is ~0.3 m)
OUTSIDE_W = 0.1          # soft, not hard -- see roi_weights
MIN_INSIDE = 32          # below this, don't gate at all


def _fxfycxcy(K):
    """This workspace stores K as (fx, fy, cx, cy); accept a 3x3 matrix too."""
    K = np.asarray(K, float)
    if K.shape == (3, 3):
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if K.size == 4:
        fx, fy, cx, cy = K.ravel()
        return fx, fy, cx, cy
    raise ValueError(f"unsupported K of shape {K.shape}; want (fx,fy,cx,cy) or 3x3")


def backproject(u, v, z, K):
    """Pixel + camera-frame z -> 3D camera-frame points (N,3)."""
    u = np.asarray(u, float); v = np.asarray(v, float); z = np.asarray(z, float)
    fx, fy, cx, cy = _fxfycxcy(K)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def fit_ground_plane(pts, iters=200, thresh=0.03, up_hint=(0.0, -1.0, 0.0), rng=None):
    """RANSAC a floor plane through 3D points. Returns (n, d) with |n| = 1 and
    n . p + d = 0 on the plane, n oriented so that n . up_hint > 0 (i.e. n points UP).

    up_hint is the approximate up direction in camera coords -- the camera is roughly
    upright, so -y. It is only used to orient and to reject near-vertical candidates
    (a wall is a plane too, and RANSAC will happily find it).
    Returns None if no plane is supported.
    """
    pts = np.asarray(pts, float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 16:
        return None
    rng = rng or np.random.default_rng(0)
    up = np.asarray(up_hint, float)
    up = up / max(np.linalg.norm(up), 1e-9)

    best_n, best_d, best_cnt = None, None, 0
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        if abs(float(n @ up)) < 0.7:      # reject walls: floor normal ~ parallel to up
            continue
        if float(n @ up) < 0:
            n = -n
        d = -float(n @ s[0])
        cnt = int((np.abs(pts @ n + d) < thresh).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt

    if best_n is None or best_cnt < 16:
        return None
    # refit on the inliers for a less noisy plane
    inl = pts[np.abs(pts @ best_n + best_d) < thresh]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c)
    n = vt[2] / max(np.linalg.norm(vt[2]), 1e-9)
    if float(n @ up) < 0:
        n = -n
    return n, -float(n @ c)


def height_and_reach(pts, plane):
    """(height above the plane, horizontal distance from the camera in the plane)."""
    n, d = plane
    pts = np.asarray(pts, float)
    height = pts @ n + d                       # signed: + is above the floor
    origin_h = d                               # camera height above the plane
    # project camera and points onto the plane, measure distance there
    p_flat = pts - np.outer(height, n)
    c_flat = np.zeros(3) - origin_h * n
    reach = np.linalg.norm(p_flat - c_flat, axis=-1)
    return height, reach


def inside_mask(pts, plane, reach_max=REACH_MAX_M, height_max=HEIGHT_MAX_M):
    """All-True when there is no plane: an ROI we cannot compute must not silently
    delete data. Every consumer degrades to 'no gating', i.e. today's behaviour."""
    pts = np.asarray(pts, float)
    if plane is None:
        return np.ones(pts.shape[:-1], bool)
    h, r = height_and_reach(pts, plane)
    return (r <= reach_max) & (h <= height_max)


class PlaneTracker:
    """Keeps a usable floor plane across frames that cannot fit one themselves.

    Two real cases motivate this, both raised from the field:
      * the perimeter wall is out of frame or only partly visible -- fine for the ROI
        itself (it is defined by floor geometry, not by seeing a wall) but the *fit*
        still needs floor points;
      * the robot is nose-in to a wall, so the wall fills the view and few ToF zones
        land on the floor at all.

    The camera is rigidly mounted, so the floor plane in camera coords is near-constant:
    measured across three independent captures the normal held to within 0.05 and the
    camera height to within 0.02 m (n = [-0.02..-0.05, -0.99..-1.00, -0.07..-0.12],
    height 0.15-0.17 m). So a cached plane is a good answer when the current frame has
    nothing to say, and an EMA suppresses per-frame jitter without lagging real pitch
    changes much.

    update() returns the plane to use, or None if none has ever been established.
    """

    def __init__(self, alpha=0.2, max_tilt=0.35, max_height_jump=0.10, refit_every=10):
        self.plane = None
        self.alpha = alpha
        self.max_tilt = max_tilt                  # reject a fit this far off the cached normal
        self.max_height_jump = max_height_jump
        # RANSAC here is a 200-iteration Python loop and cost 20.9 ms/frame measured on the
        # Orin -- the single largest component of the ROI stage. But the camera is rigidly
        # mounted and the plane is near-constant (see above: normal stable to 0.05, height to
        # 0.02 m across three captures), so refitting every frame buys nothing. Refit every
        # Nth frame and serve the cached, EMA-smoothed plane in between; that is already the
        # documented behaviour when a frame cannot fit one at all.
        self.refit_every = max(1, int(refit_every))
        self._n = 0

    def update(self, pts, **kw):
        self._n += 1
        if self.plane is not None and (self._n % self.refit_every) != 1:
            return self.plane
        fit = fit_ground_plane(pts, **kw)
        if fit is None:
            return self.plane                     # keep the last good one
        n, d = fit
        if self.plane is not None:
            n0, d0 = self.plane
            # Reject implausible jumps -- usually RANSAC latching onto a wall or a
            # table top in a floor-starved frame.
            if float(n @ n0) < 1.0 - self.max_tilt or abs(d - d0) > self.max_height_jump:
                return self.plane
            a = self.alpha
            n = a * n + (1 - a) * n0
            n = n / max(np.linalg.norm(n), 1e-9)
            d = a * d + (1 - a) * d0
        self.plane = (n, d)
        return self.plane


def roi_weights(pts, plane, weights=None, reach_max=REACH_MAX_M, height_max=HEIGHT_MAX_M,
                outside_w=OUTSIDE_W, min_inside=MIN_INSIDE):
    """Multiplicative ROI weights for a set of 3D points.

    Soft rather than hard: points outside keep `outside_w` instead of 0. Hard gating
    scored marginally better on the captures (6.0% vs 6.4%) but discards the far anchors
    entirely, so a frame aimed down a long open run can be left with too few points to
    fit. `min_inside` is the backstop -- if too few points fall inside, the gate is
    skipped and uniform weights are returned, which beats fitting on a handful.
    """
    pts = np.asarray(pts, float)
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, float).copy()
    if plane is None:
        return w
    ins = inside_mask(pts, plane, reach_max, height_max)
    if int(ins.sum()) < min_inside:
        return w
    return w * np.where(ins, 1.0, outside_w)


def pixel_roi_mask(depth, K, plane, reach_max=REACH_MAX_M, height_max=HEIGHT_MAX_M,
                   stride=1):
    """Dense per-pixel ROI mask from a metric depth map -- the form Network B's loss and
    the evaluation metrics need. Covers the whole frame, including everywhere the ToF has
    no coverage, because it runs off Network A's dense depth rather than the anchors."""
    D = np.asarray(depth, float)
    h, w = D.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = D[::stride, ::stride]
    pts = backproject(us.ravel(), vs.ravel(), z.ravel(), K)
    m = inside_mask(pts, plane, reach_max, height_max).reshape(z.shape)
    m &= z > 0
    if stride == 1:
        return m
    # UPSAMPLE, do not scatter. The previous version wrote `out[::stride, ::stride] = m`
    # and left every other pixel False, so the inside-fraction collapsed with stride
    # (measured 0.683 at stride 1 -> 0.011 at stride 8). Any caller using stride for speed
    # would have marked almost the whole frame as outside the ROI. The mask is a smooth
    # geometric region, so nearest-neighbour upsampling is the right reconstruction.
    return np.repeat(np.repeat(m, stride, axis=0), stride, axis=1)[:h, :w]

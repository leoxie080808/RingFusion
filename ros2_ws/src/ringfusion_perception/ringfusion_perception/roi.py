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

    VECTORISED: all `iters` hypotheses are drawn and scored as one (N, iters) matrix rather
    than one per Python iteration. The anchor set is at most 32x32 = 1024 points, so the
    whole problem is ~200x1024 = 205k distance evaluations -- small enough that the old
    Python loop was pure interpreter overhead. Measured on the Orin over 120 real anchor sets
    (N median 839, max 1023 -- the hardware cannot exceed 1024): 19.6 ms -> 1.18 ms median,
    p90 1.31, max 4.47. That 16.6x is what let the caller stop amortising the fit across
    frames (see PlaneTracker.refit_every).

    A CUDA version of exactly this was built and measured at 2.6-2.9 ms -- SLOWER than numpy,
    on every real frame. The problem is ~1.6 MB, well under the transfer + kernel-launch
    floor, and the refit below stays on the CPU regardless. Even a free hypothesis stage
    would floor at ~0.4 ms, worse than the 1.18 ms this already achieves. Do not "optimise"
    this onto the GPU.

    Sampling is WITH replacement, unlike the loop's rng.choice(replace=False). A duplicated
    index makes two sample points identical, so the cross product is ~0 and the hypothesis
    is dropped by the `ln > 1e-9` mask below -- the degenerate case is already handled, and
    at N~859 it costs ~1% of hypotheses. The loop discarded a comparable share to its own
    wall-rejection `continue`.
    """
    pts = np.asarray(pts, float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 16:
        return None
    rng = rng or np.random.default_rng(0)
    up = np.asarray(up_hint, float)
    up = up / max(np.linalg.norm(up), 1e-9)

    s = pts[rng.integers(0, len(pts), size=(iters, 3))]      # (iters,3,3)
    n = np.cross(s[:, 1] - s[:, 0], s[:, 2] - s[:, 0])       # (iters,3)
    ln = np.sqrt((n * n).sum(-1))
    ok = ln > 1e-9                                           # drop degenerate triples
    n = n / np.where(ok, ln, 1.0)[:, None]
    dot = n @ up
    ok &= np.abs(dot) >= 0.7          # reject walls: floor normal ~ parallel to up
    n = np.where((dot < 0)[:, None], -n, n)                  # orient UP
    d = -(n * s[:, 0]).sum(-1)                               # (iters,)

    # (N, iters) support counts -- every hypothesis scored against every point at once.
    cnt = np.where(ok, (np.abs(pts @ n.T + d[None, :]) < thresh).sum(0), 0)
    best = int(cnt.argmax())          # argmax takes the FIRST max, as `cnt > best_cnt` did
    if int(cnt[best]) < 16:
        return None

    # Refit on the inliers for a less noisy plane: the normal is the minor axis of the
    # inliers' scatter.
    #
    # This used to be np.linalg.svd(inl - c), which defaults to full_matrices=True and so
    # allocates and computes the full (N, N) left-singular matrix -- at the measured median
    # of 344 inliers that is a 344x344 decomposition, of which we use exactly one 3-vector,
    # and it grows with floor coverage. The old Python-loop hypothesis stage was 19.6 ms, so
    # this was invisible; once that dropped it became worth fixing (1.51 -> 1.18 ms on the
    # same 120 real frames).
    #
    # The 3x3 scatter matrix has the same eigenvectors as the right-singular vectors of
    # (inl - c), so eigh on it is exact, not an approximation -- and it is O(N) to form plus
    # a fixed 3x3 solve, i.e. independent of inlier count. eigh returns ASCENDING
    # eigenvalues, so column 0 is the least-variance direction = the plane normal.
    inl = pts[np.abs(pts @ n[best] + d[best]) < thresh]
    if len(inl) < 16:
        return None
    c = inl.mean(axis=0)
    q = inl - c
    nn = np.linalg.eigh(q.T @ q)[1][:, 0]
    nn = nn / max(np.linalg.norm(nn), 1e-9)
    if float(nn @ up) < 0:
        nn = -nn
    return nn, -float(nn @ c)


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

    def __init__(self, alpha=0.2, max_tilt=0.35, max_height_jump=0.10, refit_every=1):
        self.plane = None
        self.alpha = alpha
        self.max_tilt = max_tilt                  # reject a fit this far off the cached normal
        self.max_height_jump = max_height_jump
        # refit_every used to default to 10 for one reason only: fit_ground_plane was a
        # 200-iteration Python loop costing 20.9 ms/frame, the single largest component of
        # the ROI stage. Skipping 9 frames in 10 amortised it to ~2 ms -- but it also left a
        # 20.9 ms SPIKE every 10th frame, which is what a real-time consumer actually feels,
        # and it starved the EMA below of updates.
        #
        # Vectorising the fit (see fit_ground_plane) took it to 1.5 ms, so that trade is gone
        # and the default is now to refit EVERY frame. This is the better answer on accuracy
        # too: measured over 5 seeds on 60 real frames, RANSAC's own normal scatter is a
        # median 4.0 deg but a p90 of 21.6 deg, so a single unlucky fit held for 10 frames is
        # a real error source. Refitting every frame lets the EMA average that scatter down
        # instead of latching one draw of it.
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
                   stride=1, expand=True):
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
    if not expand:
        # Hand back the STRIDED mask and let the caller expand it where the data already
        # lives. gpu_ops.roi_sigma_floor_lowres does exactly that on the GPU: the expansion
        # below allocates and writes a full 2 MP bool array on the CPU only for it to be
        # copied straight to the GPU, which profiled at 23.5 ms/frame on the robot.
        return m
    # UPSAMPLE, do not scatter. The previous version wrote `out[::stride, ::stride] = m`
    # and left every other pixel False, so the inside-fraction collapsed with stride
    # (measured 0.683 at stride 1 -> 0.011 at stride 8). Any caller using stride for speed
    # would have marked almost the whole frame as outside the ROI. The mask is a smooth
    # geometric region, so nearest-neighbour upsampling is the right reconstruction.
    return np.repeat(np.repeat(m, stride, axis=0), stride, axis=1)[:h, :w]

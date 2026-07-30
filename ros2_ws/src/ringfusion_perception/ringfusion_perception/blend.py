"""Distance-weighted blend of raw ToF against the network's depth.

Measured 2026-07-28 (tools/diagnostics/baselines.py, 1234 logged pairs): NEITHER source
is best everywhere, and the crossover is sharp.

    median abs error, by angular distance from the nearest ToF anchor
                       0-3 deg   3-6 deg   6-10 deg  10-15 deg  15-30 deg
      nearest-zone ToF   0.027     0.072     0.122     0.180     0.239
      Network B (v4)     0.048     0.047     0.052     0.042     0.038

Copying the nearest ToF reading wins by ~2x under 3 deg and loses by ~6x past 15 deg.
The network is flat because it reads structure out of the image; the ToF degrades because
"copy your neighbour" only works while the neighbour is near. Crossover is ~3 deg.

So take the ToF where it is close and the network where it is not, with a smoothstep
between so no seam appears in the depth map. Weight is driven by ANGLE, not pixels, so it
follows the optics rather than the resolution.

CAVEAT worth keeping in mind: the numbers above are measured AT ToF zone centres, which is
exactly where nearest-neighbour looks best. Between zones -- especially across a depth
discontinuity -- the nearest anchor can belong to a different surface, while the network
has the image and gets the edge right. So this blend is expected to help on the smooth
interior of the ToF cone and to be neutral-or-worse on object boundaries within it. Score
it with baselines.py (method B6) before trusting it; do not assume it is free.
"""
import numpy as np

NEAR_DEG = 2.0     # fully trust ToF at or below this angular distance from an anchor
FAR_DEG = 5.0      # fully trust the network at or above this

# Scene-bounded far-field cap. See scene_cap() below.
SCENE_CAP_K = 2.0
SCENE_CAP_FLOOR_M = 1.0


def scene_cap(anchor_depth, anchor_mask, k=SCENE_CAP_K, floor_m=SCENE_CAP_FLOOR_M,
              hard_max=None):
    """Per-frame far-field ceiling, derived from what the ToF actually measured.

    D = 1/(a*disp + b), so wherever (a*disp + b) approaches zero the depth runs away. The
    fixed MAX_DEPTH_M=20 ceiling bounds that but is unrelated to the scene: on a room whose
    ToF sees 0.33-3.24 m, a pixel emitting 18 m is the fit extrapolating into a low-disparity
    region, not a surface 18 m away. RMSE squares errors, so a handful of those dominate it --
    on ZJU-L5 we match published methods on delta1/Rel and sit 2.2-2.8x worse on RMSE, and the
    gap is concentrated OUTSIDE the ToF footprint (1.398 vs 0.418 inside).

    This returns k * (furthest valid anchor this frame), floored so a frame that only sees
    close surfaces cannot clamp everything to near-zero.

    Uses ONLY sensor input -- never ground truth -- so applying it is not benchmark tuning.
    But `k` IS a hyperparameter: choose it on our own logs or a train split, never on the
    test set being reported.

    HONEST TRADE, because this is not free. Two kinds of pixel get capped:
      * fit blew up, truth is near   -> error shrinks a lot          (the win)
      * genuinely far, e.g. a wall seen through a doorway at 15 m
        while the ToF only reaches 3 m -> now wrongly pulled in      (the cost)
    So it swaps UNBOUNDED errors for BOUNDED ones; it does not make anything more accurate.
    Fine for navigation, where "further than the cap" is operationally just "far", and a real
    limitation for general-purpose depth. Callers should mark capped pixels (see the returned
    mask in apply_scene_cap) rather than pass them off as measurements.

    Returns None if there are no usable anchors, meaning "no opinion -- leave depth alone".
    """
    m = np.asarray(anchor_mask) > 0
    if not m.any():
        return None
    d = np.asarray(anchor_depth, np.float32)[m]
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return None
    cap = max(float(k) * float(d.max()), float(floor_m))
    if hard_max is not None:
        cap = min(cap, float(hard_max))
    return cap


def apply_scene_cap(D, anchor_depth, anchor_mask, k=SCENE_CAP_K,
                    floor_m=SCENE_CAP_FLOOR_M, hard_max=None):
    """-> (D_capped, capped_mask, cap). capped_mask marks pixels whose value is now a LOWER
    BOUND, not an estimate; feed it to the variance channel so consumers can tell the
    difference. cap is None when no anchors were usable and D is returned unchanged."""
    cap = scene_cap(anchor_depth, anchor_mask, k, floor_m, hard_max)
    D = np.asarray(D, np.float32)
    if cap is None:
        return D, np.zeros(D.shape, bool), None
    capped = D > cap
    if not capped.any():
        return D, capped, cap
    return np.where(capped, np.float32(cap), D).astype(np.float32), capped, cap


def blend_depth(D_net, anchor_depth, anchor_mask, fx,
                near_deg=NEAR_DEG, far_deg=FAR_DEG):
    """Blend nearest-anchor ToF depth into D_net near the anchors.

    D_net         (H,W) float32 metric depth from the pipeline (closed-form or +residual)
    anchor_depth  (H,W) float32, ToF depth splatted at anchor pixels, 0 elsewhere
    anchor_mask   (H,W) nonzero at anchor pixels
    fx            focal length in px, to convert the angular thresholds to pixels

    Returns (D_blended, w) where w is 1 where the ToF is trusted and 0 where the network
    is. Falls back to D_net unchanged if there are no anchors.
    """
    import cv2
    m = np.asarray(anchor_mask) > 0
    D_net = np.asarray(D_net, np.float32)
    if not m.any():
        return D_net, np.zeros_like(D_net)

    # distanceTransform measures distance to the nearest ZERO pixel, so anchors are the
    # zeros here. DIST_LABEL_PIXEL additionally returns, per pixel, the label of the
    # nearest zero pixel -- labels numbered from 1 in raster order over those zeros,
    # which is the same order np.nonzero yields, so a plain LUT maps label -> depth.
    src = np.where(m, 0, 255).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

    ys, xs = np.nonzero(m)
    lut = np.zeros(int(labels.max()) + 1, np.float32)
    lut[1:ys.size + 1] = np.asarray(anchor_depth, np.float32)[ys, xs]
    D_tof = lut[labels]

    near_px = float(fx) * np.tan(np.deg2rad(near_deg))
    far_px = float(fx) * np.tan(np.deg2rad(far_deg))
    t = np.clip((dist - near_px) / max(far_px - near_px, 1e-6), 0.0, 1.0)
    w = (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)   # smoothstep, 1 near -> 0 far

    # Guard against anchors that splatted a nonpositive depth: fall back to the network
    # there rather than blending toward zero.
    w = np.where(D_tof > 0, w, 0.0).astype(np.float32)
    return (w * D_tof + (1.0 - w) * D_net).astype(np.float32), w

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

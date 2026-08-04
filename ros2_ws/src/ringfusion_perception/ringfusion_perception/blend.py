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

# Downsample factor for the distance transform. cv2.distanceTransformWithLabels over a full
# 2 MP frame costs ~57 ms on the Orin CPU, which alone took pipeline.run from 19.2 to 9.2 Hz.
# The blend weight is a smoothstep over ~20 px and the nearest-anchor depth is piecewise
# constant (a Voronoi diagram), so both survive being computed at 1/scale resolution and
# upsampled. 4 quantises distance to 4 px against a ~20 px ramp.
BLEND_SCALE = 4

# Scene-bounded far-field cap. See scene_cap() below.
SCENE_CAP_K = 2.0
SCENE_CAP_FLOOR_M = 1.0

# --- sigma terms this module can supply, added 2026-08-04 after LIVE-4 -------------------
# LIVE-4 measured sigma against 11 tape points and found two specific failures, both of
# which are visible right here in the blend and were being discarded:
#
#   1. MIXED RETURNS. A marker on a banner 2.89 m back whose ToF zone clips a bottle 0.33 m
#      away: the pipeline reported 0.34 m -- off by 2.55 m -- with sigma 0.66, i.e. 3.9
#      sigma. The blend KNEW: it was pulling depth from 2.9 m to 0.33 m, a 2.5 m move it
#      made with full confidence. The size of that move is the uncertainty signal.
#      "Far surface with a near object in front" is what an obstacle IS, so this is the
#      worst case to be confident about.
#
#   2. ANGLE ASYMMETRY. A marker 47.8 deg off-axis -- outside the +/-36.75 deg cone -- was
#      wrong by 0.47 m with sigma 0.23, while markers outside the cone VERTICALLY reported
#      sigma above 1.8 for comparable errors. The vertical ones were caught by the Stage 7d
#      ROI floor (they are above height_max); the horizontal one sits inside the ROI box and
#      so was missed entirely. Angular distance to the nearest anchor is a Euclidean
#      distance transform, so keying off it is symmetric by construction.
#
# Both are expressed as sigma in METRES and combined in quadrature with the existing
# analytic + learned variance -- they add uncertainty, they never reduce it.
DISAGREE_K = 1.0        # sigma contribution per metre the blend moves the depth
SUPPORT_FRAC = 0.35     # sigma as a fraction of D, per "far_deg" of missing angular support
SPREAD_K = 1.0          # sigma contribution per metre of disagreement BETWEEN nearby zones
# Neighbourhood for that disagreement, in REDUCED-grid cells. It must span more than one ToF
# zone or it cannot see two zones disagree, which is the whole point. At 1640x1232 the 32x32
# grid projects to a 567x443 px footprint -> ~17.7 px per zone across, ~4.4 reduced cells at
# BLEND_SCALE 4. 11 cells is ~44 px, about 2.5 zones. The first version used 3 cells (12 px),
# SMALLER than a single zone, and consequently did almost nothing on real data while passing a
# synthetic test whose anchors happened to be 20 px apart.
SPREAD_WIN = 11

# Why a third term rather than a bigger DISAGREE_K. First attempt used only ToF-vs-network
# disagreement and moved the mixed-return marker from 3.9 to 2.6 sigma -- better, not fixed.
# The reason is that at that marker the NETWORK is wrong too (it reads ~1.06 m against a
# 2.89 m tape), so the two sources agree with each other while both being wrong, and their
# disagreement understates the error. Raising DISAGREE_K to compensate would inflate sigma
# everywhere else to buy one point.
#
# Spread BETWEEN NEIGHBOURING ZONES does not have that weakness: it needs no opinion from
# the network at all. Where the nearest-anchor field jumps from 0.33 m to 2.9 m across a few
# cells, the zone grid is straddling a depth discontinuity, and that is true whether or not
# anything else in the pipeline has noticed.


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


def sigma_support_var(D_net, dist_r, tof_r, scale, fx,
                      near_deg=NEAR_DEG, far_deg=FAR_DEG):
    """Extra VARIANCE (m^2, full resolution) from the two blend-visible failure modes.

    Computed on the REDUCED grid and block-replicated up, the same trick blend_apply_lowres
    uses: both terms vary over the ~20 px blend ramp, so 1/scale resolution loses nothing
    and costs 1/scale^2 of the arithmetic.

    disagreement -- w * |D_tof - D_net|, the distance the blend actually moves the depth.
        Weighted by the blend weight because that is how much of the move is real: far from
        any anchor w -> 0, the nearest anchor is an irrelevant Voronoi neighbour, and its
        disagreement should not inflate anything.
    support -- how far past far_deg the nearest anchor is, as a fraction of D. Zero
        wherever an anchor is within far_deg, so it never touches the well-supported
        interior of the cone.
    """
    import numpy as _np
    S = max(1, int(scale))
    h, w_full = D_net.shape
    net_r = _np.ascontiguousarray(D_net[::S, ::S][:dist_r.shape[0], :dist_r.shape[1]])
    dist_px = _np.asarray(dist_r, _np.float32) * float(S)
    tof_r = _np.asarray(tof_r, _np.float32)

    near_px = float(fx) * _np.tan(_np.deg2rad(near_deg))
    far_px = float(fx) * _np.tan(_np.deg2rad(far_deg))
    t = _np.clip((dist_px - near_px) / max(far_px - near_px, 1e-6), 0.0, 1.0)
    wgt = 1.0 - t * t * (3.0 - 2.0 * t)
    wgt = _np.where(tof_r > 0, wgt, 0.0).astype(_np.float32)

    disagree = DISAGREE_K * wgt * _np.abs(tof_r - net_r)

    ang = _np.degrees(_np.arctan(dist_px / max(float(fx), 1e-6)))
    short = _np.clip((ang - float(far_deg)) / max(float(far_deg), 1e-6), 0.0, None)
    support = SUPPORT_FRAC * _np.maximum(net_r, 0.0) * short

    # Disagreement between NEIGHBOURING zones. tof_r is the nearest-anchor field, so it is
    # piecewise constant over Voronoi cells; a large max-min across a small window means
    # adjacent cells belong to different surfaces, i.e. the zone grid straddles a depth
    # edge. Valid depths only -- empty splat cells are 0 and would fake a huge spread.
    # Deviation of THIS pixel's own anchor from its neighbourhood, not the neighbourhood's
    # full max-min. max-min fires on any pixel merely NEAR an edge, which in a cluttered
    # scene is most of the frame: it took a correct 2.02 m reading from sigma 0.97 to 2.72.
    # What actually signals a straddled zone is the pixel's own anchor being the ODD ONE
    # OUT. A local mean over valid cells only (blur of the masked field divided by blur of
    # the mask) gives that at box-filter cost.
    import cv2 as _cv
    kw = (int(SPREAD_WIN), int(SPREAD_WIN))
    valid = (tof_r > 0).astype(_np.float32)
    num = _cv.blur(_np.where(tof_r > 0, tof_r, 0.0).astype(_np.float32), kw)
    den = _cv.blur(valid, kw)
    local = _np.where(den > 1e-6, num / _np.maximum(den, 1e-6), tof_r)
    spread = _np.where(tof_r > 0, _np.abs(tof_r - local), 0.0)
    spread = SPREAD_K * wgt * spread.astype(_np.float32)

    var_r = (disagree ** 2 + support ** 2 + spread ** 2).astype(_np.float32)

    from . import gpu_ops as _g
    if _g.available():
        return _g.upsample_var(var_r, S, h, w_full)
    return _np.repeat(_np.repeat(var_r, S, axis=0), S, axis=1)[:h, :w_full]


def blend_depth(D_net, anchor_depth, anchor_mask, fx,
                near_deg=NEAR_DEG, far_deg=FAR_DEG, scale=BLEND_SCALE,
                return_fields=False):
    """Blend nearest-anchor ToF depth into D_net near the anchors.

    D_net         (H,W) float32 metric depth from the pipeline (closed-form or +residual)
    anchor_depth  (H,W) float32, ToF depth splatted at anchor pixels, 0 elsewhere
    anchor_mask   (H,W) nonzero at anchor pixels
    fx            focal length in px, to convert the angular thresholds to pixels

    Returns (D_blended, w) where w is 1 where the ToF is trusted and 0 where the network
    is. Falls back to D_net unchanged if there are no anchors.

    With return_fields=True the tuple gains a third element (dist_r, tof_r, scale): the
    nearest-anchor distance and depth on the reduced grid. Stage 6's sigma needs exactly
    these and recomputing the distance transform to get them would double the stage cost,
    so they are handed out rather than rebuilt. See sigma_support_var.
    """
    import cv2
    m = np.asarray(anchor_mask) > 0
    D_net = np.asarray(D_net, np.float32)
    if not m.any():
        z = np.zeros_like(D_net)
        return (D_net, z, None) if return_fields else (D_net, z)

    # distanceTransform measures distance to the nearest ZERO pixel, so anchors are the
    # zeros here. DIST_LABEL_PIXEL additionally returns, per pixel, the label of the
    # nearest zero pixel -- labels numbered from 1 in raster order over those zeros,
    # which is the same order np.nonzero yields, so a plain LUT maps label -> depth.
    h, w = m.shape
    S = max(1, int(scale))
    ys, xs = np.nonzero(m)
    ad = np.asarray(anchor_depth, np.float32)

    if S == 1:
        src = np.where(m, 0, 255).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(
            src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(int(labels.max()) + 1, np.float32)
        lut[1:ys.size + 1] = ad[ys, xs]
        D_tof = lut[labels]
        fields = (dist, D_tof, 1)
    else:
        # Scatter anchors into a 1/S grid rather than resizing the sparse mask -- resizing
        # would drop most single-pixel anchors. Collisions within a cell keep one anchor,
        # which is fine: they are within S px of each other.
        hs, ws = (h + S - 1) // S, (w + S - 1) // S
        red_m = np.zeros((hs, ws), bool)
        red_d = np.zeros((hs, ws), np.float32)
        red_m[ys // S, xs // S] = True
        red_d[ys // S, xs // S] = ad[ys, xs]
        src = np.where(red_m, 0, 255).astype(np.uint8)
        dist_r, labels_r = cv2.distanceTransformWithLabels(
            src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        rys, rxs = np.nonzero(red_m)
        lut = np.zeros(int(labels_r.max()) + 1, np.float32)
        lut[1:rys.size + 1] = red_d[rys, rxs]
        tof_r = lut[labels_r]
        # Keep the reduced maps; they are expanded on the GPU below (or on the CPU in the
        # numpy fallback). Expanding here would build two 2 MP arrays just to copy them.
        near_px_e = float(fx) * np.tan(np.deg2rad(near_deg))
        far_px_e = float(fx) * np.tan(np.deg2rad(far_deg))
        fields = (dist_r, tof_r, S)
        from . import gpu_ops as _g
        if _g.available():
            out, wgt = _g.blend_apply_lowres(D_net, dist_r, tof_r, S, near_px_e, far_px_e)
            return (out, wgt, fields) if return_fields else (out, wgt)
        # dist_r is in reduced pixels -> scale back to full-res pixel units
        dist = np.repeat(np.repeat(dist_r * S, S, axis=0), S, axis=1)[:h, :w]
        D_tof = np.repeat(np.repeat(tof_r, S, axis=0), S, axis=1)[:h, :w]

    near_px = float(fx) * np.tan(np.deg2rad(near_deg))
    far_px = float(fx) * np.tan(np.deg2rad(far_deg))

    # The tail (ramp + guard + mix) is ~8 full-frame float passes. Offload it, exactly as
    # ResidualRefiner.refine's apply step was; numpy fallback keeps off-robot behaviour.
    from . import gpu_ops
    if gpu_ops.available():
        out, wgt = gpu_ops.blend_apply(D_net, dist, D_tof, near_px, far_px)
        return (out, wgt, fields) if return_fields else (out, wgt)

    t = np.clip((dist - near_px) / max(far_px - near_px, 1e-6), 0.0, 1.0)
    w = (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)   # smoothstep, 1 near -> 0 far

    # Guard against anchors that splatted a nonpositive depth: fall back to the network
    # there rather than blending toward zero.
    w = np.where(D_tof > 0, w, 0.0).astype(np.float32)
    out = (w * D_tof + (1.0 - w) * D_net).astype(np.float32)
    return (out, w, fields) if return_fields else (out, w)

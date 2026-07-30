"""RingFusion perception pipeline -- the pure-numpy math, with no ROS or CUDA
dependency so it can be imported and unit-tested on any machine.

`run()` executes stages 2-8 of the technical reference:

  2  backbone inference            disp = f(rgb)                (backbone arg)
  3  zone projection               ToF zone -> pixel + z_cam    (geometry.py)
  4  anchor pairing                (disp_i, 1/z_i, w_i)
  5  closed-form anchoring         fit (a, b), make map metric  (anchoring.py)
  6  analytic covariance           per-pixel var via delta method
  7  residual refinement           per-pixel (da, db) + extra var (residual arg)
  8  unprojection                  metric depth -> point cloud  (geometry.py)

The backbone and residual are injected so the same pipeline runs with mocks on a
dev PC and with TensorRT engines on the Jetson. Pass residual=None to stop after
the closed-form fit (Stage 6); an untrained/mock residual is the identity anyway.
"""
import numpy as np

from . import geometry as geo
from . import anchoring as anc
from . import gpu_ops
from . import blend as blend_mod
from . import roi
from .blend import blend_depth
from .residual import MAX_DEPTH_M

# Sigma floor outside the ROI, as a fraction of depth. 1.0 = "100% relative uncertainty",
# i.e. no useful bound. Measured errors there exceed 100% of the predicted value, so this
# is not conservative -- it is the minimum honest signal.
ROI_OUTSIDE_SIGMA_FRAC = 1.0

# roi.pixel_roi_mask back-projects every pixel into float64 3D points: at 2 MP that is
# ~271 ms, which alone took pipeline.run from 9.2 to 2.6 Hz. The mask only gates the sigma
# floor, so an 8 px boundary quantisation is irrelevant to its purpose. NOTE this argument
# was BROKEN until 2026-07-30 -- it scattered instead of upsampling, collapsing the
# inside-fraction from 0.683 to 0.011 -- so it had never been usable.
ROI_MASK_STRIDE = 8


def splat_anchors(u, v, z, inb, shape):
    """Write each in-image ToF zone's camera-frame depth into an empty map at its
    projected pixel, and mark it in a validity mask. These two sparse channels are
    what let the residual net reason about *where* the anchors were (error grows
    with distance from an anchor). u, v are integer pixel arrays; inb selects the
    zones that landed in-image."""
    h, w = shape
    anchor_depth = np.zeros((h, w), np.float32)
    anchor_mask = np.zeros((h, w), np.float32)
    uu, vv = u[inb], v[inb]
    anchor_depth[vv, uu] = z[inb].astype(np.float32)
    anchor_mask[vv, uu] = 1.0
    return anchor_depth, anchor_mask


def run(rgb, tof_dist_m, tof_valid, calib, backbone, residual=None,
        confidence=None, min_confidence=-1, cloud_stride=4, use_gpu=None,
        blend=True, blend_near=blend_mod.NEAR_DEG, blend_far=blend_mod.FAR_DEG,
        roi_enable=True, roi_weight_fit=False,
        roi_reach_max=roi.REACH_MAX_M, roi_height_max=roi.HEIGHT_MAX_M,
        plane_tracker=None, roi_mask_stride=ROI_MASK_STRIDE, timings=None):
    """One perception frame.

    Args:
      rgb          HxWx3 uint8, rectified camera image.
      tof_dist_m   (rows, cols) float32 ToF ranges in metres, NaN where no return.
      tof_valid    (rows, cols) bool, a real range was measured.
      calib        dict from load_calib: K, dist, model, T_cam_tof, fov_h, fov_v.
      backbone     object with .infer(rgb) -> disparity HxW.
      residual     object with .refine(...) -> (depth, extra_var), or None.
      confidence   (rows, cols) per-zone confidence, or None.
      min_confidence  if >= 0 and confidence given, reject zones below this and
                      weight the fit by confidence. Default -1 = ignore confidence
                      entirely (uniform weights -- the original tested behaviour).
      timings      optional dict; if given, each stage records its wall-clock ms into it.
                   None (the default) skips every timer, so the deployed path pays nothing.
                   Needed because the deployed rate (7.2 Hz) disagreed sharply with the
                   offline benchmark (12.3 Hz) and a per-stage breakdown is the only way to
                   find out which stage costs more on the robot than it does on logged
                   frames. GPU stages are safe to time here: every gpu_ops call ends in a
                   .cpu() copy, which blocks until the GPU has finished, so the elapsed
                   time is the true cost rather than just the launch time.

    Returns dict:
      ok         bool. False if too few anchors to fit (nothing else populated).
      n_anchors  int, zones used in the fit.
      metric     HxW float32 metric depth (m).
      var        HxW float per-pixel depth variance (analytic + residual), or None.
      cloud      (M,3) float32 camera-frame point cloud.
      a, b       fitted affine inverse-depth parameters.
    """
    h, w = rgb.shape[:2]
    K = calib['K']
    rows, cols = tof_dist_m.shape

    if timings is None:
        def _t(_name):                    # no-op: deployed path pays nothing
            return None
    else:
        import time as _time

        _mark = [_time.perf_counter()]

        def _t(name):
            now = _time.perf_counter()
            timings[name] = timings.get(name, 0.0) + (now - _mark[0]) * 1e3
            _mark[0] = now

    # Stage 2 -- backbone relative disparity
    disp = backbone.infer(rgb)
    _t('2_backbone')

    # Stage 3 -- project each ToF zone to a camera pixel (parallax-correct)
    proj = geo.project_zone_to_pixel(
        tof_dist_m, tof_valid, cols, rows, calib['fov_h'], calib['fov_v'],
        calib['T_cam_tof'], K, calib['dist'], model=calib['model'])
    uv = proj['uv']; z = proj['z_cam']; ok = proj['valid']

    # Stage 4 -- pair zones with the backbone's disparity at their pixels
    finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
    u = np.round(np.where(finite, uv[:, 0], -1)).astype(int)
    v = np.round(np.where(finite, uv[:, 1], -1)).astype(int)
    inb = ok & finite & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)

    conf_flat = None
    if confidence is not None and min_confidence >= 0:
        conf_flat = np.asarray(confidence, np.float32).reshape(-1)
        inb = inb & (conf_flat >= min_confidence)

    du = disp[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)]
    disp_at = du[inb]
    inv_depth = 1.0 / z[inb]                       # s_i uses camera-frame z, not slant range
    if conf_flat is not None:
        weights = np.maximum(conf_flat[inb], 1.0)  # trust high-confidence zones more
    else:
        weights = np.ones_like(inv_depth)

    _t('3_4_project_pair')

    # Stage 4b -- geometric ROI weighting (roi.py). The depth map exists to serve the
    # local traversable area; the far wall and the shelving above it drag a 2-parameter
    # global fit around. roi.py was written for exactly this, measured at 7.5% -> 6.0%
    # median relative error on the anchors a robot can drive to, and then never imported
    # by anything -- range weighting had already been disabled (RANGE_WEIGHT_P = 0.0) on
    # the grounds that ROI "replaces it", so the pipeline shipped with NEITHER mitigation.
    #
    # Measured 2026-07-29 with neither active: the ceiling 1/b sits at a median 1.43 m
    # while the ToF's furthest anchor is 4.16 m, so 14.4% of anchors are not even
    # expressible, and anchors beyond 1.5 m carry 69% of total error while being 16% of
    # anchors. Weights are SOFT (outside_w=0.1) so a frame aimed down a long open run is
    # not left with too few points to fit.
    # roi_weight_fit is OFF by default. A/B'd on 200 of our logs (plane found on 200/200):
    # weighting the fit by ROI moved the ceiling 1.33 -> 1.11 m and degraded the far field
    # (1.5-3 m median |e| 1.075 -> 1.200, 3-6.5 m 2.615 -> 2.842, pooled MAE 0.328 ->
    # 0.344) while buying only 0.109 -> 0.101 at 0.5-1.5 m and nothing under 0.5 m.
    # That is the documented trade working as designed -- roi.py narrows SCOPE, it does not
    # fix the under-read -- but it is not a default worth paying for. The plane and mask
    # are still computed, because Stage 7d needs them.
    plane = None
    if roi_enable:
        pts_a = roi.backproject(u[inb], v[inb], z[inb], K)
        plane = (plane_tracker.update(pts_a) if plane_tracker is not None
                 else roi.fit_ground_plane(pts_a))
        if plane is not None and roi_weight_fit:
            weights = roi.roi_weights(pts_a, plane, weights,
                                      reach_max=roi_reach_max, height_max=roi_height_max)

    _t('4b_roi_plane')

    # Stage 5 -- closed-form weighted least squares + one Huber pass
    fit = anc.solve_robust(disp_at, inv_depth, weights, iters=1)
    if fit is None:
        return {'ok': False, 'n_anchors': int(inb.sum())}
    a, b = fit
    # The per-pixel 2 MP math (metric depth, variance, cloud) is the pipeline's real
    # cost -- the GPU sits idle while numpy grinds it on the CPU. Offload it to torch
    # when CUDA is present; fall back to numpy (fp32) so off-robot tests still run.
    gpu = gpu_ops.available() if use_gpu is None else use_gpu

    # Stage 5 -- closed-form metric depth (D0)
    metric = (gpu_ops.to_metric_depth(disp, a, b) if gpu
              else anc.to_metric_depth(disp, a, b))

    _t('5_fit_metric')

    # Stage 6 -- analytic per-pixel variance by the delta method.
    # Var[D](p) = D^4 * j^T Cov(a,b) j,  j = (disp, 1). The D^4 factor is why far
    # pixels carry much larger variance -- inverse-depth error amplifies with range.
    var = None
    cov = anc.covariance(disp_at, inv_depth, weights, a, b)
    if cov is not None:
        if gpu:
            var = gpu_ops.analytic_variance(disp, metric, cov)
        else:
            j0, j1 = disp, np.ones_like(disp)
            quad = (j0 * (cov[0, 0] * j0 + cov[0, 1] * j1) +
                    j1 * (cov[1, 0] * j0 + cov[1, 1] * j1))
            var = (metric.astype(np.float32) ** 4) * quad      # fp32 (was fp64)

    # Anchors are needed by BOTH the residual and the blend, so splat once regardless.
    anchor_depth, anchor_mask = splat_anchors(u, v, z, inb, (h, w))

    _t('6_variance')

    # Stage 7 -- residual refinement (identity if residual is None/mock). Stays numpy;
    # gpu_ops returns numpy, so Network B sees exactly the types it did before -- in
    # particular an UNCLAMPED D0, which is what build_residual_inputs produced during
    # training. Clamping before this point would be a train/deploy mismatch.
    if residual is not None:
        metric, var_extra = residual.refine(rgb, metric, disp,
                                             anchor_depth, anchor_mask, a, b)
        if var is not None and var_extra is not None:
            var = var + var_extra                  # total = analytic + learned

    _t('7_residual')

    # Stage 7b -- far-field clamp. anc/gpu_ops.to_metric_depth bound inverse depth at
    # min_disp=1e-4, so a near-singular fit emits up to 10 000 m. ResidualRefiner.refine
    # already caps its own output, which meant the Network-B-OFF fallback was the only
    # unclamped path -- and it publishes straight into /cloud. Measured 2026-07-28 on the
    # island protocol: clamping moves closed-form MAE 18.092 -> 0.359 m with the median
    # unchanged at 0.066, i.e. it removes a pure far-field tail and touches nothing else.
    # Applied before the blend so no 10 000 m value can be averaged into a good one.
    metric = np.clip(metric, None, MAX_DEPTH_M)

    _t('7b_clamp')

    # Stage 7c -- distance-weighted blend of raw ToF against the network (see blend.py).
    # Neither source wins everywhere: nearest-zone ToF is ~2x better under 3 deg from an
    # anchor, the network ~6x better past 15 deg. Measured on the 61-frame clean split,
    # blending beats BOTH (interpolation medAE 0.009 vs ToF 0.010 and network 0.091;
    # extrapolation 0.042 vs 0.108 and 0.045).
    if blend:
        Kf = np.asarray(K, np.float64).ravel()
        fx = Kf[0] if Kf.size == 4 else Kf.reshape(3, 3)[0, 0]
        metric, _ = blend_depth(metric, anchor_depth, anchor_mask, fx=float(fx),
                                near_deg=blend_near, far_deg=blend_far)

    _t('7c_blend')

    # Stage 7d -- tell the truth about depth OUTSIDE the ROI.
    #
    # The far-field under-read is a documented, deliberate scope decision (see roi.py:
    # the map serves the traversable area). What was NOT documented is that sigma does
    # not say so. Measured on ZJU-L5 against dense independent GT: at 10-20 m true depth
    # the error is ~12 m while sigma reports ~0.08 m -- 150x too small -- and sigma
    # *decreases* with range, because Var[D] = D^4 * j^T Cov j and D is itself capped by
    # the 1/b ceiling. corr(sigma,|error|) 0.196, coverage at 1-sigma 0.229 vs an ideal
    # 0.683. So the system was most confident exactly where it was most wrong, and any
    # consumer of /depth_var or /cloud would have believed it.
    #
    # Being out of scope is fine; publishing out-of-scope depth with a confident sigma is
    # not. Outside the ROI we have no calibrated basis for sigma at all, so floor it at
    # ROI_OUTSIDE_SIGMA_FRAC * D -- a "could be anywhere" signal rather than a number.
    # The mask is returned so consumers can drop those points instead of trusting them.
    roi_mask = None
    if roi_enable and plane is not None:
        # On GPU, keep the mask STRIDED and expand it device-side (see
        # gpu_ops.roi_sigma_floor_lowres). Expanding on the CPU built a 2 MP bool array
        # purely to copy it across, and profiled at 23.5 ms/frame on the robot.
        roi_mask = roi.pixel_roi_mask(metric, K, plane, reach_max=roi_reach_max,
                                      height_max=roi_height_max, stride=roi_mask_stride,
                                      expand=not gpu)
        if var is not None:
            if gpu:
                var = gpu_ops.roi_sigma_floor_lowres(var, metric, roi_mask,
                                                     roi_mask_stride,
                                                     ROI_OUTSIDE_SIGMA_FRAC)
            else:
                floor_var = (ROI_OUTSIDE_SIGMA_FRAC * metric.astype(np.float32)) ** 2
                var = np.where(roi_mask, var, np.maximum(var, floor_var)).astype(np.float32)

    _t('7d_roi_sigma')

    # Stage 8 -- unproject metric depth to a camera-frame point cloud
    cloud = (gpu_ops.unproject_cloud(metric, K, cloud_stride) if gpu
             else geo.unproject_depth_to_cloud(metric, K, model='pinhole', stride=cloud_stride))

    _t('8_cloud')

    return {'ok': True, 'n_anchors': int(inb.sum()),
            'metric': metric.astype(np.float32),
            'var': None if var is None else var.astype(np.float32),
            'cloud': cloud, 'a': a, 'b': b,
            # roi_mask is STRIDED on the GPU path and full-resolution on the numpy one --
            # expanding it CPU-side purely to return it would reintroduce the 2 MP
            # allocation this stage was optimised to avoid. roi_mask_stride says which:
            # 1 means full resolution, N means each mask cell covers an NxN pixel block, so
            # a consumer wanting full resolution does
            #     np.repeat(np.repeat(m, N, 0), N, 1)[:h, :w]
            'roi_mask': roi_mask, 'roi_mask_stride': (roi_mask_stride if
                                                      (roi_mask is not None and gpu and
                                                       roi_mask_stride > 1) else 1),
            'plane': plane}

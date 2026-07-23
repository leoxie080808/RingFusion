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
        confidence=None, min_confidence=-1, cloud_stride=4, use_gpu=None):
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

    # Stage 2 -- backbone relative disparity
    disp = backbone.infer(rgb)

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

    # Stage 7 -- residual refinement (identity if residual is None/mock). Stays numpy;
    # gpu_ops returns numpy, so Network B sees exactly the types it did before.
    if residual is not None:
        anchor_depth, anchor_mask = splat_anchors(u, v, z, inb, (h, w))
        metric, var_extra = residual.refine(rgb, metric, disp,
                                             anchor_depth, anchor_mask, a, b)
        if var is not None and var_extra is not None:
            var = var + var_extra                  # total = analytic + learned

    # Stage 8 -- unproject metric depth to a camera-frame point cloud
    cloud = (gpu_ops.unproject_cloud(metric, K, cloud_stride) if gpu
             else geo.unproject_depth_to_cloud(metric, K, model='pinhole', stride=cloud_stride))

    return {'ok': True, 'n_anchors': int(inb.sum()),
            'metric': metric.astype(np.float32),
            'var': None if var is None else var.astype(np.float32),
            'cloud': cloud, 'a': a, 'b': b}

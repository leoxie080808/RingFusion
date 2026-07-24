"""Bridge from training code to the deployed perception math.

Residual training must produce the residual net's inputs (D0, analytic variance,
anchor channels, a, b) with EXACTLY the geometry + anchoring the robot runs at
inference, or the net learns to correct a fit it will never see. Rather than
re-derive that math here, we import the real modules from the ROS package
(`ringfusion_perception.geometry` / `.anchoring` / `.pipeline` -- all pure numpy)
so there is a single source of truth.

Also provides `simulate_tof`: sample a rendered depth map on the ToF zone grid
with dToF-style noise, so the residual can be pretrained on synthetic data before
any real ground truth exists (§5.3).
"""
import os
import sys

import numpy as np

# Import the deployed perception math (pure numpy; no ROS/CUDA needed).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_perception')
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
from ringfusion_perception import geometry as geo   # noqa: E402
from ringfusion_perception import anchoring as anc   # noqa: E402
from ringfusion_perception import pipeline           # noqa: E402


def default_calib(h=288, w=384, hfov_deg=90.0, tof_fov=(61.0, 45.0)):
    """A plausible pinhole calib at the training resolution, for synthetic data.
    Replace with the real calibration.yaml values when training on real frames."""
    fx = fy = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return {
        'K': (fx, fy, w / 2.0, h / 2.0),
        'dist': np.zeros(4),
        'model': 'pinhole',
        'T_cam_tof': geo.make_T_cam_tof([0.0, 20.195, 1.13], [0.0, 0.0, 0.0]),
        'img_w': w, 'img_h': h,
        'fov_h': tof_fov[0], 'fov_v': tof_fov[1],
    }


def calib_from_yaml(yaml_path, train_size=(288, 384)):
    """Real calib for REAL-frame residual training, matching what the robot runs.

    The deployed perception node rectifies fisheye->pinhole and runs everything on
    the RECTIFIED image with K_rect (perception_node.load_calib + FisheyeRectifier).
    So the captured training RGB must be that rectified image, and here we rebuild
    the SAME K_rect from calibration.yaml and scale it from capture resolution down
    to the training resolution. ToF FOV + extrinsic come straight from the yaml.

    train_size is (H, W) to match ResidualDepthDataset / args.size.
    """
    import yaml
    from ringfusion_perception.rectify import FisheyeRectifier    # noqa: E402
    with open(yaml_path) as f:
        c = yaml.safe_load(f)
    cam, rect, tof, ex = c['camera'], c.get('rectify', {}), c['tof'], c['extrinsics_cam_tof']
    K = (cam['fx'], cam['fy'], cam['cx'], cam['cy'])
    size_in = (cam['width'], cam['height'])
    size_out = (rect.get('width', cam['width']), rect.get('height', cam['height']))
    rectifier = FisheyeRectifier(K, np.asarray(cam['dist'], float), cam.get('model', 'fisheye'),
                                 size_in, size_out,
                                 balance=rect.get('balance', 0.0),
                                 fov_scale=rect.get('fov_scale', 1.0))
    fx, fy, cx, cy = rectifier.K_rect                 # pinhole intrinsics at capture res
    w_cap, h_cap = rectifier.size
    ht, wt = train_size
    sx, sy = wt / float(w_cap), ht / float(h_cap)     # scale K to the training resolution
    return {
        'K': (fx * sx, fy * sy, cx * sx, cy * sy),
        'dist': np.zeros(4),
        'model': 'pinhole',                           # rectified image -> pinhole, like the robot
        'T_cam_tof': geo.make_T_cam_tof(ex['translation_mm'], ex['rotation_rpy_deg']),
        'img_w': wt, 'img_h': ht,
        'fov_h': tof['fov_h_deg'], 'fov_v': tof['fov_v_deg'],
    }


def build_real_supervision(disp, tof_dist_m, tof_valid, calib, holdout_frac=0.25,
                           rng=None, confidence=None, min_confidence=-1, min_anchors=16,
                           min_range=0.15, max_range=5.0):
    """Held-out-anchor supervision from ONE real (disp, ToF) sample.

    A sparse sensor can't be a dense GT: its zones coincide with the anchors the
    closed-form fit already nails, so supervising there teaches the identity. Instead
    we split the frame's valid ToF zones into two disjoint sets:

      * ANCHOR set  -> drives the fit AND feeds the net's anchor channels (exactly
                       what the robot does at inference), via build_residual_inputs.
      * HOLD-OUT set -> fed to nobody as input; splatted into a sparse target so the
                       loss scores the net at pixels where it had a real measurement
                       but NO input -> true generalisation + real error stats for the
                       NLL variance head (the one thing synthetic can't give).

    Returns the build_residual_inputs dict plus D_gt (HxW, metric depth at held-out
    pixels, 0 else), valid_gt (HxW bool), and n_holdout. None if underdetermined.
    """
    rng = rng or np.random.default_rng()
    h, w = disp.shape
    rows, cols = tof_dist_m.shape
    K = calib['K']

    proj = geo.project_zone_to_pixel(
        tof_dist_m, tof_valid, cols, rows, calib['fov_h'], calib['fov_v'],
        calib['T_cam_tof'], K, calib['dist'], model=calib['model'])
    uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']
    finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
    u = np.round(np.where(finite, uv[:, 0], -1)).astype(int)
    v = np.round(np.where(finite, uv[:, 1], -1)).astype(int)
    inb = ok & finite & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    # Range gate: drop near-noise (< min_range) and far-unreliable (> max_range) ToF
    # zones -- both from anchoring AND supervision, so bad far/near readings can't be
    # spurious ground truth. TMF8829 accuracy degrades with range.
    inb = inb & (z >= min_range) & (z <= max_range)
    # Confidence gate: reject weak/low-SNR zones (multipath, glancing surfaces).
    if confidence is not None and min_confidence >= 0:
        inb = inb & (np.asarray(confidence, np.float32).reshape(-1) >= min_confidence)

    idx = np.flatnonzero(inb)
    if idx.size < 2 * min_anchors:                    # need enough for both sets
        return None
    rng.shuffle(idx)
    n_hold = max(1, int(round(idx.size * holdout_frac)))
    hold_idx, anchor_idx = idx[:n_hold], idx[n_hold:]
    if anchor_idx.size < min_anchors:
        return None

    # Anchor-only validity mask fed back through the SAME inference math.
    va = np.zeros(rows * cols, bool)
    va[anchor_idx] = True
    info = build_residual_inputs(disp, tof_dist_m, va.reshape(rows, cols),
                                 calib, confidence=confidence, min_confidence=min_confidence)
    if info is None or info['var_analytic'] is None:
        return None

    # Sparse hold-out target: metric depth only at the held-out zone pixels.
    D_gt = np.zeros((h, w), np.float32)
    valid_gt = np.zeros((h, w), bool)
    D_gt[v[hold_idx], u[hold_idx]] = z[hold_idx].astype(np.float32)
    valid_gt[v[hold_idx], u[hold_idx]] = True
    info['D_gt'], info['valid_gt'], info['n_holdout'] = D_gt, valid_gt, int(hold_idx.size)
    return info


def simulate_tof(gt_depth, calib, rows=32, cols=48, noise_frac=0.02,
                 dropout=0.05, rng=None):
    """Sample a dense metric depth map on the ToF zone grid.

    gt_depth is optical-axis depth (z), metres, at the calib resolution. Each zone
    ray is projected to a pixel (pinhole, parallax ignored -- the 20 mm baseline is
    negligible at ToF range), the depth is read there and converted to slant range
    rho = z / dir_z so the pipeline's geometry recovers z_cam correctly. dToF noise
    grows with distance (sigma proportional to range); some zones drop out.

    Returns (dist_m (rows,cols) float32 NaN where invalid, valid bool, conf uint8).
    """
    if rng is None:
        rng = np.random.default_rng()
    assert calib['model'] == 'pinhole', "simulate_tof assumes pinhole (rendered) frames"
    fx, fy, cx, cy = calib['K']
    h, w = gt_depth.shape

    dirs = geo.zone_directions(cols, rows, calib['fov_h'], calib['fov_v'])   # (rows,cols,3)
    dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    u = np.round(fx * dx / dz + cx).astype(int)
    v = np.round(fy * dy / dz + cy).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    z = np.full((rows, cols), np.nan, np.float32)
    z[inb] = gt_depth[v[inb], u[inb]]
    valid = inb & np.isfinite(z) & (z > 0)

    rho = np.where(valid, z / dz, np.nan)                    # slant range
    noise = rng.normal(0.0, noise_frac, size=rho.shape).astype(np.float32)
    rho = rho * (1.0 + noise)                                # sigma grows with range

    drop = rng.random(rho.shape) < dropout
    valid = valid & ~drop
    dist_m = np.where(valid, rho, np.nan).astype(np.float32)

    # crude confidence: nearer + valid -> stronger (matches firmware's small ints)
    conf = np.clip(60.0 - 15.0 * np.nan_to_num(rho), 5, 63).astype(np.uint8)
    conf[~valid] = 0
    return dist_m, valid, conf


def build_residual_inputs(disp, tof_dist_m, tof_valid, calib,
                          confidence=None, min_confidence=-1):
    """Run pipeline stages 3-6 for one sample and return the residual net's inputs.

    Mirrors pipeline.run exactly (same functions), minus the backbone and the
    final unprojection. Returns None if the fit is underdetermined.

    Returns dict: a, b, D0 (HxW), var_analytic (HxW or None),
    anchor_depth (HxW), anchor_mask (HxW), n_anchors.
    """
    h, w = disp.shape
    K = calib['K']
    rows, cols = tof_dist_m.shape

    proj = geo.project_zone_to_pixel(
        tof_dist_m, tof_valid, cols, rows, calib['fov_h'], calib['fov_v'],
        calib['T_cam_tof'], K, calib['dist'], model=calib['model'])
    uv, z, ok = proj['uv'], proj['z_cam'], proj['valid']

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
    inv_depth = 1.0 / z[inb]
    weights = np.maximum(conf_flat[inb], 1.0) if conf_flat is not None else np.ones_like(inv_depth)

    fit = anc.solve_robust(disp_at, inv_depth, weights, iters=1)
    if fit is None:
        return None
    a, b = fit
    D0 = anc.to_metric_depth(disp, a, b)

    var = None
    cov = anc.covariance(disp_at, inv_depth, weights, a, b)
    if cov is not None:
        j0, j1 = disp, np.ones_like(disp)
        quad = (j0 * (cov[0, 0] * j0 + cov[0, 1] * j1) +
                j1 * (cov[1, 0] * j0 + cov[1, 1] * j1))
        var = ((D0.astype(np.float64) ** 4) * quad).astype(np.float32)

    anchor_depth, anchor_mask = pipeline.splat_anchors(u, v, z, inb, (h, w))
    return {'a': float(a), 'b': float(b), 'D0': D0, 'var_analytic': var,
            'anchor_depth': anchor_depth, 'anchor_mask': anchor_mask,
            'n_anchors': int(inb.sum())}

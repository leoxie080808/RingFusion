"""Optional GPU (torch) acceleration for the heavy per-pixel pipeline tail.

Profiling on the Orin showed the bottleneck is NOT the neural net (backbone ~17 ms,
GPU idle) but the pure-numpy 2 MP math: analytic variance (depth**4, ~74 ms), cloud
unprojection (~43 ms), and metric depth (~13 ms). Those are embarrassingly parallel,
so they belong on the GPU that's otherwise sitting at ~0%.

Each function takes numpy in / numpy out (self-contained H2D/D2H) so the pipeline stays
numpy-typed and the residual (Network B) path is unchanged. If torch/CUDA is missing
(dev PC, CI), `available()` is False and the pipeline uses its numpy path -- so the
"pure-numpy, testable anywhere" property is preserved.
"""
import numpy as np

try:
    import torch
    _CUDA = bool(torch.cuda.is_available())
except Exception:                       # noqa: BLE001  (no torch / no CUDA -> numpy path)
    torch = None
    _CUDA = False


def available():
    return _CUDA


def _to_cuda(a, dtype):
    return torch.from_numpy(np.ascontiguousarray(a, dtype)).cuda(non_blocking=True)


def to_metric_depth(disp, a, b, min_disp=1e-4):
    """D0 = 1 / clamp(a*disp + b). Mirrors anchoring.to_metric_depth in fp32 on GPU."""
    d = _to_cuda(disp, np.float32)
    inv = torch.clamp(float(a) * d + float(b), min=min_disp)
    return (1.0 / inv).cpu().numpy()


def analytic_variance(disp, metric, cov):
    """Var[D](p) = D^4 * jᵀ Cov j,  j = (disp, 1).  cov is a 2x2 numpy array."""
    d = _to_cuda(disp, np.float32)
    m = _to_cuda(metric, np.float32)
    c00, c01, c10, c11 = (float(cov[0, 0]), float(cov[0, 1]),
                          float(cov[1, 0]), float(cov[1, 1]))
    j1 = torch.ones_like(d)
    quad = d * (c00 * d + c01 * j1) + j1 * (c10 * d + c11 * j1)
    return ((m ** 4) * quad).cpu().numpy()


def unproject_cloud(depth_m, K, stride=4):
    """Metric depth image -> (M,3) camera-frame cloud (pinhole), on GPU."""
    fx, fy, cx, cy = (float(v) for v in K)
    h0, w0 = depth_m.shape
    z = _to_cuda(depth_m, np.float32)[::stride, ::stride]
    h, w = z.shape
    vs = torch.arange(0, h0, stride, device="cuda", dtype=torch.float32)[:h]
    us = torch.arange(0, w0, stride, device="cuda", dtype=torch.float32)[:w]
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    pts = pts[pts[:, 2] > 0]
    return pts.cpu().numpy().astype(np.float32)


def residual_apply(disp, da_s, db_s, logtau_s, a, b, min_disp=1e-4, max_depth=20.0):
    """Network B's apply on GPU. Upsamples the 3 engine-resolution residual fields
    (da, db, log_tau2) to disp's full resolution and applies the affine + exp -- the
    2 MP per-pixel work that was ~70 ms on the CPU (bilinear resize + numpy affine).
    disp is full-res HxW; da_s/db_s/logtau_s are the engine-res fields. Returns
    (D metric HxW, tau2 HxW), both numpy -- same contract as the CPU path."""
    d = _to_cuda(disp, np.float32)
    h, w = d.shape
    fields = torch.stack([_to_cuda(da_s, np.float32),
                          _to_cuda(db_s, np.float32),
                          _to_cuda(logtau_s, np.float32)])[None]              # (1,3,h0,w0)
    up = torch.nn.functional.interpolate(fields, size=(h, w), mode="bilinear",
                                         align_corners=False)[0]              # (3,H,W)
    inv = torch.clamp((float(a) + up[0]) * d + (float(b) + up[1]), min=min_disp)
    # Cap the output depth: a degenerate residual drives inv->min_disp, giving D up to
    # 1/min_disp = 10 km. Clamp to a sane indoor ceiling so B never publishes a 10 km
    # pixel into /depth or the cloud on a bad frame (B's variance still flags it).
    D = torch.clamp(1.0 / inv, max=max_depth).cpu().numpy().astype(np.float32)
    tau2 = torch.exp(up[2].clamp(-8.0, 8.0)).cpu().numpy().astype(np.float32)  # clamp: no inf variance
    return D, tau2


def resplat_anchors(anchor_depth, anchor_mask, out_h, out_w):
    """Move sparse anchors from full res to engine res WITHOUT resizing.

    A nearest/bilinear resize of the sparse anchor maps drops ~95% of the ~1024 ToF
    anchors (2 MP -> 288x384 keeps only ~5%). But Network B was TRAINED with anchors
    splatted directly at engine res (build_residual_inputs uses disp.shape == 288x384,
    so ~all 1024 survive) -- so resizing here is a train/deploy domain shift that
    starves the net's anchor channels and leaves it under-confident + jittery on-robot.
    Re-project each nonzero pixel's coords onto the engine grid instead, keeping them
    all. Returns (anchor_depth, anchor_mask) at (out_h, out_w), numpy float32."""
    am = np.asarray(anchor_mask, np.float32)
    ad = np.asarray(anchor_depth, np.float32)
    h, w = am.shape
    ys, xs = np.nonzero(am > 0)
    oy = np.minimum((ys.astype(np.float32) * out_h / h).astype(np.int64), out_h - 1)
    ox = np.minimum((xs.astype(np.float32) * out_w / w).astype(np.int64), out_w - 1)
    od = np.zeros((out_h, out_w), np.float32)
    om = np.zeros((out_h, out_w), np.float32)
    od[oy, ox] = ad[ys, xs]
    om[oy, ox] = 1.0
    return od, om


def pack_residual_input(rgb, D0, anchor_depth, anchor_mask, out_h, out_w):
    """Build Network B's 6-channel engine input on GPU: resize rgb + logD0 to the engine
    resolution, ImageNet-normalize the rgb, nearest-resize the sparse anchor channels.
    Uses **bilinear, align_corners=False** to match the TRAINING pack (training/data._resize)
    -- more train-consistent than the old cv2.INTER_AREA CPU deploy, and on GPU. Returns
    (1,6,h,w) numpy."""
    F = torch.nn.functional
    rgb_t = _to_cuda(rgb, np.float32).permute(2, 0, 1)[None] / 255.0          # (1,3,H,W) RGB [0,1]
    rgb_s = F.interpolate(rgb_t, size=(out_h, out_w), mode="bilinear", align_corners=False)[0]
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(3, 1, 1)
    rgb_s = (rgb_s - mean) / std
    d = _to_cuda(D0, np.float32)
    pos = d[d > 0]
    med = pos.median() if pos.numel() else torch.tensor(1.0, device="cuda")
    logD0 = torch.log(torch.clamp(d, min=1e-3) / torch.clamp(med, min=1e-3))
    logD0 = F.interpolate(logD0[None, None], size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
    # Sparse anchors: re-splat at engine res (a resize would drop ~95% of them and
    # is a train/deploy domain shift -- see resplat_anchors).
    ad_np, am_np = resplat_anchors(anchor_depth, anchor_mask, out_h, out_w)
    ad = _to_cuda(ad_np, np.float32)
    am = _to_cuda(am_np, np.float32)
    return torch.stack([rgb_s[0], rgb_s[1], rgb_s[2], logD0, ad, am], 0)[None].cpu().numpy().astype(np.float32)


def blend_apply(D_net, dist_px, D_tof, near_px, far_px):
    """GPU tail for blend.blend_depth -- the smoothstep and the mix, over the full frame.

    The distance transform itself stays on the CPU (cv2, and it is cheap once computed at
    1/BLEND_SCALE resolution). What was expensive is everything after it: ~8 full-frame
    2 MP float passes for the ramp, the guard and the two-way mix. Measured on the Orin at
    1640x1232 the whole CPU blend cost 56 ms, 33 ms after the scale fix -- against a 51.8 ms
    pipeline budget. Same situation, and same remedy, as ResidualRefiner.refine (~70 ms on
    the CPU before it was offloaded here).

    dist_px and D_tof arrive already upsampled to full resolution. Returns numpy so the
    caller's types are unchanged.
    """
    d = _to_cuda(dist_px, np.float32)
    tof = _to_cuda(D_tof, np.float32)
    net = _to_cuda(D_net, np.float32)
    span = max(float(far_px) - float(near_px), 1e-6)
    t = torch.clamp((d - float(near_px)) / span, 0.0, 1.0)
    w = 1.0 - t * t * (3.0 - 2.0 * t)                    # smoothstep, 1 near -> 0 far
    w = torch.where(tof > 0, w, torch.zeros_like(w))     # no opinion where the splat is empty
    out = w * tof + (1.0 - w) * net
    return out.cpu().numpy(), w.cpu().numpy()


def _upsample_nearest(t, scale, h, w):
    """Block-replicate `t` by an integer factor and crop to (h, w).

    Exactly equivalent to np.repeat(np.repeat(t, s, 0), s, 1)[:h, :w] -- for an integer
    scale factor, 'nearest' interpolation IS block replication. Verified elementwise
    against the numpy form in the unit tests.
    """
    up = torch.nn.functional.interpolate(t[None, None], scale_factor=float(scale),
                                         mode='nearest')[0, 0]
    return up[:h, :w]


def blend_apply_lowres(D_net, dist_r, D_tof_r, scale, near_px, far_px):
    """blend_apply, but taking the REDUCED-resolution distance/depth maps.

    blend_depth used to expand both maps to full resolution with np.repeat ON THE CPU and
    then hand two 2 MP arrays to blend_apply, which copied them to the GPU. Profiled on the
    robot that stage cost 23.2 ms/frame. The expansion is pure block replication, so doing it
    on the GPU instead means copying 1/scale^2 as much data (at scale 4: 126 k values instead
    of 2 M, a 16x reduction) and skipping two 2 MP CPU allocations entirely.

    Numerically identical to blend_apply on the expanded inputs -- see test_blend_lowres.
    """
    d_small = _to_cuda(dist_r, np.float32)
    tof_small = _to_cuda(D_tof_r, np.float32)
    net = _to_cuda(D_net, np.float32)
    h, w = net.shape
    s = int(scale)
    # dist_r is in REDUCED pixel units; scale to full-res pixels (was `dist_r * S` on CPU)
    d = _upsample_nearest(d_small * float(s), s, h, w)
    tof = _upsample_nearest(tof_small, s, h, w)
    span = max(float(far_px) - float(near_px), 1e-6)
    t = torch.clamp((d - float(near_px)) / span, 0.0, 1.0)
    wgt = 1.0 - t * t * (3.0 - 2.0 * t)                  # smoothstep, 1 near -> 0 far
    wgt = torch.where(tof > 0, wgt, torch.zeros_like(wgt))
    out = wgt * tof + (1.0 - wgt) * net
    return out.cpu().numpy(), wgt.cpu().numpy()


def upsample_var(var_r, scale, h, w):
    """Block-replicate a reduced-resolution variance map to full resolution.

    blend.sigma_support_var builds its terms on the 1/scale grid the distance transform
    already lives on; this is the only full-frame allocation it needs, and doing it here
    keeps it off the CPU for the same reason blend_apply_lowres does.
    """
    t = _to_cuda(var_r, np.float32)
    return _upsample_nearest(t, int(scale), int(h), int(w)).cpu().numpy()


def roi_mask_and_sigma_floor(var, metric, K, plane, reach_max, height_max, stride, frac):
    """Stage 7d end to end on the GPU: build the ROI mask AND apply the sigma floor.

    Replaces roi.pixel_roi_mask + roi_sigma_floor_lowres, which together profiled at 24.0 ms
    on the robot -- the largest remaining cost after the blend fix. Three things were wrong
    with doing it CPU-side:

      1. `np.asarray(depth, float)` promoted the WHOLE 2 MP float32 map to float64 -- a 16 MB
         allocation -- when only the ~31 k strided samples are ever read from it.
      2. backprojection and the height/reach geometry then ran in float64 on the CPU.
      3. `metric` and `var` were already being copied to the GPU for the floor anyway, so the
         mask was computed on one device using data that had to reach the other regardless.

    Here the strided subsample, the backprojection, the plane geometry, the mask and the floor
    all run on device in float32, from the single copy of metric/var the floor already needed.

    Returns (var_floored, mask_small) -- mask_small is the STRIDED mask, matching what
    roi.pixel_roi_mask(expand=False) returns, so the pipeline's return value is unchanged.
    """
    v = _to_cuda(var, np.float32)
    m = _to_cuda(metric, np.float32)
    h, w = m.shape

    fx, fy, cx, cy = (float(x) for x in (np.asarray(K).ravel() if np.asarray(K).size == 4
                                         else (K[0, 0], K[1, 1], K[0, 2], K[1, 2])))
    s = int(stride)
    z = m[::s, ::s]                                    # strided view, no full-size copy
    hs, ws = z.shape
    us = torch.arange(0, w, s, device='cuda', dtype=torch.float32)[:ws]
    vs = torch.arange(0, h, s, device='cuda', dtype=torch.float32)[:hs]
    x = (us[None, :] - cx) * z / fx
    y = (vs[:, None] - cy) * z / fy

    n, d = plane
    n0, n1, n2 = (float(n[0]), float(n[1]), float(n[2]))
    d = float(d)
    height = x * n0 + y * n1 + z * n2 + d              # signed distance above the floor
    # Project points and the camera origin onto the plane, then measure the distance there.
    # c_flat = -d * n, so (p_flat - c_flat) = p - height*n + d*n.
    ex = x - height * n0 + d * n0
    ey = y - height * n1 + d * n1
    ez = z - height * n2 + d * n2
    reach = torch.sqrt(ex * ex + ey * ey + ez * ez)

    inside_s = (reach <= float(reach_max)) & (height <= float(height_max)) & (z > 0)
    inside = _upsample_nearest(inside_s.to(torch.float32), s, h, w) > 0.5
    floor = (float(frac) * m) ** 2
    out = torch.where(inside, v, torch.maximum(v, floor))
    return out.cpu().numpy(), inside_s.cpu().numpy()


def roi_sigma_floor_lowres(var, metric, mask_small, stride, frac):
    """roi_sigma_floor, but taking the STRIDED mask instead of the expanded one.

    Same problem and same fix as blend_apply_lowres. roi.pixel_roi_mask computes the mask on
    a stride-8 grid and then np.repeats it to a full 2 MP bool array on the CPU purely so it
    can be copied to the GPU; profiled at 23.5 ms/frame on the robot. At stride 8 the strided
    mask is 63x smaller (31 k values against 2 M).
    """
    v = _to_cuda(var, np.float32)
    m = _to_cuda(metric, np.float32)
    h, w = v.shape
    small = torch.from_numpy(
        np.ascontiguousarray(mask_small, np.float32)).cuda(non_blocking=True)
    inside = _upsample_nearest(small, int(stride), h, w) > 0.5
    floor = (float(frac) * m) ** 2
    return torch.where(inside, v, torch.maximum(v, floor)).cpu().numpy()


def roi_sigma_floor(var, metric, roi_mask, frac):
    """GPU tail for Stage 7d: floor sigma outside the ROI at frac*D.

    Three 2 MP passes (square, maximum, where) that cost ~20 ms of the ROI stage's 41 ms.
    """
    v = _to_cuda(var, np.float32)
    m = _to_cuda(metric, np.float32)
    inside = torch.from_numpy(np.ascontiguousarray(roi_mask)).cuda(non_blocking=True)
    floor = (float(frac) * m) ** 2
    return torch.where(inside, v, torch.maximum(v, floor)).cpu().numpy()

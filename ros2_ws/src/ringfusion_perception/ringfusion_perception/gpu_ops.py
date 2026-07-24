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
    ad = F.interpolate(_to_cuda(anchor_depth, np.float32)[None, None], size=(out_h, out_w), mode="nearest")[0, 0]
    am = F.interpolate(_to_cuda(anchor_mask, np.float32)[None, None], size=(out_h, out_w), mode="nearest")[0, 0]
    return torch.stack([rgb_s[0], rgb_s[1], rgb_s[2], logD0, ad, am], 0)[None].cpu().numpy().astype(np.float32)

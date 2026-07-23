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

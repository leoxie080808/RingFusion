"""Network B -- the residual correction net (g_phi).

Stage 5 anchors the whole image with two numbers (a, b), which assumes the
backbone's error is spatially uniform. It very nearly is, but not exactly: error
concentrates at depth edges, thin structures, and the image periphery far from
any ToF anchor. g_phi predicts a per-pixel correction to (a, b) plus an extra
variance term:

    D(p) = 1 / [ (a + da(p)) * disp(p) + (b + db(p)) ]
    var_total(p) = var_analytic(p) + tau2(p)

The correction lives in the SAME affine parameter space as the closed-form fit,
so the two stages compose into one estimator. Because the net's final conv is
zero-initialized (da = db = 0), an untrained residual is exactly the identity --
which is what MockResidual reproduces so the pipeline runs and ships before
g_phi is trained. Swap in ResidualRefiner once the .engine exists.

Contract for both:
    refine(rgb, D0, disp, anchor_depth, anchor_mask, a, b)
        -> (depth_float32_HxW, extra_variance_float32_HxW)
"""
import numpy as np

from . import gpu_ops

# Inference ceiling on B's output depth. A degenerate residual can drive the
# inverse-depth to its 1e-4 floor -> D up to 10 km; cap it to a sane indoor value
# so B never emits a 10 km pixel into /depth or the cloud on a bad frame.
MAX_DEPTH_M = 20.0


class MockResidual:
    """Identity. da = db = 0, no extra variance -> the estimator reduces exactly
    to the closed-form fit. Lets the whole pipeline run before g_phi is trained."""
    name = "mock"

    def refine(self, rgb, D0, disp, anchor_depth, anchor_mask, a, b):
        D0 = np.asarray(D0, np.float32)
        return D0, np.zeros_like(D0)


class ResidualRefiner:
    """TensorRT g_phi. Runs at a fixed engine resolution, then upsamples the
    (da, db, log_tau2) fields to full image resolution and applies them to the
    full-resolution disparity -- same upsample pattern as the backbone.

    Build this engine at FP16, not INT8: it is small enough that INT8 buys little
    and quantizing the variance head degrades calibration (see the tech ref)."""
    name = "residual_trt"

    def __init__(self, engine_path, input_hw=(288, 384)):
        from .trt_util import TRTRunner
        self.h, self.w = input_hw
        self._engine = TRTRunner(
            engine_path,
            in_shape=(1, 6, self.h, self.w),     # rgb(3) + logD0 + anchor_depth + mask
            out_shape=(1, 3, self.h, self.w))    # da, db, log_tau2

    def _pack(self, rgb, D0, anchor_depth, anchor_mask):
        """Assemble the 6-channel input at engine resolution.

        Uses the GPU pack (gpu_ops.pack_residual_input) when CUDA is present: it's
        ~3x faster AND uses bilinear/align_corners=False, which byte-matches the
        TRAINING pack (training/data._resize) -- the old cv2.INTER_AREA path here was
        off from training by up to ~2.7 on the normalized rgb, a real domain shift.
        The numpy fallback (below) keeps off-robot behaviour working."""
        if gpu_ops.available():
            return gpu_ops.pack_residual_input(rgb, D0, anchor_depth, anchor_mask, self.h, self.w)
        import cv2
        img = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        img = (img - mean) / std
        rgb_c = img.transpose(2, 0, 1)                       # (3,h,w)

        D0 = np.asarray(D0, np.float32)
        med = float(np.median(D0[D0 > 0])) if np.any(D0 > 0) else 1.0
        logD0 = np.log(np.clip(D0, 1e-3, None) / max(med, 1e-3))
        logD0 = cv2.resize(logD0, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # Anchors are sparse: nearest-neighbour keeps the few non-zero pixels
        # instead of averaging them away.
        ad = cv2.resize(np.asarray(anchor_depth, np.float32), (self.w, self.h),
                        interpolation=cv2.INTER_NEAREST)
        am = cv2.resize(np.asarray(anchor_mask, np.float32), (self.w, self.h),
                        interpolation=cv2.INTER_NEAREST)

        x = np.stack([rgb_c[0], rgb_c[1], rgb_c[2], logD0, ad, am], axis=0)
        return x[None]                                       # (1,6,h,w)

    def refine(self, rgb, D0, disp, anchor_depth, anchor_mask, a, b):
        out = self._engine.run(self._pack(rgb, D0, anchor_depth, anchor_mask))
        # Upsample the 3 fields to full res + apply the affine over the whole 2 MP.
        # On GPU when CUDA is present (was ~70 ms on the CPU -> halved the pipeline
        # rate); numpy fallback keeps off-robot behaviour identical.
        if gpu_ops.available():
            return gpu_ops.residual_apply(disp, out[0, 0], out[0, 1], out[0, 2], a, b)
        import cv2
        H, W = disp.shape
        da = cv2.resize(out[0, 0], (W, H), interpolation=cv2.INTER_LINEAR)
        db = cv2.resize(out[0, 1], (W, H), interpolation=cv2.INTER_LINEAR)
        log_tau2 = cv2.resize(out[0, 2], (W, H), interpolation=cv2.INTER_LINEAR)
        inv = (a + da) * disp + (b + db)
        D = np.clip(1.0 / np.clip(inv, 1e-4, None), None, MAX_DEPTH_M)   # cap 10 km degenerates
        return D.astype(np.float32), np.exp(np.clip(log_tau2, -8.0, 8.0)).astype(np.float32)

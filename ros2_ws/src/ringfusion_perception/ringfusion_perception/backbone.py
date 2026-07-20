"""Network A -- the monocular relative-depth backbone. Returns a relative
*disparity* map (inverse depth, up to a global scale/shift) from an RGB image.
Disparity rather than depth because the scale ambiguity is then affine, which
makes the anchoring stage linear.

MockBackbone lets the whole pipeline run and lets you validate the geometry and
anchoring today on a dev PC. Swap TensorRTBackbone in once the distilled student
(Depth Anything V2 -> MobileNetV3+DPT-lite) is exported to a .engine on the
Jetson. The infer(rgb) -> disparity contract is identical, so it is a one-line
change in perception_node and nothing downstream is affected.
"""
import numpy as np


class MockBackbone:
    """Synthetic disparity: nearer toward image bottom, plus mild texture.
    Physically arbitrary, but well-conditioned for the scale-shift fit."""
    name = "mock"

    def infer(self, rgb):
        h, w = rgb.shape[:2]
        v = np.linspace(0.2, 1.0, h, dtype=np.float32)[:, None]
        base = np.repeat(v, w, axis=1)
        tex = rgb.mean(axis=2).astype(np.float32) / 255.0
        disp = base + 0.05 * tex + np.random.normal(0, 0.005, (h, w)).astype(np.float32)
        return np.clip(disp, 1e-3, None)


class TensorRTBackbone:
    """Distilled Depth Anything V2 student, INT8 TensorRT. Runs at a fixed engine
    resolution, then upsamples disparity back to the source image size.

    tensorrt/pycuda/cv2 are Jetson-only and imported lazily, so importing this
    module on a dev PC is fine; only instantiating this class needs the runtime."""
    name = "tensorrt"

    def __init__(self, engine_path, input_hw=(288, 384), batch=1):
        from .trt_util import TRTRunner
        self.h, self.w = input_hw
        self._engine = TRTRunner(
            engine_path,
            in_shape=(batch, 3, self.h, self.w),
            out_shape=(batch, 1, self.h, self.w))

    def _preprocess(self, rgb):
        import cv2
        img = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        img = (img - mean) / std
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

    def infer(self, rgb):
        """rgb: HxWx3 uint8 -> disparity at the ORIGINAL image size."""
        import cv2
        out = self._engine.run(self._preprocess(rgb))
        disp = out[0, 0]
        disp = cv2.resize(disp, (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        return np.clip(disp, 1e-3, None)

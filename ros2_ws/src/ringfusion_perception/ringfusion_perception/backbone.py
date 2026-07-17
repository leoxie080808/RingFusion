"""Monocular relative-depth backbone. Returns a relative *disparity* map
(inverse depth, up to global scale/shift) from an RGB image.

MockBackbone lets the whole pipeline run and lets you validate the geometry and
anchoring today. Swap TensorRTBackbone in once the distilled student is exported.
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
    """Load a distilled depth engine (Depth Anything V2 student) and run INT8.
    TODO: implement with tensorrt + pycuda once the .engine exists."""
    name = "tensorrt"

    def __init__(self, engine_path):
        raise NotImplementedError(
            "Export the distilled student to a TensorRT .engine and implement "
            "load/infer here. Use MockBackbone until then.")

    def infer(self, rgb):
        raise NotImplementedError

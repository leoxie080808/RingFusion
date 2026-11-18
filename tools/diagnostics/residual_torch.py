#!/usr/bin/env python3
"""Score a residual refiner from its TRAINING CHECKPOINT (.pth), not a TensorRT engine.

WHY THIS EXISTS. T9 produced six .pth checkpoints; ResidualRefiner only loads .engine files.
The alternative was converting all six, which would fold six separate ONNX->TensorRT FP16
quantisations into a comparison whose whole point is to isolate the effect of the hold-out
protocol. T10 already showed that quantisation is worth ~0.0001 in rho, so it would be noise
on top of the signal rather than signal -- but it would be SIX DIFFERENT noises, one per arm,
and that is exactly the confound to avoid.

HOW IT STAYS FAITHFUL. Only network execution is swapped. This subclasses ResidualRefiner and
replaces `self._engine` with an object exposing the same `.run(x) -> (1,3,h,w) float32`, so
the deployed `_pack` (including the GPU bilinear pack and anchor re-splat) and the deployed
`refine` (upsample, affine, tau2 clip) both run completely unchanged. Anything this measures
that differs between two checkpoints is a difference in the WEIGHTS.
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'ringfusion_perception'))
sys.path.insert(0, os.path.join(_REPO, 'training'))

from ringfusion_perception.residual import ResidualRefiner      # noqa: E402


class _TorchEngine:
    """Duck-types trt_util.TRTRunner.run for a torch module."""

    def __init__(self, ckpt_path, device=None):
        import torch
        from models.residual import ResidualRefinerNet
        self._torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        net = ResidualRefinerNet()
        sd = torch.load(ckpt_path, map_location='cpu')
        # Checkpoints are saved either bare or wrapped; accept both rather than guessing.
        for key in ('model', 'state_dict', 'residual'):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f'{os.path.basename(ckpt_path)}: state_dict mismatch\n'
                             f'  missing: {list(missing)[:6]}\n'
                             f'  unexpected: {list(unexpected)[:6]}')
        self.net = net.to(self.device).eval()

    def run(self, x):
        t = self._torch
        with t.no_grad():
            out = self.net(t.from_numpy(np.ascontiguousarray(x)).to(self.device))
        return out.detach().float().cpu().numpy()


class TorchResidualRefiner(ResidualRefiner):
    """ResidualRefiner with the TensorRT engine swapped for the torch checkpoint."""

    name = 'residual_torch'

    def __init__(self, ckpt_path, input_hw=(288, 384)):
        self.h, self.w = input_hw
        self._engine = _TorchEngine(ckpt_path)          # no super().__init__: that loads TRT


if __name__ == '__main__':
    r = TorchResidualRefiner(sys.argv[1])
    rgb = np.zeros((1232, 1640, 3), np.uint8)
    D0 = np.full((1232, 1640), 2.0, np.float32)
    disp = np.full((1232, 1640), 0.5, np.float32)
    ad = np.zeros((1232, 1640), np.float32); am = np.zeros((1232, 1640), np.float32)
    ad[::40, ::40] = 2.0; am[::40, ::40] = 1.0
    D, tau2 = r.refine(rgb, D0, disp, ad, am, 1.0, 0.1)
    print(f'ok: D {D.shape} {D.dtype} median {np.median(D):.3f} m, '
          f'tau2 median {np.median(tau2):.5f}')

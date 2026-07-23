"""Dataset for REAL residual training: rectified RGB paired with a raw 32x32 ToF
frame (NOT dense GT). The held-out-anchor split in training turns the sparse ToF
into both the net's anchor input and a sparse supervision target -- see
anchoring_bridge.build_real_supervision.

Unlike ResidualDepthDataset (synthetic dense depth + simulated ToF), this needs no
dense ground truth: the ToF sensor IS the measurement, and no synthetic->real
domain gap, because the RGB is the same rectified fisheye the robot feeds the net.

Expected on-disk layout (produced by the paired ToF+image logger), matched by stem:
    rgb_dir/<...>/000123.png      # the RECTIFIED (pinhole) camera image, as deployed
    tof_dir/<...>/000123.npz      # np.savez(dist_m=(32,32) float32 NaN-invalid,
                                  #          confidence=(32,32) uint8)
Capture stationary "stations" (stop, average a few ToF frames, snap one image) for
clean pairs, or push slowly and pair each ToF frame with the nearest-in-time image.
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from data import IMAGE_EXTS, normalize, load_image01, _resize


def _find_tof(tof_dir, stem):
    cands = glob.glob(os.path.join(tof_dir, '**', stem + '.npz'), recursive=True)
    return cands[0] if cands else None


class ResidualRealDataset(Dataset):
    """Yields (rgb_norm 3xHxW, tof_dist_m (rows,cols) float32 NaN-invalid,
    tof_conf (rows,cols) uint8). The ToF is kept at native resolution; projection
    and the anchor/hold-out split happen per-sample in training."""

    def __init__(self, rgb_dir, tof_dir, size=(288, 384)):
        self.rgb = []
        for ext in IMAGE_EXTS:
            self.rgb += glob.glob(os.path.join(rgb_dir, '**', ext), recursive=True)
        self.rgb = sorted(self.rgb)
        if not self.rgb:
            raise FileNotFoundError(f"no images under {rgb_dir}")
        self.tof_dir = tof_dir
        self.size = size

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        rgb_path = self.rgb[i]
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        tof_path = _find_tof(self.tof_dir, stem)
        if tof_path is None:
            raise FileNotFoundError(f"no ToF .npz for {rgb_path} (stem {stem})")
        d = np.load(tof_path)
        dist = np.asarray(d['dist_m'], np.float32)       # (rows,cols), NaN where no return
        conf = np.asarray(d['confidence'], np.uint8)

        img = _resize(load_image01(rgb_path), self.size, 'bilinear')   # (3,H,W)
        return normalize(img), torch.from_numpy(dist), torch.from_numpy(conf)

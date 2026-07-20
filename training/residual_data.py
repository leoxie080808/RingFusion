"""Dataset for residual training: RGB paired with dense metric ground-truth depth.

Start on synthetic renders (Hypersim / Replica / Blender), where per-pixel depth
is exact and free (§5.3), then fine-tune on whatever real GT is available. ToF is
*simulated* from the GT in the training loop (anchoring_bridge.simulate_tof), so
this dataset only needs (rgb, depth) pairs.

Expected layout: parallel directories with matching file stems, e.g.
    rgb_dir/scene0/0001.png   depth_dir/scene0/0001.npy
Depth is read as .npy (metres) or 16-bit PNG (millimetres, use --depth-scale 0.001).
"""
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from data import IMAGE_EXTS, normalize, load_image01, _resize


def _find_depth(depth_dir, stem):
    for ext in ('.npy', '.png', '.exr'):
        cands = glob.glob(os.path.join(depth_dir, '**', stem + ext), recursive=True)
        if cands:
            return cands[0]
    return None


def load_depth_m(path, scale):
    if path.endswith('.npy'):
        d = np.load(path).astype(np.float32)
    else:
        d = np.asarray(Image.open(path), dtype=np.float32) * scale
    return d


class ResidualDepthDataset(Dataset):
    """Yields (rgb_norm 3xHxW, gt_depth 1xHxW metres, valid 1xHxW bool)."""

    def __init__(self, rgb_dir, depth_dir, size=(288, 384), depth_scale=1.0,
                 max_depth=20.0):
        self.rgb = []
        for ext in IMAGE_EXTS:
            self.rgb += glob.glob(os.path.join(rgb_dir, '**', ext), recursive=True)
        self.rgb = sorted(self.rgb)
        if not self.rgb:
            raise FileNotFoundError(f"no images under {rgb_dir}")
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.size = size
        self.depth_scale = depth_scale
        self.max_depth = max_depth

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        rgb_path = self.rgb[i]
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path = _find_depth(self.depth_dir, stem)
        if depth_path is None:
            raise FileNotFoundError(f"no depth for {rgb_path} (stem {stem})")

        img = load_image01(rgb_path)                                   # (3,H0,W0)
        depth = torch.from_numpy(
            load_depth_m(depth_path, self.depth_scale)).unsqueeze(0)   # (1,H0,W0)

        img = _resize(img, self.size, 'bilinear')
        depth = _resize(depth, self.size, 'nearest')                   # don't blend edges
        valid = torch.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        depth = torch.nan_to_num(depth, nan=0.0)
        return normalize(img), depth, valid

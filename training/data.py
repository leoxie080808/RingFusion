"""Datasets and augmentation for backbone distillation.

The distillation target is the teacher's cached disparity, pixel-aligned with each
source image (see cache_teacher.py). Geometric augmentation is therefore applied
*identically* to image and target; photometric jitter only to the image. Vertical
flip is deliberately excluded -- monocular depth priors are gravity-dependent.
"""
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMAGE_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')


def list_images(root):
    files = []
    for ext in IMAGE_EXTS:
        files += glob.glob(os.path.join(root, '**', ext), recursive=True)
    return sorted(files)


def normalize(img01):
    """(3,H,W) in [0,1] -> ImageNet-normalized."""
    return (img01 - IMAGENET_MEAN) / IMAGENET_STD


def load_image01(path):
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)          # (3,H,W)


def _color_jitter(img, rng, b=0.2, c=0.2, s=0.2):
    """Lightweight brightness/contrast/saturation jitter on a (3,H,W) [0,1] image
    (no torchvision dependency)."""
    img = img * (1.0 + rng.uniform(-b, b))
    m = img.mean()
    img = (img - m) * (1.0 + rng.uniform(-c, c)) + m
    gray = img.mean(dim=0, keepdim=True)
    img = (img - gray) * (1.0 + rng.uniform(-s, s)) + gray
    return img.clamp(0.0, 1.0)


def _resize(t, size, mode):
    return torch.nn.functional.interpolate(
        t.unsqueeze(0), size=size,
        mode=mode, align_corners=None if mode == 'nearest' else False).squeeze(0)


class DistillDataset(Dataset):
    """Pairs each image with its cached teacher disparity (mirror path + .npy)."""

    def __init__(self, image_dir, cache_dir, size=(288, 384), augment=True,
                 crop_scale=(0.6, 1.0), seed=0, exclude_stems=None):
        # exclude_stems holds out frames from DISTILLATION so the benchmarks have a set
        # Network A never saw. Needed because list_images() globs '**' recursively, so
        # distilling over data/rect swept up data/rect/paired -- 1228 frames byte-identical
        # to ros2_ws/data/real/rgb, i.e. every frame the benchmarks score on. Rankings were
        # unaffected (all methods share the backbone) and the inflation is bounded (the
        # student scores BELOW its teacher on those frames, so it has not memorised them),
        # but absolute numbers were optimistic by an unquantified amount.
        self.images = list_images(image_dir)
        if exclude_stems:
            ex = set(exclude_stems)
            self.images = [p for p in self.images
                           if os.path.splitext(os.path.basename(p))[0] not in ex]
        if not self.images:
            raise FileNotFoundError(f"no images under {image_dir}")
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.size = size
        self.augment = augment
        self.crop_scale = crop_scale
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.images)

    def _cache_path(self, img_path):
        rel = os.path.relpath(img_path, self.image_dir)
        return os.path.join(self.cache_dir, os.path.splitext(rel)[0] + '.npy')

    def __getitem__(self, i):
        img_path = self.images[i]
        img = load_image01(img_path)                       # (3,H0,W0)
        target = torch.from_numpy(
            np.load(self._cache_path(img_path)).astype(np.float32)).unsqueeze(0)
        if target.shape[-2:] != img.shape[-2:]:            # keep them aligned
            target = _resize(target, img.shape[-2:], 'bilinear')

        if self.augment:
            img, target = self._augment(img, target)
        else:
            img = _resize(img, self.size, 'bilinear')
            target = _resize(target, self.size, 'bilinear')

        return normalize(img), target

    def _augment(self, img, target):
        _, h0, w0 = img.shape
        # random resized crop (shared box), then resize to output size
        scale = self.rng.uniform(*self.crop_scale)
        ch = max(1, int(round(h0 * scale)))
        cw = max(1, int(round(w0 * scale)))
        top = int(self.rng.integers(0, h0 - ch + 1))
        left = int(self.rng.integers(0, w0 - cw + 1))
        img = img[:, top:top + ch, left:left + cw]
        target = target[:, top:top + ch, left:left + cw]
        img = _resize(img, self.size, 'bilinear')
        target = _resize(target, self.size, 'bilinear')

        if self.rng.random() < 0.5:                        # horizontal flip only
            img = torch.flip(img, dims=[-1])
            target = torch.flip(target, dims=[-1])
        img = _color_jitter(img, self.rng)
        return img, target

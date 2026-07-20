"""Precompute Depth Anything V2 disparity targets and cache them to disk as fp16.

The teacher (~335M ViT-L) is the expensive part and never changes, so caching its
outputs converts a multi-day distillation into a few hours and lets you iterate on
the student freely (§3.5). Distillation needs images only -- no measured depth.

CRITICAL: run on RECTIFIED images. The teacher expects rectilinear input and
produces garbage on raw fisheye, and that garbage would become your target (§5.2).

Usage:
    python cache_teacher.py --images data/rect --cache data/teacher \
        --model depth-anything/Depth-Anything-V2-Large-hf
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import list_images


def load_teacher(model_id, device):
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.no_grad()
def infer_disparity(proc, model, image, out_hw, device):
    """Depth Anything predicts disparity-like inverse depth (larger = closer),
    which is exactly the student's target. Resized to the source image size so the
    cached target is pixel-aligned with the stored image."""
    inputs = proc(images=image, return_tensors='pt').to(device)
    pred = model(**inputs).predicted_depth        # (1, h, w)
    pred = F.interpolate(pred.unsqueeze(1).float(), size=out_hw,
                         mode='bilinear', align_corners=False)
    return pred[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='dir of rectified images')
    ap.add_argument('--cache', required=True, help='output dir for .npy targets')
    ap.add_argument('--model', default='depth-anything/Depth-Anything-V2-Large-hf')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=0, help='cap #images (0 = all)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    images = list_images(args.images)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"no images under {args.images}")
    print(f"caching {len(images)} teacher targets from {args.model} on {args.device}")

    proc, model = load_teacher(args.model, args.device)
    done = 0
    for i, path in enumerate(images):
        rel = os.path.relpath(path, args.images)
        out_path = os.path.join(args.cache, os.path.splitext(rel)[0] + '.npy')
        if os.path.exists(out_path) and not args.overwrite:
            continue
        image = Image.open(path).convert('RGB')
        disp = infer_disparity(proc, model, image, (image.height, image.width),
                               args.device)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, disp.astype(np.float16))
        done += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)}")
    print(f"done: wrote {done} new targets to {args.cache}")


if __name__ == '__main__':
    main()

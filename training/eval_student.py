"""Distillation-fidelity metrics: how well does the student mimic the teacher?

Reproduces the SAME held-out val split distill_backbone.py uses (random_split,
seed 0), then reports interpretable, scale-shift-invariant agreement metrics between
the student and the cached teacher disparity:

  rho     Pearson correlation (structure match, 0-1; >0.98 excellent)
  SSI-MAE scale-shift-invariant MAE (matches val_ssi from training)
  AbsRel  mean |student-teacher|/teacher after alignment (a fraction)
  d1.25   fraction of pixels agreeing with the teacher within 25%

NOTE: this measures fidelity TO THE TEACHER, not correctness of depth. Real accuracy
needs measured ground truth (residual training + system validation).

    python training/eval_student.py --ckpt runs/student/student_best.pth
"""
import argparse
import glob
import os

import numpy as np
import cv2
import torch

from data import load_image01, normalize
from models.student import DepthStudent


def align(s, t):
    """Least-squares scale+shift mapping s -> t (same alignment as training)."""
    A = np.stack([s.ravel(), np.ones(s.size, np.float64)], 1)
    ab, *_ = np.linalg.lstsq(A, t.ravel().astype(np.float64), rcond=None)
    return (ab[0] * s + ab[1]).astype(np.float32)


def val_indices(n, val_frac=0.05, seed=0):
    """Exactly reproduce torch random_split([n_train, n_val], seed)."""
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[n - n_val:]                      # val_set is the tail slice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/student/student_best.pth')
    ap.add_argument('--images', default='data/rect')
    ap.add_argument('--cache', default='data/teacher')
    ap.add_argument('--size', type=int, nargs=2, default=[288, 384])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    H, W = args.size

    model = DepthStudent(pretrained=False).to(args.device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    images = sorted(glob.glob(os.path.join(args.images, '**', '*.png'), recursive=True))
    idxs = val_indices(len(images))
    print(f"evaluating {len(idxs)} held-out frames (of {len(images)})")

    rhos, maes, absrels, d1s = [], [], [], []
    for i in idxs:
        path = images[i]
        img01 = load_image01(path)
        inp = torch.nn.functional.interpolate(
            normalize(img01).unsqueeze(0), size=(H, W),
            mode='bilinear', align_corners=False).to(args.device)
        with torch.no_grad():
            s = model(inp)[0, 0].cpu().numpy().astype(np.float32)

        rel = os.path.relpath(path, args.images)
        t = np.load(os.path.join(args.cache, os.path.splitext(rel)[0] + '.npy'))
        t = cv2.resize(t.astype(np.float32), (W, H))

        rhos.append(float(np.corrcoef(s.ravel(), t.ravel())[0, 1]))
        sa = align(s, t)
        maes.append(float(np.abs(sa - t).mean()))
        # ratio-based metrics belong in DEPTH space: disparity has near-zero far
        # values that blow up division. depth = 1/disparity (floored).
        eps = 1e-2
        sd = 1.0 / np.clip(sa, eps, None)
        td = 1.0 / np.clip(t, eps, None)
        absrels.append(float((np.abs(sd - td) / td).mean()))
        ratio = np.maximum(sd / td, td / sd)
        d1s.append(float((ratio < 1.25).mean()))

    print(f"  rho     {np.mean(rhos):.4f}   (Pearson corr; structure match)")
    print(f"  SSI-MAE {np.mean(maes):.4f}   (compare to training val_ssi)")
    print(f"  AbsRel  {np.mean(absrels):.4f}   ({100*np.mean(absrels):.1f}% mean rel. error vs teacher)")
    print(f"  d1.25   {np.mean(d1s):.4f}   ({100*np.mean(d1s):.1f}% of pixels within 25% of teacher)")


if __name__ == '__main__':
    main()

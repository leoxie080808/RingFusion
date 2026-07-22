"""Visual sanity check: student disparity vs the cached teacher disparity.

Renders a montage [ RGB | student | teacher ] for a handful of frames so you can
confirm the distilled student reproduces the teacher's structure (near/far right,
edges crisp, far field not flattened). Each disparity is normalized by its own 2/98
percentiles, so the comparison judges STRUCTURE, not absolute scale (which is exactly
what the scale-shift-invariant training optimizes, and what the ToF fixes at runtime).

    python training/compare_student.py --ckpt runs/student/student_best.pth \
        --images data/rect --cache data/teacher --out student_vs_teacher.png
"""
import argparse
import glob
import os

import numpy as np
import cv2
import torch

from data import load_image01, normalize
from models.student import DepthStudent


def colorize(disp):
    """(H,W) disparity -> (H,W,3) BGR heatmap, robust-normalized by 2/98 pct."""
    lo, hi = np.percentile(disp, 2), np.percentile(disp, 98)
    d = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/student/student_best.pth')
    ap.add_argument('--images', default='data/rect')
    ap.add_argument('--cache', default='data/teacher')
    ap.add_argument('--out', default='student_vs_teacher.png')
    ap.add_argument('--n', type=int, default=4, help='number of frames to show')
    ap.add_argument('--size', type=int, nargs=2, default=[288, 384])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    H, W = args.size

    model = DepthStudent(pretrained=False).to(args.device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    images = sorted(glob.glob(os.path.join(args.images, '**', '*.png'), recursive=True))
    if not images:
        raise SystemExit(f"no images under {args.images}")
    pick = np.linspace(0, len(images) - 1, args.n).astype(int)   # spread across the set

    rows = []
    for idx in pick:
        path = images[idx]
        img01 = load_image01(path)                                # (3,H0,W0) in [0,1]
        inp = torch.nn.functional.interpolate(
            normalize(img01).unsqueeze(0), size=(H, W),
            mode='bilinear', align_corners=False).to(args.device)
        with torch.no_grad():
            student = model(inp)[0, 0].cpu().numpy()              # (H,W)

        rel = os.path.relpath(path, args.images)
        teacher = np.load(os.path.join(args.cache, os.path.splitext(rel)[0] + '.npy'))
        teacher = cv2.resize(teacher.astype(np.float32), (W, H))

        rgb = cv2.cvtColor(
            (img01.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (W, H))

        row = np.hstack([rgb, colorize(student), colorize(teacher)])
        cv2.putText(row, os.path.basename(path), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)

    montage = np.vstack(rows)
    header = np.zeros((28, montage.shape[1], 3), np.uint8)
    for i, label in enumerate(('RGB', 'STUDENT', 'TEACHER')):
        cv2.putText(header, label, (i * W + 8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(args.out, np.vstack([header, montage]))
    print(f"wrote {args.out}  ({len(pick)} frames, columns: RGB | student | teacher)")


if __name__ == '__main__':
    main()

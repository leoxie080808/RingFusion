"""Step 0 -- sanity-check the distillation teacher on a real RECTIFIED frame.

The single cheapest de-risking step (technical reference Â§8, Step 0): before
spending days distilling, confirm Depth Anything V2 produces sane depth on OUR
rectified fisheye crop. Whatever ceiling the teacher has propagates straight to
the student, so if it looks bad here the plan changes before you invest in data
collection and training.

Runs the SAME teacher + inference that cache_teacher.py uses, so a good result
here means the cached distillation targets will be good too.

    pip install transformers            # one-time (on the Jetson: JetPack torch)
    # a frame straight off the camera is raw fisheye -> let it rectify first:
    python training/step0_sanity.py --image raw_frame.png --raw --out step0.png
    # or an already-rectified image:
    python training/step0_sanity.py --image rect_frame.png --out step0.png

Output: a side-by-side  [ rectified RGB | teacher inverse-depth (JET) ]  plus
printed disparity stats, so you can judge it at a glance. Near surfaces should be
warm (red/yellow), far ones cool (blue), edges crisp, no big smeared/flat blobs.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

# reuse the exact teacher load + inference from the caching script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_teacher import load_teacher, infer_disparity

DEFAULT_CALIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'ros2_ws', 'src', 'ringfusion_bringup', 'config', 'calibration.yaml')


def rectify_raw(bgr, calib_path):
    """Rectify a raw fisheye BGR frame with calibration.yaml. Inline (no ROS
    dependency) but mirrors ringfusion_perception/rectify.py exactly."""
    import cv2
    import yaml
    c = yaml.safe_load(open(calib_path))
    cam, rec = c['camera'], c.get('rectify', {})
    if cam.get('model') != 'fisheye':
        return bgr
    K = np.array([[cam['fx'], 0, cam['cx']],
                  [0, cam['fy'], cam['cy']], [0, 0, 1]], float)
    D = np.asarray(cam.get('dist', [0, 0, 0, 0]), float).reshape(4, 1)
    sz_in = (cam['width'], cam['height'])
    sz_out = (rec.get('width', cam['width']), rec.get('height', cam['height']))
    P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, sz_in, np.eye(3), balance=rec.get('balance', 0.0),
        new_size=sz_out, fov_scale=rec.get('fov_scale', 1.0))
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), P, sz_out, cv2.CV_16SC2)
    return cv2.remap(bgr, m1, m2, cv2.INTER_LINEAR)


def colorize(disp):
    """Inverse-depth -> JET (near = warm). 2-98 percentile stretch for contrast."""
    import cv2
    d = disp.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True, help='RGB frame (raw or rectified)')
    ap.add_argument('--raw', action='store_true',
                    help='input is raw fisheye; rectify with --calib first')
    ap.add_argument('--calib', default=DEFAULT_CALIB)
    ap.add_argument('--model', default='depth-anything/Depth-Anything-V2-Large-hf')
    ap.add_argument('--out', default='step0_sanity.png')
    ap.add_argument('--long-side', type=int, default=0,
                    help='downscale so the long side is this many px before the '
                         'teacher (0 = full res). Use ~640 on CPU -- the ViT is '
                         'much faster on a smaller input and the sanity read is fine.')
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    import cv2
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    if args.raw:
        bgr = rectify_raw(bgr, args.calib)
        print(f"rectified raw frame -> {bgr.shape[1]}x{bgr.shape[0]}")
    if args.long_side and max(bgr.shape[:2]) > args.long_side:
        s = args.long_side / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)))
        print(f"downscaled for the teacher -> {bgr.shape[1]}x{bgr.shape[0]}")
    pil = Image.fromarray(bgr[:, :, ::-1])          # BGR -> RGB

    print(f"loading teacher {args.model} on {args.device} "
          f"(first run downloads the model)...")
    proc, model = load_teacher(args.model, args.device)
    disp = infer_disparity(proc, model, pil, (pil.height, pil.width), args.device)

    finite = np.isfinite(disp)
    print(f"disparity: min={disp.min():.3f} max={disp.max():.3f} "
          f"mean={disp.mean():.3f} std={disp.std():.3f} "
          f"finite={finite.all()} ({finite.mean()*100:.1f}%)")
    sbs = np.hstack([bgr, colorize(disp)])
    cv2.imwrite(args.out, sbs)
    print(f"wrote {args.out}  ([ rectified RGB | teacher inverse-depth ])")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Fisheye (Kannala-Brandt) intrinsic calibration for the Arducam B0179 (IMX219,
2.5 mm M12, ~170D x 155H x 115V). At that field of view the lens is necessarily a
fisheye -- a rectilinear 2.5 mm lens on this 1/4" sensor tops out near 73 deg
horizontal, so the 155 deg it delivers can only be a fisheye projection. Calibrate
with cv2.fisheye (the same model calibration.yaml's `model: fisheye` assumes) and
paste the printed block into src/ringfusion_bringup/config/calibration.yaml.

Two steps -- capture, then calibrate:

  # 1. Collect ~20 checkerboard views (headless auto-capture; move the board
  #    slowly around the frame, including the corners where fisheye distortion
  #    is strongest). 9x6 = inner corners of a standard 10x7-square board.
  python tools/calibrate_camera.py --capture calib_imgs --cols 9 --rows 6

  # 2. Calibrate from the saved images and print the yaml block.
  python tools/calibrate_camera.py --images calib_imgs --cols 9 --rows 6 \
         --square-mm 25 --width 1640 --height 1232

Notes
- Capture through the SAME geometry perception uses (default flip = 180, matching
  camera.py FLIP_180), so the intrinsics match the published frames. Color
  correction is deliberately NOT applied -- geometry only needs corners.
- Intrinsics are resolution-specific. Calibrate at the resolution you will deploy.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    print("This tool needs OpenCV (cv2):", e, file=sys.stderr)
    sys.exit(1)

FLIP_180 = 2  # matches ringfusion_drivers/camera.py


def gst_pipeline(sensor_id, w, h, fps, flip):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1, "
        f"format=NV12 ! nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


def capture(outdir, cols, rows, num, sensor_id, w, h, fps, flip):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(gst_pipeline(sensor_id, w, h, fps, flip), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Could not open camera. Is the Arducam driver installed and "
              "PYTHONNOUSERSITE=1 set (so cv2 has GStreamer)?", file=sys.stderr)
        return 1
    print(f"Auto-capturing {num} checkerboard views to {outdir}/ ."
          f" Move the board slowly; cover the edges/corners. Ctrl-C to stop.")
    saved, tries = 0, 0
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    try:
        while saved < num:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, _ = cv2.findChessboardCorners(gray, (cols, rows), flags)
            tries += 1
            if found:
                path = os.path.join(outdir, f"view_{saved:03d}.png")
                cv2.imwrite(path, frame)
                saved += 1
                print(f"  [{saved}/{num}] saved {path}")
                time.sleep(0.8)   # give time to reposition the board
            elif tries % 15 == 0:
                print(f"  ...no board detected yet ({saved}/{num} so far)")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        cap.release()
    print(f"Collected {saved} views. Now run with --images {outdir} to calibrate.")
    return 0


def calibrate(images_dir, cols, rows, square_mm, w, h):
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                   glob.glob(os.path.join(images_dir, "*.jpg")))
    if not paths:
        print(f"No images in {images_dir}", file=sys.stderr)
        return 1

    # one 3D object-point grid, scaled to real square size (units: mm -> the
    # translation vectors come out in mm; intrinsics are unaffected by the scale)
    objp = np.zeros((1, cols * rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    objpoints, imgpoints = [], []
    size = None
    subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    used = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print(f"  skip {p}: size {gray.shape[::-1]} != {size}")
            continue
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if not found:
            print(f"  skip {p}: no board")
            continue
        corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix)
        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1
    if used < 5:
        print(f"Only {used} usable views; need >=5 (ideally 15-25).", file=sys.stderr)
        return 1
    if size != (w, h):
        print(f"WARNING: image size {size} != requested {(w, h)}; using {size}.")
        w, h = size

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    calib_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                   cv2.fisheye.CALIB_FIX_SKEW)
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, size, K, D,
        flags=calib_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    k = [float(D[i, 0]) for i in range(4)]
    print(f"\nCalibrated on {used} views. RMS reprojection error: {rms:.4f} px")
    print("(good < ~0.5 px; > 1 px means bad detections or too few/uniform views)\n")

    block = f"""# ---- paste into src/ringfusion_bringup/config/calibration.yaml ----
camera:
  width: {w}
  height: {h}
  model: fisheye
  fx: {fx:.4f}
  fy: {fy:.4f}
  cx: {cx:.4f}
  cy: {cy:.4f}
  dist: [{k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f}, {k[3]:.6f}]   # k1..k4 Kannala-Brandt

rectify:
  # rectilinear (pinhole) output that the perception pipeline consumes.
  # Same WxH is fine; tune fov_scale/balance after eyeballing the rectified image.
  width: {w}
  height: {h}
  balance: 0.0        # 0 = crop to all-valid pixels, 1 = keep all source pixels
  fov_scale: 1.0      # >1 zooms in (narrower FOV, less extreme edge stretch)
"""
    print(block)
    out = os.path.join(images_dir, "calibration_block.yaml")
    with open(out, "w") as f:
        f.write(block)
    print(f"(also written to {out})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", metavar="DIR", help="capture checkerboard views to DIR")
    ap.add_argument("--images", metavar="DIR", help="calibrate from images in DIR")
    ap.add_argument("--cols", type=int, default=9, help="inner corners across (default 9)")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down (default 6)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="square size in mm")
    ap.add_argument("--num", type=int, default=20, help="views to capture (default 20)")
    ap.add_argument("--width", type=int, default=1640)   # deploy res: full-FOV binned mode
    ap.add_argument("--height", type=int, default=1232)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=FLIP_180,
                    help="nvvidconv flip-method; keep matching camera.py (default 2)")
    ap.add_argument("--sensor-id", type=int, default=0)
    args = ap.parse_args()

    if args.capture:
        return capture(args.capture, args.cols, args.rows, args.num,
                       args.sensor_id, args.width, args.height, args.fps, args.flip)
    if args.images:
        return calibrate(args.images, args.cols, args.rows, args.square_mm,
                         args.width, args.height)
    ap.error("give either --capture DIR or --images DIR")


if __name__ == "__main__":
    sys.exit(main())

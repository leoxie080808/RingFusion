"""Camera source for the AGX. Two backends:

  ArducamCSI  - the B0179 via the B0472 quad adapter, opened through GStreamer
                nvarguscamerasrc (the native Jetson path). Requires the Arducam
                kernel driver + JetPack camera stack set up first.
  ImageFile   - a saved image/video, so the pipeline runs with no camera.

NOTE on the B0472: it combines up to 4 cameras into ONE stitched frame. Set
sensor_id and the per-camera crop once you know your capture layout; for a single
camera on the adapter, capture the full frame and use it directly.
"""
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


# nvvidconv flip-method: 0=none, 2=rotate 180. The module is mounted upside
# down on the ring, so 2 is the correct default for this rig.
FLIP_180 = 2


def gst_pipeline(sensor_id=0, capture_w=1280, capture_h=720, fps=15, flip=FLIP_180):
    """GStreamer string for nvarguscamerasrc on Jetson."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_w}, height={capture_h}, "
        f"framerate={fps}/1, format=NV12 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


def _auto_tone_lut(sample_bgr):
    """Per-channel gray-world gain + 2-98 percentile contrast stretch, as a LUT.

    The IMX219 modules have no ISP color-tuning profile installed, so raw
    frames come out with a strong color cast and the mid-tones compressed
    into a ~15/255-wide band (worse when the wide-angle lens catches a bright
    light directly) - flat and washed out. Computing stats on a strided
    sample and applying via cv2.LUT keeps this under ~10ms/frame; doing the
    same math with np.percentile over the full frame cost >200ms.
    """
    sample = sample_bgr[::4, ::4, :].reshape(-1, 3).astype(np.float32)
    means = sample.mean(axis=0)
    gains = np.clip(means.mean() / np.clip(means, 1, None), 0.5, 3.0)
    lo = np.percentile(sample, 2, axis=0) * gains
    hi = np.percentile(sample, 98, axis=0) * gains
    scale = 255.0 / np.clip(hi - lo, 1, None)
    x = np.arange(256, dtype=np.float32)
    luts = np.zeros((3, 256), dtype=np.uint8)
    for c in range(3):
        luts[c] = np.clip((x * gains[c] - lo[c]) * scale[c], 0, 255).astype(np.uint8)
    return luts


def auto_tone_correct(frame_bgr):
    """Apply gray-world white balance + contrast stretch to a BGR uint8 frame."""
    luts = _auto_tone_lut(frame_bgr)
    b, g, r = cv2.split(frame_bgr)
    b, g, r = cv2.LUT(b, luts[0]), cv2.LUT(g, luts[1]), cv2.LUT(r, luts[2])
    return cv2.merge([b, g, r])


class ArducamCSI:
    def __init__(self, sensor_id=0, width=1280, height=720, fps=15, flip=FLIP_180):
        if cv2 is None:
            raise RuntimeError("opencv-python required for camera capture")
        self.cap = cv2.VideoCapture(
            gst_pipeline(sensor_id, width, height, fps, flip), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open CSI camera. Check the Arducam driver install, "
                "JetPack camera stack, and that nvarguscamerasrc works "
                "(test: gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! autovideosink).")

    def read(self):
        ok, frame = self.cap.read()          # BGR
        if not ok:
            return None
        frame = auto_tone_correct(frame)
        return frame[:, :, ::-1].copy()      # -> RGB

    def close(self):
        self.cap.release()


class ImageFile:
    """Serve a still image (or the first frame of a video) repeatedly."""
    def __init__(self, path):
        if cv2 is None:
            raise RuntimeError("opencv-python required")
        img = cv2.imread(path)
        if img is None:
            cap = cv2.VideoCapture(path); ok, img = cap.read(); cap.release()
            if not ok:
                raise RuntimeError(f"could not read {path}")
        self.rgb = img[:, :, ::-1].copy()

    def read(self):
        return self.rgb.copy()

    def close(self):
        pass


def synthetic_image(w=1280, h=720):
    """A gradient + grid image so the pipeline runs with no file at all."""
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    gx, gy = np.meshgrid(x, y)
    img = np.stack([gy, gx, 0.5 * (gx + gy)], axis=-1)
    img[::40, :, :] = 1.0
    img[:, ::40, :] = 1.0
    return (img * 255).astype(np.uint8)

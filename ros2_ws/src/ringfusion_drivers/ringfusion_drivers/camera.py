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


def gst_pipeline(sensor_id=0, capture_w=1280, capture_h=720, fps=15, flip=0):
    """GStreamer string for nvarguscamerasrc on Jetson."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_w}, height={capture_h}, "
        f"framerate={fps}/1, format=NV12 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


class ArducamCSI:
    def __init__(self, sensor_id=0, width=1280, height=720, fps=15):
        if cv2 is None:
            raise RuntimeError("opencv-python required for camera capture")
        self.cap = cv2.VideoCapture(
            gst_pipeline(sensor_id, width, height, fps), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open CSI camera. Check the Arducam driver install, "
                "JetPack camera stack, and that nvarguscamerasrc works "
                "(test: gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! autovideosink).")

    def read(self):
        ok, frame = self.cap.read()          # BGR
        if not ok:
            return None
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

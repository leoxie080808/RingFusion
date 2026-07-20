"""Stage 1 -- fisheye rectification.

The Arducam B0179 is a ~155 deg fisheye. The monocular backbone expects
rectilinear imagery and the cloud unprojection is pinhole, so the raw frame is
remapped to a rectilinear (pinhole) crop BEFORE it enters the pipeline. The
remap is precomputed once (Kannala-Brandt undistort), so at run time this stage
is a cv2.remap -- memory-bound, a few ms.

`FisheyeRectifier` reads the fisheye calibration + the `rectify` output spec and
exposes:
  .K_rect   (fx, fy, cx, cy) pinhole intrinsics of the rectified image
  .size     (w, h) of the rectified image
  .rectify(img) -> rectified image

A zero-distortion fisheye is still an *equidistant* fisheye (image radius grows
with the angle, not its tangent), so undistorting it to rectilinear is a real and
dominant correction even before the lens-specific k-coefficients are measured --
and the nominal focal length is already close to the true equidistant value. So a
`model: fisheye` calibration always builds a real remap; only a `pinhole` model
(or missing cv2) degrades to the identity passthrough. Real calibration refines
the result; it does not switch it on.
"""
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def _K_matrix(K):
    fx, fy, cx, cy = K
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


class FisheyeRectifier:
    def __init__(self, K, dist, model, size_in, size_out=None,
                 balance=0.0, fov_scale=1.0):
        """K: (fx,fy,cx,cy) of the raw image. dist: k1..k4. model: 'fisheye'|'pinhole'.
        size_in/size_out: (w, h). balance/fov_scale: cv2.fisheye undistort controls."""
        self.size = tuple(size_out or size_in)
        D = np.asarray(dist, np.float64).reshape(-1)
        if D.size < 4:
            D = np.concatenate([D, np.zeros(4 - D.size)])
        D = D[:4]

        if cv2 is None or model != 'fisheye':
            # already rectilinear (or no cv2) -> nothing to remap
            self._identity = True
            self.K_rect = tuple(float(v) for v in K)
            self._map1 = self._map2 = None
            return

        self._identity = False
        Km = _K_matrix(K)
        Dm = D.reshape(4, 1)
        P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            Km, Dm, tuple(size_in), np.eye(3),
            balance=float(balance), new_size=self.size, fov_scale=float(fov_scale))
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            Km, Dm, np.eye(3), P, self.size, cv2.CV_16SC2)
        self.K_rect = (float(P[0, 0]), float(P[1, 1]), float(P[0, 2]), float(P[1, 2]))

    def rectify(self, img):
        if self._identity:
            return img
        return cv2.remap(img, self._map1, self._map2, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    @property
    def is_identity(self):
        return self._identity

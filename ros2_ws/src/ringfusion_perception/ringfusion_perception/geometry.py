"""Camera/ToF geometry for RingFusion.

The load-bearing routine is `project_zone_to_pixel`: it takes a ToF zone, turns
it into a 3D point, moves it into the camera frame with the measured extrinsic,
and projects it to a pixel. This is where the 20.195 mm height offset does its
job -- without it, every zone lands at the wrong pixel and the anchoring fit is
silently biased.

Nominal-intrinsic derivation (Arducam B0179, IMX219):
  sensor: 1/4", 3280x2464 active, 1.12 um square pixels
  lens:   f = 2.5 mm
  native fx = f / pixel = 2500 um / 1.12 um = 2232 px at full 3280 width.
  At a 1280 capture the sensor is scaled/binned by ~1280/3280 = 0.39,
  so fx ~= 2232 * 0.39 ~= 460 px. (Real value comes from calibration.)
Conventions (REP-103 optical): x right, y down, z forward.
"""
import numpy as np


def make_T_cam_tof(translation_mm, rotation_rpy_deg):
    """4x4 homogeneous transform: ToF-frame point -> camera-frame point."""
    tx, ty, tz = np.asarray(translation_mm, float) / 1000.0   # mm -> m
    r, p, y = np.deg2rad(rotation_rpy_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def zone_directions(cols, rows, fov_h_deg, fov_v_deg):
    """Unit ray for each ToF zone center, in the ToF optical frame.
    Returns (rows, cols, 3). Zone (0,0) is top-left; +x right, +y down, +z fwd."""
    fh = np.deg2rad(fov_h_deg)
    fv = np.deg2rad(fov_v_deg)
    # angular offset of each zone center from the optical axis
    cx = (np.arange(cols) - (cols - 1) / 2.0) / max(cols - 1, 1)   # -0.5..0.5
    cy = (np.arange(rows) - (rows - 1) / 2.0) / max(rows - 1, 1)
    ax = cx * fh                      # horizontal angle per column
    ay = cy * fv                      # vertical angle per row
    AX, AY = np.meshgrid(ax, ay)      # (rows, cols)
    dx = np.tan(AX)
    dy = np.tan(AY)
    dz = np.ones_like(dx)
    d = np.stack([dx, dy, dz], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d


def project_points_pinhole(pts_cam, K):
    """(N,3) camera-frame points -> (N,2) pixels + validity (z>0)."""
    fx, fy, cx, cy = K
    z = pts_cam[:, 2]
    valid = z > 1e-6
    u = np.full(len(pts_cam), -1.0)
    v = np.full(len(pts_cam), -1.0)
    u[valid] = fx * pts_cam[valid, 0] / z[valid] + cx
    v[valid] = fy * pts_cam[valid, 1] / z[valid] + cy
    return np.stack([u, v], axis=-1), valid


def project_points_fisheye(pts_cam, K, dist):
    """Kannala-Brandt equidistant fisheye projection. dist = [k1,k2,k3,k4]."""
    fx, fy, cx, cy = K
    k1, k2, k3, k4 = dist
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid = z > 1e-6
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(r, z)                        # angle from optical axis
    t2 = theta * theta
    thetad = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(r > 1e-9, thetad / np.where(r > 1e-9, r, 1.0), 1.0)
    u = fx * (x * scale) + cx
    v = fy * (y * scale) + cy
    return np.stack([u, v], axis=-1), valid


def project_zone_to_pixel(dist_m, valid_mask, cols, rows, fov_h, fov_v,
                          T_cam_tof, K, dist_coeffs, model='fisheye'):
    """Project every valid ToF zone onto the camera image.

    Returns dict with pixel coords (uv), in-image mask, camera-frame z-depth,
    and the flat zone distances -- everything the anchoring step needs.
    """
    dirs = zone_directions(cols, rows, fov_h, fov_v).reshape(-1, 3)   # (N,3)
    d = dist_m.reshape(-1)
    v = valid_mask.reshape(-1).astype(bool)
    # 3D point in ToF frame = ray * measured range
    pts_tof = dirs * d[:, None]
    # -> camera frame
    ph = np.concatenate([pts_tof, np.ones((len(pts_tof), 1))], axis=1)
    pts_cam = (ph @ T_cam_tof.T)[:, :3]
    # project
    if model == 'fisheye':
        uv, zvalid = project_points_fisheye(pts_cam, K, dist_coeffs)
    else:
        uv, zvalid = project_points_pinhole(pts_cam, K)
    z_cam = pts_cam[:, 2]
    return {
        'uv': uv,                         # (N,2) pixel per zone
        'z_cam': z_cam,                   # (N,) depth along camera z (for anchoring)
        'range_m': d,                     # (N,) raw ToF range
        'valid': v & zvalid,              # (N,) usable this frame
        'pts_cam': pts_cam,               # (N,3) 3D in camera frame
    }


def unproject_depth_to_cloud(depth_m, K, model='pinhole', dist_coeffs=None, stride=2):
    """Dense metric depth image -> (M,3) camera-frame point cloud.
    Pinhole unprojection (used after anchoring gives metric depth)."""
    fx, fy, cx, cy = K
    h, w = depth_m.shape
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[vv, uu]
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts[pts[:, 2] > 0]

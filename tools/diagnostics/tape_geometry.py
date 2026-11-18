#!/usr/bin/env python3
"""Robot-frame tape geometry: three tape legs -> camera-frame depth. ONE implementation.

Deliberately free of ROS, OpenCV and the perception package, so every consumer can import it:
the two capture front ends (tape_capture, tape_web) need it live on the robot, and the three
offline scorers (tape_eval, tape_repeats, plane_eval) need it on a desk with no ROS installed.
tape_capture imports rclpy at module scope, so it cannot be that shared home.

Why this matters more than the usual don't-repeat-yourself: a divergence between any two of
these would mean two different ground-truth references inside one session, and the whole
purpose of the tape workstream is to be the one reference nothing else in the repo can check.

FRAME (see ros2_ws/src/ringfusion_bringup/config/robot_geometry.yaml for the full statement):
ROS REP 103 -- robot centre marked ON THE FLOOR, x forward, y LEFT, z up. Camera pitch
positive = nose down, yaw positive = turned LEFT. The checks that pin both signs are in
test_tape_geometry().

y IS LEFT-POSITIVE, matching the T3 run sheet and REP 103. The Sept 27 brief said
right-positive; the run sheet is what the operator reads while taping, so it wins.

The conversion is INVARIANT under mirroring -- flipping y and yaw together changes Z by
0.000 mm. What is not safe is MIXING them: y logged left-positive against a yaw supplied
right-positive shifts Z by 38 mm at 1.8 m off-axis and 87 mm at 2.5 m, which is the scale that
would quietly corrupt the out-of-cone claim. So y and yaw must share a handedness, and here
that handedness is LEFT.
"""
import os

import numpy as np


def load_geometry(path):
    """Items 27-29 as metres and degrees, or None if the file is missing or still UNSET.

    Returning None rather than raising is the point: a legs session captured before the CAD
    numbers land is still fully recorded and scoreable later, and a missing config must never
    cost a session that has markers on the wall.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        import yaml
    except ImportError:
        return None
    d = yaml.safe_load(open(path)) or {}
    c = d.get('camera_optical_centre_mm') or {}
    o = d.get('camera_orientation_deg') or {}
    vals = [c.get('x'), c.get('y'), c.get('z'), o.get('pitch'), o.get('yaw')]
    if any(v is None or isinstance(v, str) for v in vals):
        return None                      # 'UNSET' is a str; a real reading is a number
    return {'x_c': float(c['x']) / 1000.0, 'y_c': float(c['y']) / 1000.0,
            'z_c': float(c['z']) / 1000.0,
            'pitch_deg': float(o['pitch']), 'yaw_deg': float(o['yaw']),
            'source': d.get('source', 'unspecified')}


def legs_offset(x_m, y_m, z_m, g):
    """The three differences the conversions are built from: (d forward, y right, h drop)."""
    return (np.asarray(x_m, float) - g['x_c'],      # d, forward
            np.asarray(y_m, float) - g['y_c'],      # y, LEFT-positive
            g['z_c'] - np.asarray(z_m, float))      # h, drop from camera down to the marker


def axial_from_legs(x_m, y_m, z_m, g):
    """Camera-frame AXIAL depth Z -- the quantity the pipeline predicts and T3 compares to.

        Z = d cos(pitch) cos(yaw) + y cos(pitch) sin(yaw) + h sin(pitch)

    which is the marker offset dotted with the camera's forward axis. At yaw 0 it reduces to
    the brief's Z = d cos(theta) + h sin(theta).
    """
    d, y, h = legs_offset(x_m, y_m, z_m, g)
    t, p = np.deg2rad(g['pitch_deg']), np.deg2rad(g['yaw_deg'])
    return d * np.cos(t) * np.cos(p) + y * np.cos(t) * np.sin(p) + h * np.sin(t)


def slant_from_legs(x_m, y_m, z_m, g):
    """Straight-line range from the optical centre. A CROSS-CHECK, never the reference."""
    d, y, h = legs_offset(x_m, y_m, z_m, g)
    return np.sqrt(d * d + y * y + h * h)


def project_legs(x_m, y_m, z_m, g, K, pitch_deg=None, yaw_deg=None):
    """Where a marker at these tape legs SHOULD land on the rectified image, in pixels.

    The forward model behind solve_pitch_yaw. Camera optical frame is x right, y DOWN,
    z forward; the robot frame is x forward, y LEFT, z up, so the axes permute and one sign
    flips. With P = d cos(yaw) + y sin(yaw) and Q = d sin(yaw) - y cos(yaw):

        X_cam = Q                       (right)
        Z_cam = P cos(pitch) + h sin(pitch)   -- the axial depth, same as axial_from_legs
        Y_cam = -P sin(pitch) + h cos(pitch)  (down)

    (Z_cam, Y_cam) is just (P, h) rotated by pitch, which is what makes the inverse solvable
    in closed form for pitch once yaw is known.
    """
    fx, fy, cx, cy = K
    t = np.deg2rad(g['pitch_deg'] if pitch_deg is None else pitch_deg)
    p_ = np.deg2rad(g['yaw_deg'] if yaw_deg is None else yaw_deg)
    d, y, h = legs_offset(x_m, y_m, z_m, g)
    P = d * np.cos(p_) + y * np.sin(p_)
    Q = d * np.sin(p_) - y * np.cos(p_)
    Z = P * np.cos(t) + h * np.sin(t)
    Y = -P * np.sin(t) + h * np.cos(t)
    return fx * Q / Z + cx, fy * Y / Z + cy


def solve_pitch_yaw(marks, g, K):
    """Recover camera pitch and yaw from axis markers of KNOWN tape legs and observed pixels.

    marks: [(x_m, y_m, z_m, u_obs, v_obs), ...] -- at least one, two is the point.

    WHY THIS RATHER THAN THE RUN SHEET'S CLOSED FORM. That form,
    pitch = -atan((v-cy)/fy) and yaw = atan((u-cx)/fx cos pitch), assumes the marker sits
    EXACTLY on the axis at exactly lens height, so every millimetre of placement error is read
    as a real angle: 5 mm sideways at 2 m is a false 0.147 deg, which is the entire error
    budget. Placing a marker CENTRE on a chalk line to 5 mm, at height, is fiddly and
    unverifiable afterwards.

    Here the legs are measured like any other marker, so placement accuracy stops mattering
    and only MEASUREMENT accuracy does -- and measuring to 1 mm with a steel tape is exactly
    what the rest of the session already does. Put the markers roughly on the axis (it keeps
    them near the principal point where the lens model is best behaved) and measure where they
    actually ended up.

    Returns (pitch_deg, yaw_deg, per_mark_residual_px, spread_deg) where spread_deg is the
    largest disagreement between individual marker solutions -- the run sheet's "A1 and A2
    should agree within about 0.2 deg" check, which needs two or more marks to mean anything.
    """
    from scipy.optimize import least_squares

    def resid(th, subset):
        out = []
        for x, y, z, u, v in subset:
            pu, pv = project_legs(x, y, z, g, K, pitch_deg=th[0], yaw_deg=th[1])
            out += [pu - u, pv - v]
        return out

    fit = least_squares(resid, [0.0, 0.0], args=(marks,), xtol=1e-12, ftol=1e-12)
    pitch, yaw = float(fit.x[0]), float(fit.x[1])
    per_mark = [float(np.hypot(*resid(fit.x, [m]))) for m in marks]

    spread = 0.0
    if len(marks) > 1:
        singles = []
        for m in marks:
            f1 = least_squares(resid, [0.0, 0.0], args=([m],), xtol=1e-12, ftol=1e-12)
            singles.append((float(f1.x[0]), float(f1.x[1])))
        spread = max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                     for a in singles for b in singles)
    return pitch, yaw, per_mark, spread


def has_legs(p):
    """True if this point carries tape legs rather than a single slant range."""
    return p.get('x_m') is not None


def test_tape_geometry():
    """The checks that pin both sign conventions. Run: python3 tape_geometry.py"""
    g = {'x_c': 0.120, 'y_c': 0.0, 'z_c': 0.250, 'pitch_deg': 0.0, 'yaw_deg': 0.0}

    # the brief's worked example
    Z = float(axial_from_legs(2.120, 0.0, 0.100, g))
    r = float(slant_from_legs(2.120, 0.0, 0.100, g))
    assert abs(Z - 2.0) < 5e-5, Z
    assert abs(r - 2.0056) < 5e-5, r
    # and 0.8 m sideways leaves Z alone while r grows
    assert abs(float(axial_from_legs(2.120, 0.8, 0.100, g)) - 2.0) < 5e-5
    assert abs(float(slant_from_legs(2.120, 0.8, 0.100, g)) - 2.1593) < 5e-5

    # YAW SIGN: +1 deg of LEFT yaw at 1 m to the LEFT must ADD about 17 mm of Z
    gy = dict(g, yaw_deg=1.0)
    dz = float(axial_from_legs(2.120, 1.0, 0.250, gy)) - \
        float(axial_from_legs(2.120, 1.0, 0.250, g))
    assert 0.0165 < dz < 0.0180, f'yaw sign/size wrong: {dz*1000:.1f} mm'

    # PITCH SIGN: Z is maximised at pitch = atan(h/d), where it equals r exactly. Only true
    # for nose-down-positive; the opposite sign puts the maximum at negative pitch.
    d, y, h = legs_offset(2.120, 0.0, 0.100, g)
    best = float(np.degrees(np.arctan2(h, d)))
    assert best > 0, 'marker below the camera must need POSITIVE pitch'
    Zb = float(axial_from_legs(2.120, 0.0, 0.100, dict(g, pitch_deg=best)))
    assert abs(Zb - r) < 1e-9, f'{Zb} != {r}'
    for off in (-1.0, +1.0):
        assert float(axial_from_legs(2.120, 0.0, 0.100,
                                     dict(g, pitch_deg=best + off))) < Zb

    # SOLVER ROUND TRIP. Deliberately uses BADLY placed axis markers -- 200 mm off the axis
    # and 100 mm off in height -- because the whole point of solving rather than reading the
    # closed form is that placement accuracy stops mattering. If this ever starts depending on
    # where the markers sat, the method has silently reverted to the naive one.
    K = (379.6, 379.5, 839.2, 548.9)
    gk = dict(g, pitch_deg=0.0, yaw_deg=0.0)
    for true_p, true_y in ((1.37, -0.84), (-0.6, 1.9), (0.0, 0.0)):
        marks = []
        for x, y, z in ((2.0, 0.200, 0.050), (4.0, -0.150, 0.300)):
            u, v = project_legs(x, y, z, gk, K, pitch_deg=true_p, yaw_deg=true_y)
            marks.append((x, y, z, float(u), float(v)))
        got_p, got_y, res, spread = solve_pitch_yaw(marks, gk, K)
        assert abs(got_p - true_p) < 1e-6, f'pitch {got_p} != {true_p}'
        assert abs(got_y - true_y) < 1e-6, f'yaw {got_y} != {true_y}'
        assert max(res) < 1e-6 and spread < 1e-6

    print('tape_geometry: brief example, yaw sign, pitch sign and the pitch/yaw solver '
          'all check out')


if __name__ == '__main__':
    test_tape_geometry()

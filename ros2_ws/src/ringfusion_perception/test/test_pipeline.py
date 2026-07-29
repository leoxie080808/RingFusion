"""Pure-numpy tests for the perception pipeline -- runnable on a dev PC with no
ROS, CUDA, or cv2. Exercises Stages 2-8 with the mock backbone/residual.

Run:  cd ros2_ws/src/ringfusion_perception && python -m pytest test/ -v
  or: python test/test_pipeline.py        (plain, prints PASS/FAIL per check)
"""
import os
import sys

import numpy as np

# Make `ringfusion_perception` importable no matter how this is launched:
# add the package's parent dir (the one containing the inner package folder).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ringfusion_perception import pipeline, geometry as geo, anchoring as anc
from ringfusion_perception.backbone import MockBackbone
from ringfusion_perception.residual import MockResidual


# --- fixtures ---------------------------------------------------------------

def make_calib():
    """Pinhole calib so ToF zones project deterministically into the image."""
    return {
        'K': (460.0, 460.0, 640.0, 360.0),
        'dist': np.zeros(4),
        'model': 'pinhole',
        'T_cam_tof': geo.make_T_cam_tof([0.0, 20.195, 1.13], [0.0, 0.0, 0.0]),
        'img_w': 1280, 'img_h': 720,
        'fov_h': 61.0, 'fov_v': 45.0,
    }


def make_tof(rows=32, cols=48, weak_top=False):
    """A wall nearer toward the image bottom (so measured inverse-depth rises with
    image row, matching StubBackbone's downward-increasing disparity -> a > 0)."""
    r = np.arange(rows)[:, None] / (rows - 1)
    c = np.arange(cols)[None, :] / (cols - 1)
    dist = (2.0 - 0.8 * r + 0.1 * c).astype(np.float32)
    dist = np.broadcast_to(dist, (rows, cols)).astype(np.float32).copy()
    valid = np.ones((rows, cols), bool)
    conf = np.full((rows, cols), 40, np.uint8)
    if weak_top:
        conf[: rows // 2, :] = 3          # top half low-confidence
    return dist, valid, conf


def make_rgb(h=720, w=1280):
    g = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None]
    return np.broadcast_to(g, (h, w, 3)).copy()


class StubBackbone:
    """Deterministic vertical-gradient disparity (no RNG) for repeatable checks:
    larger toward the bottom of the image."""
    name = "stub"

    def infer(self, rgb):
        v = np.linspace(0.3, 1.2, rgb.shape[0], dtype=np.float32)[:, None]
        return np.repeat(v, rgb.shape[1], axis=1)


# --- tests ------------------------------------------------------------------

def test_pipeline_runs_end_to_end():
    np.random.seed(0)
    calib = make_calib()
    dist, valid, _ = make_tof()
    rgb = make_rgb()
    res = pipeline.run(rgb, dist, valid, calib, MockBackbone(), MockResidual())

    assert res['ok'], "anchoring should succeed with a full valid ToF frame"
    assert res['n_anchors'] > 100, f"expected many anchors, got {res['n_anchors']}"
    assert res['metric'].shape == (720, 1280)
    assert np.isfinite(res['metric']).all()
    assert (res['metric'] > 0).all()
    # plausible indoor range (the mock scene is ~1-2 m)
    assert 0.05 < res['metric'].mean() < 50.0
    assert res['cloud'].ndim == 2 and res['cloud'].shape[1] == 3
    assert len(res['cloud']) > 0
    assert res['var'] is not None and res['var'].shape == (720, 1280)
    assert np.isfinite(res['var']).all()
    assert res['var'].min() >= -1e-6, "variance must be non-negative"


def test_mock_residual_is_identity():
    """Stage 7 with the mock residual must reproduce the closed-form result
    exactly (residual is zero-initialized -> identity)."""
    calib = make_calib()
    dist, valid, _ = make_tof()
    rgb = make_rgb()
    bb = StubBackbone()

    closed = pipeline.run(rgb, dist, valid, calib, bb, residual=None)
    with_mock = pipeline.run(rgb, dist, valid, calib, bb, MockResidual())

    assert np.array_equal(closed['metric'], with_mock['metric'])
    assert np.array_equal(closed['var'], with_mock['var'])   # extra var is zero


def test_blend_pulls_depth_toward_tof_at_anchors():
    """Stage 7c must hand anchor pixels (nearly) the real ToF reading, and leave
    pixels far from any anchor to the network. Measured 2026-07-28: nearest-zone ToF
    beats the network ~2x under 3 deg and loses ~6x past 15 deg, so the blend has to
    actually switch sources, not just perturb them."""
    calib = make_calib()
    dist, valid, _ = make_tof()
    rgb = make_rgb()
    bb = StubBackbone()

    off = pipeline.run(rgb, dist, valid, calib, bb, residual=None, blend=False)
    on = pipeline.run(rgb, dist, valid, calib, bb, residual=None, blend=True)

    # Anchor pixels: recompute the splat the pipeline used, then compare both runs
    # against the true ToF depth there. Blending must strictly reduce that error.
    ad, am = _anchor_splat(rgb, dist, valid, calib)
    ys, xs = np.nonzero(am > 0)
    e_off = np.abs(off['metric'][ys, xs] - ad[ys, xs]).mean()
    e_on = np.abs(on['metric'][ys, xs] - ad[ys, xs]).mean()
    assert e_on < e_off, f"blend should cut anchor error, got {e_on:.4f} vs {e_off:.4f}"
    assert e_on < 0.02, f"at anchors the blend should be ~the ToF value, got {e_on:.4f}"

    # Far from every anchor the network must survive untouched.
    from ringfusion_perception.blend import blend_depth
    _, wgt = blend_depth(off['metric'], ad, am, fx=calib['K'][0])
    far = wgt == 0.0
    if far.any():
        assert np.allclose(on['metric'][far], off['metric'][far]), \
            "pixels with zero blend weight must be unchanged"


def test_far_field_clamp_bounds_output():
    """Invariant: whatever the fit produces, run() must not return depth above
    MAX_DEPTH_M. anc.to_metric_depth bounds inverse depth at min_disp=1e-4, so a
    near-singular fit emits up to 10 000 m; ResidualRefiner clamps its own output, which
    left the Network-B-OFF fallback as the only unclamped path into /cloud.

    Forced by patching the metric stage rather than by a crafted scene: with a monotonic
    synthetic disparity ramp solve_robust simply rescales `a` to match the anchors and
    the extrapolation stays bounded, so no synthetic input reproduces the runaway. The
    real case (closed-form MAE 18.092 m, median 0.066) came from real backbone noise --
    see docs/demo/benchmarks/baselines.json, 'center' protocol.
    """
    from ringfusion_perception.residual import MAX_DEPTH_M
    calib = make_calib()
    dist, valid, _ = make_tof()
    rgb = make_rgb()

    orig = anc.to_metric_depth

    def runaway(disp, a, b, **kw):
        D = np.array(orig(disp, a, b), copy=True)
        D[:10, :] = 9999.0                      # a near-singular far field
        return D

    pipeline.anc.to_metric_depth = runaway
    try:
        res = pipeline.run(rgb, dist, valid, calib, StubBackbone(), residual=None,
                           blend=False, use_gpu=False)
    finally:
        pipeline.anc.to_metric_depth = orig

    assert res['ok'], "fit should still succeed"
    assert res['metric'].max() <= MAX_DEPTH_M + 1e-3, \
        f"unclamped far field: {res['metric'].max():.1f} m"
    assert np.isfinite(res['metric']).all()


def _anchor_splat(rgb, dist, valid, calib):
    """Reproduce the pipeline's anchor splat for assertions."""
    h, w = rgb.shape[:2]
    rows, cols = dist.shape
    p = geo.project_zone_to_pixel(dist, valid, cols, rows, calib['fov_h'],
                                  calib['fov_v'], calib['T_cam_tof'], calib['K'],
                                  calib['dist'], model=calib['model'])
    uv, z, ok = p['uv'], p['z_cam'], p['valid']
    fin = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
    u = np.round(np.where(fin, uv[:, 0], -1)).astype(int)
    v = np.round(np.where(fin, uv[:, 1], -1)).astype(int)
    inb = ok & fin & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    return pipeline.splat_anchors(u, v, z, inb, (h, w))


def test_variance_grows_with_depth():
    """The delta-method variance carries a D^4 factor, so far pixels must be more
    uncertain than near ones. StubBackbone + this scene put far pixels at the top."""
    calib = make_calib()
    dist, valid, _ = make_tof()
    rgb = make_rgb()
    res = pipeline.run(rgb, dist, valid, calib, StubBackbone(), MockResidual())

    var = res['var']
    top = var[:180].mean()          # far (large depth)
    bottom = var[-180:].mean()      # near (small depth)
    assert res['a'] > 0, "scene was built so disparity and inv-depth correlate positively"
    assert top > bottom, f"far pixels should be more uncertain (top {top:.3g} vs bottom {bottom:.3g})"


def test_confidence_rejection():
    """min_confidence should drop low-confidence zones from the fit."""
    calib = make_calib()
    dist, valid, conf = make_tof(weak_top=True)
    rgb = make_rgb()
    bb = StubBackbone()

    keep_all = pipeline.run(rgb, dist, valid, calib, bb, MockResidual(),
                            confidence=conf, min_confidence=-1)
    reject = pipeline.run(rgb, dist, valid, calib, bb, MockResidual(),
                          confidence=conf, min_confidence=6)
    assert reject['n_anchors'] < keep_all['n_anchors'], \
        "raising min_confidence should reject the weak (conf=3) zones"
    assert reject['n_anchors'] > 0


def test_anchoring_roundtrip():
    """Core least-squares math: if inv-depth is exactly affine in disparity, the
    fit must recover the generating (a, b)."""
    rng = np.random.default_rng(1)
    disp = rng.uniform(0.2, 1.5, 500)
    a0, b0 = 1.7, -0.35
    s = a0 * disp + b0
    w = np.ones_like(disp)

    a, b = anc.solve_scale_shift(disp, s, w)
    assert abs(a - a0) < 1e-6 and abs(b - b0) < 1e-6
    a, b = anc.solve_robust(disp, s, w, iters=2)
    assert abs(a - a0) < 1e-6 and abs(b - b0) < 1e-6


# --- plain runner (no pytest needed) ----------------------------------------

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

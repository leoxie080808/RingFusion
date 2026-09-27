#!/usr/bin/env python3
"""T1 fallback: time OUR pipeline with the networks in PyTorch instead of TensorRT.

Why this exists. Table I compared our TensorRT-FP16 pipeline against DEPTHOR timed in
PyTorch FP32 -- two advantages stacked (runtime AND precision) and only one of them is a
property of the method. DEPTHOR cannot be converted to TensorRT (see t1_conversion_failure
.json), so the fallback in T1 is to bring OUR side down to PyTorch and compare there.

Like-for-like is enforced structurally: this swaps ONLY the network execution. Both
adapters keep the deployed classes' own _preprocess/_pack/refine code by duck-typing
TRTRunner.run(), so resize, normalisation, anchor re-splatting and the affine apply are
byte-identical to the TensorRT path.

The engines were exported from exactly these checkpoints (student_v4_heldout ->
student_v4_heldout.onnx, residual_v7_long -> residual_v7_fov73.onnx), so it is the same
weights either way.

FP16 is run BOTH ways per network -- .half() and torch.autocast -- and the faster is
used, so a reviewer cannot argue FP16 was set up badly. Which one won is recorded.
"""
import argparse, json, os, statistics, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, 'training'))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws/src/ringfusion_perception'))
import envinfo                                                    # noqa: E402
import torch                                                      # noqa: E402
from anchoring_bridge import calib_from_yaml                      # noqa: E402
from ringfusion_perception import pipeline, roi                   # noqa: E402
from ringfusion_perception.backbone import TensorRTBackbone       # noqa: E402
from ringfusion_perception.residual import ResidualRefiner        # noqa: E402
from models.student import DepthStudent                           # noqa: E402
from models.residual import ResidualRefinerNet                    # noqa: E402


class TorchRunner:
    """Duck-types TRTRunner.run(np) -> np, executing the net in PyTorch."""
    def __init__(self, net, mode):
        self.net, self.mode = net, mode
    def run(self, x_np):
        x = torch.from_numpy(np.ascontiguousarray(x_np)).cuda()
        if self.mode == 'half':
            x = x.half()
        with torch.no_grad():
            if self.mode == 'autocast':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    y = self.net(x)
            else:
                y = self.net(x)
        return y.float().cpu().numpy()


def make_backbone(ckpt, mode, hw=(288, 384)):
    net = DepthStudent(pretrained=False).cuda().eval()
    net.load_state_dict(torch.load(ckpt, map_location='cuda'))
    if mode == 'half':
        net = net.half()
    o = TensorRTBackbone.__new__(TensorRTBackbone)     # bypass TRT construction
    o.h, o.w = hw
    o._engine = TorchRunner(net, mode)
    return o, net


def make_residual(ckpt, mode, hw=(288, 384)):
    net = ResidualRefinerNet().cuda().eval()
    sd = torch.load(ckpt, map_location='cuda')
    net.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) else sd)
    if mode == 'half':
        net = net.half()
    o = ResidualRefiner.__new__(ResidualRefiner)
    o.h, o.w = hw
    o._engine = TorchRunner(net, mode)
    return o, net


def bench_net(net, shape, mode, iters=200, warmup=50):
    """Isolated per-network timing, used only to pick half vs autocast."""
    x = torch.randn(*shape, device='cuda')
    if mode == 'half':
        x = x.half()
    def f():
        with torch.no_grad():
            if mode == 'autocast':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    return net(x)
            return net(x)
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t = time.perf_counter()
        f()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-dir', required=True)
    ap.add_argument('--tof-dir', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--backend', choices=['torch', 'tensorrt'], default='torch',
                    help='torch = PyTorch FP16 (matched-precision arm); tensorrt = the '
                         'deployed engines, timed under the IDENTICAL protocol so the two '
                         'rows differ only by execution backend.')
    ap.add_argument('--student-ckpt', default='')
    ap.add_argument('--residual-ckpt', default='')
    ap.add_argument('--backbone-engine', default='')
    ap.add_argument('--residual-engine', default='')
    ap.add_argument('--size', default='480x640', help='HxW pipeline input')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import cv2
    H, W = (int(v) for v in a.size.split('x'))

    if a.backend == 'tensorrt':
        backbone = TensorRTBackbone(a.backbone_engine)
        residual = ResidualRefiner(a.residual_engine)
        sel = {'backbone': 'tensorrt_fp16', 'residual': 'tensorrt_fp16'}
        modes = {'note': 'deployed engines; FP16 mode selection does not apply'}
        print(f'backend=tensorrt  {os.path.basename(a.backbone_engine)} + '
              f'{os.path.basename(a.residual_engine)}')
        return _time_and_write(a, backbone, residual, sel, modes)

    # ---- pick the faster FP16 mode per network -------------------------------
    sel, modes = {}, {}
    for name, ctor, shape in (
            ('backbone', lambda m: make_backbone(a.student_ckpt, m)[1], (1, 3, 288, 384)),
            ('residual', lambda m: make_residual(a.residual_ckpt, m)[1], (1, 6, 288, 384))):
        r = {}
        for m in ('half', 'autocast'):
            try:
                r[m] = round(bench_net(ctor(m), shape, m), 3)
            except Exception as e:
                r[m] = f'FAILED: {type(e).__name__}: {e}'
        ok = {k: v for k, v in r.items() if isinstance(v, float)}
        best = min(ok, key=ok.get) if ok else 'fp32'
        sel[name], modes[name] = best, r
        print(f'{name}: ' + '  '.join(f'{k}={v}' for k, v in r.items()) + f'   -> using {best}')

    backbone, _ = make_backbone(a.student_ckpt, sel['backbone'])
    residual, _ = make_residual(a.residual_ckpt, sel['residual'])
    return _time_and_write(a, backbone, residual, sel, modes)


def _time_and_write(a, backbone, residual, sel, modes):
    import cv2
    H, W = (int(v) for v in a.size.split('x'))
    # ---- load one real frame, resized to the comparison resolution -----------
    stems = sorted(f for f in os.listdir(a.rgb_dir) if f.endswith(('.png', '.jpg')))
    rgb = cv2.cvtColor(cv2.imread(os.path.join(a.rgb_dir, stems[0])), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    z = np.load(os.path.join(a.tof_dir, os.path.splitext(stems[0])[0] + '.npz'))
    dist, conf = z['dist_m'].astype(np.float32), z['confidence']
    calib = calib_from_yaml(a.calib, (H, W))

    results = {}
    for label, kw in (('stages_off', dict(blend=False, roi_enable=False)),
                      ('deployed',   dict(blend=True,  roi_enable=True))):
        # The deployed node holds ONE PlaneTracker for its lifetime (perception_node.py:100).
        # Passing plane_tracker=None makes pipeline.run re-RANSAC the ground plane from
        # scratch every frame -- a bug the node already fixed, and ~10% of the deployed
        # frame time. Timing without it does not describe the shipped configuration.
        tracker = roi.PlaneTracker(refit_every=1) if kw['roi_enable'] else None
        def one(_t=tracker, _kw=kw):
            return pipeline.run(rgb, dist, np.isfinite(dist), calib, backbone, residual,
                                confidence=conf, min_confidence=-1,
                                plane_tracker=_t, **_kw)
        for _ in range(a.warmup):
            one()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.iters):
            torch.cuda.synchronize(); t = time.perf_counter()
            one()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
        med = statistics.median(ts)
        results[label] = {
            'median_ms': round(med, 2), 'mean_ms': round(statistics.mean(ts), 2),
            'sd_ms': round(statistics.pstdev(ts), 3),
            'p05_ms': round(float(np.percentile(ts, 5)), 2),
            'p95_ms': round(float(np.percentile(ts, 95)), 2),
            'hz': round(1000.0 / med, 2)}
        print(f"{label:11s} median {med:7.2f} ms  p5 {results[label]['p05_ms']:6.2f}  "
              f"p95 {results[label]['p95_ms']:6.2f}  {results[label]['hz']:5.2f} Hz")

    rep = {'label': ('RingFusion pipeline, networks in PyTorch FP16 (T1 matched-precision arm)'
                     if a.backend == 'torch' else
                     'RingFusion pipeline, deployed TensorRT FP16 engines, T1 protocol'),
           'backend': a.backend,
           'input_res': f'{H}x{W}', 'fp16_mode_selected': sel, 'fp16_mode_benchmark_ms': modes,
           'weights': {'student': a.student_ckpt or a.backbone_engine,
                       'residual': a.residual_ckpt or a.residual_engine},
           'procedure': {'warmup_iters': a.warmup, 'timed_iters': a.iters,
                         'cuda_synchronized': True, 'batch_reused': True,
                         'plane_tracker': 'PlaneTracker(refit_every=1), as deployed',
                         'gpu_otherwise_idle': 'asserted by operator'},
           'configs': results, 'env': envinfo.capture(note='time_pipeline_torch')}
    if a.out:
        json.dump(rep, open(a.out, 'w'), indent=1); print('wrote', a.out)


if __name__ == '__main__':
    main()

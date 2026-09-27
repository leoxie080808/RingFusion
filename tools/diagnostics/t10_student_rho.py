#!/usr/bin/env python3
"""T10: distillation fidelity of the DEPLOYED backbone.

The paper's rho = 0.996 came from the 2000-image PILOT student, not student_v4_heldout,
which is what the robot runs. This scores the deployed backbone on the 200 frames in
heldout_stems.txt -- frames distill_backbone.py was told to EXCLUDE (--exclude-stems-file),
so the student never saw them.

All 200 are valid here: this tests the BACKBONE against the teacher. The N2 contamination
caveat concerns the refiner, which is not involved.

Metrics are eval_student.py's, unchanged: Pearson rho, SSI-MAE (= val_ssi), AbsRel and
d1.25, each after the same least-squares scale+shift alignment training used.
"""
import argparse, glob, json, os, sys
import numpy as np, cv2, torch

_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(_R, 'training'),
                os.path.join(_R, 'ros2_ws/src/ringfusion_perception'),
                os.path.dirname(os.path.abspath(__file__))]
import envinfo
from data import load_image01, normalize
from models.student import DepthStudent


def align(s, t):
    A = np.stack([s.ravel(), np.ones(s.size, np.float64)], 1)
    ab, *_ = np.linalg.lstsq(A, t.ravel().astype(np.float64), rcond=None)
    return (ab[0] * s + ab[1]).astype(np.float32)


def score(infer, stems, images_dir, cache_dir, H, W):
    rhos, maes, absrels, d1s, used = [], [], [], [], 0
    for st in stems:
        p = os.path.join(images_dir, f'frame_{st}.png')
        tpath = os.path.join(cache_dir, f'frame_{st}.npy')
        if not (os.path.exists(p) and os.path.exists(tpath)):
            continue
        s = infer(p, H, W)
        t = cv2.resize(np.load(tpath).astype(np.float32), (W, H))
        rhos.append(float(np.corrcoef(s.ravel(), t.ravel())[0, 1]))
        sa = align(s, t)
        maes.append(float(np.abs(sa - t).mean()))
        eps = 1e-2
        sd, td = 1.0 / np.clip(sa, eps, None), 1.0 / np.clip(t, eps, None)
        absrels.append(float((np.abs(sd - td) / td).mean()))
        ratio = np.maximum(sd / td, td / sd)
        d1s.append(float((ratio < 1.25).mean()))
        used += 1
    f = lambda a: round(float(np.mean(a)), 4)
    return {'n_frames': used, 'rho': f(rhos), 'val_ssi_mae': f(maes),
            'absrel': f(absrels), 'd1_25': f(d1s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stems-file', required=True)
    ap.add_argument('--images', default='data/rect'); ap.add_argument('--cache', default='data/teacher')
    ap.add_argument('--ckpt', default=''); ap.add_argument('--engine', default='')
    ap.add_argument('--size', type=int, nargs=2, default=[288, 384])
    ap.add_argument('--control-prep', action='store_true',
                    help='add an FP32 arm using the deployed preprocessing, to separate '
                         'preprocessing effects from quantisation effects')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    H, W = a.size
    stems = [l.strip() for l in open(a.stems_file) if l.strip()]
    print(f'{len(stems)} held-out stems from {os.path.basename(a.stems_file)}')

    arms = {}
    if a.ckpt:
        m = DepthStudent(pretrained=False).cuda().eval()
        m.load_state_dict(torch.load(a.ckpt, map_location='cuda'))
        def infer_pt(p, H, W):
            x = torch.nn.functional.interpolate(normalize(load_image01(p)).unsqueeze(0),
                                                size=(H, W), mode='bilinear',
                                                align_corners=False).cuda()
            with torch.no_grad():
                return m(x)[0, 0].cpu().numpy().astype(np.float32)
        arms['fp32_checkpoint'] = (infer_pt, os.path.basename(a.ckpt))
    if a.ckpt and a.control_prep:
        # CONTROL: same FP32 weights, but the DEPLOYED preprocessing (cv2.INTER_AREA +
        # ImageNet norm) instead of the training path (F.interpolate bilinear). Isolates
        # preprocessing from precision -- if this lands near the FP16 row, the gap
        # between the first two arms is the resize, not the quantisation.
        _mean = np.array([0.485, 0.456, 0.406], np.float32)
        _std = np.array([0.229, 0.224, 0.225], np.float32)
        def infer_ctl(p, H, W):
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(
                ((img - _mean) / _std).transpose(2, 0, 1)[None])).cuda()
            with torch.no_grad():
                return m(x)[0, 0].cpu().numpy().astype(np.float32)
        arms['fp32_with_deployed_preprocessing'] = (infer_ctl, os.path.basename(a.ckpt) + ' (deployed prep)')

    if a.engine:
        from ringfusion_perception.backbone import TensorRTBackbone
        bb = TensorRTBackbone(a.engine, input_hw=(H, W))
        def infer_trt(p, H, W):
            # Take the engine's NATIVE (H,W) output. bb.infer() upsamples to the source
            # image size, and resizing that back down would resample twice where the
            # FP32 arm resamples once -- an artefact of the harness, not of FP16.
            # Preprocessing stays bb._preprocess: cv2.INTER_AREA + ImageNet norm, i.e.
            # exactly what the robot feeds the engine.
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            return bb._engine.run(bb._preprocess(rgb))[0, 0].astype(np.float32)
        arms['fp16_tensorrt_deployed'] = (infer_trt, os.path.basename(a.engine))

    res = {}
    for name, (fn, w) in arms.items():
        r = score(fn, stems, a.images, a.cache, H, W); r['weights'] = w
        res[name] = r
        print(f"\n{name}  ({w})")
        print(f"  frames  {r['n_frames']}")
        print(f"  rho     {r['rho']:.4f}   (Pearson; pilot recorded 0.9962)")
        print(f"  val_ssi {r['val_ssi_mae']:.4f}   (pilot 3.51)")
        print(f"  AbsRel  {r['absrel']:.4f}")
        print(f"  d1.25   {r['d1_25']:.4f}   (pilot 0.89)")

    n_all = len(glob.glob(os.path.join(a.images, '**', '*.png'), recursive=True))
    rep = {'label': 'T10: deployed student distillation fidelity on the 200 excluded frames',
           'pilot_for_reference': {'rho': 0.9962, 'val_ssi': 3.51, 'd1_25': 0.89,
                                   'corpus': '2000-image pilot', 'note': 'what the paper currently cites'},
           'distillation_corpus': {'total_frames': n_all, 'excluded_stems': len(stems),
                                   'available_to_distillation': n_all - len(stems)},
           'arms': res, 'env': envinfo.capture(note='t10_student_rho')}
    print(f"\ndistillation corpus: {n_all} total, {len(stems)} excluded "
          f"-> {n_all-len(stems)} available to distillation")
    if a.out:
        json.dump(rep, open(a.out, 'w'), indent=1); print('wrote', a.out)


if __name__ == '__main__':
    main()

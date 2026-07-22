"""Export a trained student or residual checkpoint to ONNX with FIXED shapes.

Static shapes let TensorRT pick optimal kernels and pre-plan memory (§3.3, §7.1).
Verifies PyTorch vs ONNXRuntime parity (max abs diff should be < 1e-4) when
onnxruntime is available.

Usage:
    python tools/export_onnx.py --ckpt runs/student/student_best.pth --arch student \
        --input-hw 288 384 --out student.onnx
    python tools/export_onnx.py --ckpt runs/residual/residual_best.pth --arch residual \
        --input-hw 288 384 --out residual.onnx
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))
from models.student import DepthStudent          # noqa: E402
from models.residual import ResidualRefinerNet   # noqa: E402


def build(arch, ckpt, device):
    if arch == 'student':
        model = DepthStudent(pretrained=False)
        in_ch = 3
    elif arch == 'residual':
        model = ResidualRefinerNet()
        in_ch = 6
    else:
        raise ValueError(arch)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval(), in_ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arch', required=True, choices=['student', 'residual'])
    ap.add_argument('--input-hw', type=int, nargs=2, default=[288, 384])
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--no-simplify', action='store_true')
    args = ap.parse_args()

    h, w = args.input_hw
    model, in_ch = build(args.arch, args.ckpt, args.device)
    dummy = torch.randn(args.batch, in_ch, h, w, device=args.device)

    # dynamo=False forces the legacy TorchScript exporter. Torch >=2.9 defaults to
    # the new torch.export-based exporter, which needs the `onnxscript` package (not
    # in the Jetson's pinned torch env) and can emit ops the TensorRT parser dislikes.
    # For these plain-conv nets the legacy path is well-tested and produces clean ONNX.
    torch.onnx.export(
        model, dummy, args.out,
        input_names=['input'], output_names=['output'],
        opset_version=args.opset, do_constant_folding=True, dynamo=False)
    print(f"exported {args.arch} -> {args.out}  (input {tuple(dummy.shape)})")

    if not args.no_simplify:
        try:
            import onnx
            from onnxsim import simplify
            model_onnx = onnx.load(args.out)
            model_simp, ok = simplify(model_onnx)
            if ok:
                onnx.save(model_simp, args.out)
                print("onnxsim: simplified")
        except Exception as e:                       # noqa: BLE001
            print(f"onnxsim skipped ({e})")

    # parity check
    try:
        import onnxruntime as ort
        with torch.no_grad():
            ref = model(dummy).cpu().numpy()
        sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        got = sess.run(None, {'input': dummy.cpu().numpy()})[0]
        diff = float(np.abs(ref - got).max())
        print(f"PyTorch vs ONNXRuntime max abs diff: {diff:.2e} "
              f"({'OK' if diff < 1e-4 else 'HIGH -- investigate before building engine'})")
    except ImportError:
        print("onnxruntime not installed; skipped parity check")


if __name__ == '__main__':
    main()

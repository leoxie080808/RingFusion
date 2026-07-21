# RingFusion training

Off-robot training for the two learned components, plus export to TensorRT. See
`RingFusion_technical_reference_updateP2.md` for the full design. Everything here
runs on a desktop GPU or on the Orin; only `tools/build_engine.py` must run on the
Jetson (engines are hardware-specific).

```
training/
├── models/
│   ├── student.py        Network A: MobileNetV3-Large + DPT-lite  (infer -> disparity)
│   └── residual.py       Network B: g_phi U-Net  (zero-init -> identity at step 0)
├── losses.py             SSI + gradient (distill); log-depth + NLL + coverage (residual)
├── data.py               DistillDataset (+ shared image/normalize helpers)
├── residual_data.py      ResidualDepthDataset (rgb + dense GT depth)
├── anchoring_bridge.py   reuses the deployed geometry/anchoring; simulate_tof
├── cache_teacher.py      Depth Anything V2 -> cached fp16 disparity targets
├── distill_backbone.py   teacher -> student
└── train_residual.py     residual + NLL calibration
tools/
├── export_onnx.py        PyTorch -> ONNX (+ parity check)
└── build_engine.py       ONNX -> TensorRT INT8/FP16   (run on the Jetson)
```

The residual training reuses the **exact** geometry + anchoring the robot runs at
inference (imported from `ros2_ws/src/ringfusion_perception`), so the net learns to
correct the fit it will actually see. Single source of truth, no duplicated math.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
# On the Orin: install torch/torchvision from JetPack wheels instead of PyPI.
```

### GPU torch on the Orin — working recipe (RESOLVED)

A plain PyPI `torch` on the Jetson has a **broken cuBLAS** — `CUBLAS_STATUS_ALLOC_FAILED`
on the first matmul even though `torch.cuda.is_available()` is True. Cause: PyPI torch drags
in pip CUDA libs (`nvidia-cublas-cu12` **12.9**, cu13 variants) that need a newer CUDA driver
than the Jetson's **12.6** (the driver ships with JetPack and can't be upgraded separately).
The Jetson's own CUDA 12.6 has cuBLAS/cuDNN/cuSPARSE/cuSOLVER; torch just needs to use them.

Recipe that works on **L4T 36.5 / CUDA 12.6 / py3.10 / Orin** (verified: cuBLAS + cuDNN pass):

```bash
# 1. Remove PyPI torch + ALL the mismatched pip CUDA libs it pulled in
pip3 uninstall -y $(pip3 list | grep -iE "^torch |^torchvision |^nvidia-" | awk '{print $1}')

# 2. Jetson-native torch/torchvision, --no-deps so it does NOT re-pull pip CUDA libs
pip3 install --user --no-deps torch==2.11.0 torchvision==0.26.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 3. The one lib not in system CUDA: libcudss.so.0 (cuDSS). It drags cublas-cu12 back in,
#    so install then remove that (torch then uses SYSTEM cuBLAS 12.6).
pip3 install --user nvidia-cudss-cu12
pip3 uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 cuda-toolkit

# 4. Verify
python3 -c "import torch; a=torch.randn(64,64,device='cuda'); print('cuBLAS OK', (a@a).sum().item())"
```

> **⚠ Gotcha:** never let anything `pip install nvidia-cublas-cu12` — it's built for a newer
> CUDA than the Orin's 12.6 driver and re-breaks cuBLAS. torch must use the **system** cuBLAS.
> Also ensure `pillow>=10` (older Pillow breaks recent `transformers`).

## 1. Backbone (Network A) — needs images only

The teacher generates the targets, so distillation needs **no measured depth**.

**Why images alone are enough — and why unusual objects don't break it.** Monocular
depth is *relative only*: from a single image a small near object and a large far one
are pixel-identical (scale ambiguity), so the teacher — and therefore the student —
predicts disparity *up to a global scale + shift*, **never metres**. It recovers that
relative structure from object-*agnostic* geometric cues (occlusion, perspective,
ground contact, relative size, texture gradient, shading), **not** object recognition,
so it generalizes to machinery and uncommon items it has never seen — they still
occlude, sit on the floor, and show perspective. Absolute scale is never the network's
job: the **ToF supplies it at inference** and the closed-form anchoring (Stage 5)
stretches the relative map onto those real ranges. Two consequences for training:
(1) targets are free — images only, no measured depth; (2) where the teacher genuinely
fails (glass, mirrors, sky, blank walls — no geometric cues), the SSI loss **trims the
worst 20%** so the student never learns those mistakes. Always validate the teacher on
*your own* rectified frames first — that is the Step-0 sanity check (`step0.png`).

```bash
# Collect ~20k RECTIFIED frames (rectify first — the teacher is garbage on raw
# fisheye, and that garbage becomes your target). Then:

# a) cache teacher disparity (expensive, once)
python training/cache_teacher.py --images data/rect --cache data/teacher \
    --model depth-anything/Depth-Anything-V2-Large-hf

# b) distill the student
python training/distill_backbone.py --images data/rect --cache data/teacher \
    --out runs/student --epochs 60 --batch 16
# -> runs/student/student_best.pth
```

## 2. Residual (Network B) — needs measured depth

Start on synthetic renders (Hypersim/Replica/Blender) where depth is exact; ToF is
simulated from the GT. Fine-tune on real GT later.

```
data/syn/rgb/*.png      data/syn/depth/*.npy   (matching stems; depth in metres)
```

```bash
python training/train_residual.py --rgb data/syn/rgb --depth data/syn/depth \
    --student-ckpt runs/student/student_best.pth --out runs/residual \
    --epochs 40 --hfov 90
# watch coverage -> 0.68 (calibrated). --depth-scale 0.001 for 16-bit-mm PNGs.
```

The residual is **zero-initialized to the identity**, so you can ship the closed-form
system before it is trained — an untrained residual changes nothing.

## 3. Export → TensorRT (build engines on the Jetson)

```bash
python tools/export_onnx.py --ckpt runs/student/student_best.pth  --arch student  --out student.onnx
python tools/export_onnx.py --ckpt runs/residual/residual_best.pth --arch residual --out residual.onnx

# ON THE ORIN:
python tools/build_engine.py --onnx student.onnx  --out student_int8.engine \
    --precision int8 --calib-dir calib_images/ --calib-cache student.cache   # REAL images
python tools/build_engine.py --onnx residual.onnx --out residual_fp16.engine --precision fp16
```

Then run perception with the engines (no code change):

```bash
ros2 launch ringfusion_bringup single_module.launch.py \
    backbone_engine:=/path/student_int8.engine residual_engine:=/path/residual_fp16.engine
```

## Notes / gotchas

- **Rectify before the teacher.** The single most common way to poison the dataset.
- **INT8 the backbone, FP16 the residual.** Quantizing the variance head wrecks calibration.
- **Cache the teacher once.** It never changes; caching turns days into hours.
- **Parameter counts** print when a model is built (`python training/models/student.py`);
  replace the paper's ~6.1M / ~0.5M placeholders with the measured values.
- **No vertical-flip augmentation** — monocular depth priors are gravity-dependent.

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
├── residual_data.py      ResidualDepthDataset (rgb + dense GT depth; synthetic path)
├── residual_real_data.py ResidualRealDataset (rgb + raw 32x32 ToF; real held-out-anchor path)
├── anchoring_bridge.py   reuses the deployed geometry/anchoring; simulate_tof,
│                         calib_from_yaml, build_real_supervision (held-out anchors)
├── cache_teacher.py      Depth Anything V2 -> cached fp16 disparity targets
├── distill_backbone.py   teacher -> student
├── train_residual.py     residual + NLL calibration
├── compare_student.py    student vs teacher depth montage (eyeball check)
└── eval_student.py       distillation-fidelity metrics (rho / SSI-MAE / d1.25)
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
# Collect ~20k RECTIFIED frames FROM THE DEPLOYMENT ENVIRONMENT (not a generic/online
# dataset — the student only gets good at scenes that look like its training set, and
# you can't rectify a normal photo to mimic this fisheye). Rectify first — the teacher
# is garbage on raw fisheye, and that garbage becomes your target. Use the ROS collector
# `ros2 run ringfusion_perception collect_frames` (see ros2_ws/README.md "Collecting
# training images") — it saves rectified frames through the exact deployment path, with
# dedup + blur rejection. Diversity > volume. Then:

# a) cache teacher disparity (expensive, once)
python training/cache_teacher.py --images data/rect --cache data/teacher \
    --model depth-anything/Depth-Anything-V2-Large-hf

# b) distill the student
python training/distill_backbone.py --images data/rect --cache data/teacher \
    --out runs/student --epochs 60 --batch 16
# -> runs/student/student_best.pth
```

**Expected training behaviour:** `val_ssi` may sit on a **flat plateau for the first
~10-15 epochs** (the scale-shift alignment can't lock onto a structureless early
student, so the loss parks at ~the mean target magnitude) and then **collapse quickly**
once the student develops structure. That plateau-then-drop is normal — don't kill the
run during the flat part.

### Validate the distilled student

A low loss alone can hide blur/artifacts, so **check the student before exporting**:

```bash
# eyeball: montage [ RGB | student | teacher ] for N frames spread across the set
python training/compare_student.py --ckpt runs/student/student_best.pth --n 5   # -> student_vs_teacher.png

# scorecard: mimicry metrics on the SAME held-out val split training used
python training/eval_student.py --ckpt runs/student/student_best.pth
```

Track **rho** (Pearson correlation with the teacher — structure match; >0.98 excellent)
as the mimicry metric, **d1.25** (fraction within 25% of the teacher) secondary. These
measure fidelity **to the teacher**, not correctness — real accuracy needs measured
ground truth (residual + system validation), and AbsRel is only meaningful there.
Re-run after any retrain to quantify the change. **Pilot baseline (2000 imgs):
rho 0.9962, d1.25 0.89, val_ssi 3.51.**

## 2. Residual (Network B) — needs measured depth

Network B predicts a per-pixel correction `(da, db)` to the closed-form affine fit
plus a calibrated variance `tau²`. It is **zero-initialized to the identity**, so you
can ship the closed-form system before it is trained — an untrained residual changes
nothing. Two ways to train it:

### 2a. Real data — held-out ToF anchors (RECOMMENDED here)

No synthetic→real domain gap, and it's the *only* source of the real backbone's real
error statistics (what actually calibrates `tau²`). Collect paired `(rectified image,
32×32 ToF)` logs by pushing the robot around the deployment environment.

**Why not just use the ToF as dense GT?** A sparse sensor can't densely supervise a
dense residual: its zones coincide with the anchors the closed-form fit already nails,
so supervising there teaches the identity. Instead each frame's valid zones are split —
an **anchor set** drives the fit + feeds the net (exactly as the robot runs), and a
disjoint **hold-out set** becomes a sparse target at pixels the net had *no* input for.
That's true generalization signal, in-environment, real error stats. (`build_real_supervision`
guarantees the two sets are pixel-disjoint — no target leakage.) The ToF FOV (~73.5°×60.5°)
is narrower than the fisheye, so the periphery gets no supervision and B safely stays
identity there — acceptable, since that's exactly where it has no information anyway.

> ⚠ **Network B must be re-trained (2026-08-03).** `build_real_supervision` places anchors
> via `geo.zone_directions(cols, rows, calib['fov_h'], calib['fov_v'])`, and every existing
> checkpoint was trained with `fov_h` at 45° — a value tape measurement has since put at
> **73.5°**. Two of B's four inputs, `anchor_depth` and `anchor_mask`, are now generated
> differently at inference than they were in training: a train/deploy mismatch, not just a
> stale number. Also update `anchoring_bridge.default_calib`, which still defaults to
> `tof_fov=(61.0, 45.0)`. See
> [`ros2_ws/README.md`](../ros2_ws/README.md#tof-field-of-view-measured-against-tape).

Capture format (matched by stem; produced by the paired ToF+image logger):
```
data/real/rgb/<...>/000123.png    # the RECTIFIED (pinhole) image, as the robot feeds the net
data/real/tof/<...>/000123.npz    # np.savez(dist_m=(32,32) float32 NaN-invalid, confidence=(32,32) uint8)
```
Prefer stationary "stations" (stop, average a few ToF frames, snap one image) for clean
pairs; slow rolling works if you pair each ToF frame with the nearest-in-time image.
Vary geometry (near/far walls, clutter, corners); the loss only sees ToF-covered pixels,
so coverage across scenes matters more than raw frame count.

```bash
python training/train_residual.py --real \
    --rgb data/real/rgb --tof data/real/tof \
    --calib ros2_ws/src/ringfusion_bringup/config/calibration.yaml \
    --student-ckpt runs/student/student_best.pth --out runs/residual \
    --epochs 40 --holdout-frac 0.25
# --calib rebuilds the SAME rectified pinhole K_rect the robot uses, scaled to 288x384.
# watch coverage -> 0.68 (calibrated).
```

### 2b. Synthetic pretrain (optional bootstrap)

Renders (Hypersim/Replica/Blender) where depth is exact; ToF is *simulated* from the GT.
Useful to warm-start before real fine-tuning, but mind the fisheye/FOV domain gap —
render at matching intrinsics or it learns the wrong periphery behaviour.
```
data/syn/rgb/*.png      data/syn/depth/*.npy   (matching stems; depth in metres)
```
```bash
python training/train_residual.py --rgb data/syn/rgb --depth data/syn/depth \
    --student-ckpt runs/student/student_best.pth --out runs/residual \
    --epochs 40 --hfov 90
# --depth-scale 0.001 for 16-bit-mm PNGs.
```

### Testing Network B

1. **Identity sanity** — untrained B must equal the closed-form baseline (zero-init); B must never regress *below* it.
2. **Accuracy** — B's depth vs the held-out ToF zones (AbsRel/RMSE at hold-out pixels), gains concentrated at edges.
3. **Calibration** — `coverage` → **0.68** @1σ (the headline number); reported live during training.
4. **On-robot** — build the FP16 engine (§3), `residual_engine:=…`, confirm `/depth` rate holds and the variance field is sane.

### First training run — results & concerns (2026-07-23)

First end-to-end Network B run: **563 real paired frames** (collected with `paired_logger`
`auto` mode, well-lit, median 818 valid ToF zones/frame), trained 40 epochs, `--holdout-frac 0.25`,
exported to `residual_fp16.engine`, integrated live (`residual=residual_trt`).

| Metric | Result | Notes |
|---|---|---|
| Best val loss | **0.263** | converged by ~epoch 25 |
| **Coverage** | **~0.88** | plateaued; did **not** reach 0.68 (see concern 3) |
| B correction \|A+B − A\| | **median 41 mm** | B makes genuine local corrections (~8% on a 0.5 m scene) |
| B correction max | **~10 km (clamped)** | ⚠️ degenerate pixels (concern 1) |
| Uncertainty (std) | **median ~490 mm** | large — the under-confidence |
| Live rate `/depth` | **~7 Hz** (was ~15) | B's apply is on CPU (concern 2) |

**B works** (trained → exported → live, making real corrections), but it is **not yet a clean
win**. Three concerns to resolve before trusting it:

1. **Degenerate pixels (heavy tail).** Median correction is a healthy 41 mm, but the max hits
   the 10 km far-clamp: at some pixels `(a+da)·disp + (b+db)` → ≈0, blowing depth up. Those junk
   points pollute the cloud. Signals B is **under-constrained** — wants more/varied data and/or a
   **regularizer or clamp on `da, db`** magnitude.
2. **Rate 15 → 7 Hz.** `ResidualRefiner.refine()` upsamples 3 fields to 2 MP and applies them on
   the **CPU** (not GPU-offloaded like the main pipeline). Recoverable by moving the apply to
   `gpu_ops` — same fix already done for the closed-form tail. Not a training issue.
3. **Calibration capped at ~0.88 (under-confident).** Total variance = `analytic + tau²` with
   `tau² ≥ 0`, so B can only **add** uncertainty — it can't shrink an already-over-conservative
   analytic variance down to 0.68. Errs on the safe side (over- not under-estimating uncertainty),
   but not tight. Fixing calibration means letting B **scale** the analytic variance, not just add.

**Go/no-go still pending:** the **held-out-anchor accuracy eval** — does A+B predict the held-out
ToF zones *more accurately* than A-only? We have B's correction *magnitude* (41 mm) but not its
*sign*. If A+B beats A-only → keep B, fix the tail + rate. If not → **re-collect** more/varied data
and add a stability regularizer, then retrain.

## 3. Export → TensorRT (build engines on the Jetson)

**3a. Export to ONNX.** `export_onnx.py` forces the **legacy TorchScript exporter**
(`dynamo=False`). Torch >= 2.9 defaults to the new torch.export exporter, which needs
the `onnxscript` package — absent from the Jetson's pinned torch env, so the default
fails with `ModuleNotFoundError: No module named 'onnxscript'`. The legacy path needs
no extra install and emits cleaner ONNX for the TensorRT parser on these plain-conv nets.
(If you still hit the onnxscript error, you're on an older copy of the tool.)

```bash
python tools/export_onnx.py --ckpt runs/student/student_best.pth  --arch student  --out student.onnx
python tools/export_onnx.py --ckpt runs/residual/residual_best.pth --arch residual --out residual.onnx
```

A `DeprecationWarning` about the legacy exporter is expected — ignore it. **Watch the
parity check:** `PyTorch vs ONNXRuntime max abs diff: <n> (OK)` — `OK` (< 1e-4) means
the ONNX matches the PyTorch model. If it prints `HIGH`, fix that before building an engine.

**3b. Build engines (ON THE ORIN — engines are hardware-specific).** De-risk in two
passes: a quick FP16 build to prove the path works, then INT8 for the headline speed.

```bash
# smoke test first (fast, no calibration)
python tools/build_engine.py --onnx student.onnx --out student_fp16.engine --precision fp16
# then INT8 (calibrates on REAL rectified images; INT8 the backbone, FP16 the residual)
python tools/build_engine.py --onnx student.onnx  --out student_int8.engine \
    --precision int8 --calib-dir data/rect --calib-cache student.cache
python tools/build_engine.py --onnx residual.onnx --out residual_fp16.engine --precision fp16
```

**3c. Run perception with the engine + measure FPS** (no code change — swaps out the
mocks; retires `MockBackbone`):

```bash
ros2 launch ringfusion_bringup single_module.launch.py \
    backbone_engine:=$HOME/RingFusion/student_int8.engine
# add residual_engine:=$HOME/RingFusion/residual_fp16.engine once the residual is trained
ros2 topic hz /depth      # headline throughput (/cloud also works)
```

Before the full launch, smoke-test the **runtime** in isolation (it has clear errors
vs a buried ROS failure): load an engine and run one inference through `TRTRunner`.

**3d. Export/engine gotchas (all fixed in-tree — documented so nobody re-hits them):**

- **ONNX exporter** — `export_onnx.py` pins the legacy exporter (`dynamo=False`). The
  torch >= 2.9 default routes through `onnxscript`, which is absent from the Jetson's
  pinned torch env → `ModuleNotFoundError: No module named 'onnxscript'`.
- **Student encoder trace-safety** — the encoder taps **fixed integer block indices**
  (found once at build time), not a dict keyed by `stride = h0 // feat_h`. Under ONNX
  tracing the shape-derived keys become traced tensors and the plain-int lookup misses
  → `KeyError: 4`. Keep it integer-indexed (see `models/student.py`).
- **No pycuda on the Jetson** — it isn't installed and is painful to build. BOTH the INT8
  calibrator (`build_engine.py`) and the runtime (`ros2_ws/.../trt_util.TRTRunner`) use
  **torch CUDA tensors** — `.data_ptr()` for the device address, `torch.cuda.Stream().cuda_stream`
  for the stream — instead of pycuda. **Never `pip install pycuda`**; torch is the CUDA runtime here.
- **Engines are not portable** — hardware- and TensorRT-version-specific. Always build on the target Orin.

## Notes / gotchas

- **Rectify before the teacher.** The single most common way to poison the dataset.
- **INT8 the backbone, FP16 the residual.** Quantizing the variance head wrecks calibration.
- **Cache the teacher once.** It never changes; caching turns days into hours.
- **Parameter counts** print when a model is built (`python training/models/student.py`);
  replace the paper's ~6.1M / ~0.5M placeholders with the measured values.
- **No vertical-flip augmentation** — monocular depth priors are gravity-dependent.

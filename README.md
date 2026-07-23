# RingFusion

Real-time metric depth + point clouds from **one wide-angle camera and a sparse
multizone ToF sensor**, on an NVIDIA Jetson AGX Orin. A monocular depth network
supplies dense *structure*; the ToF supplies absolute *scale*; a closed-form
least-squares fit joins them with **no learned parameters**. Full design in
[`RingFusion_technical_reference_updateP2.md`](RingFusion_technical_reference_updateP2.md).

## Repository layout

| Path | What |
|---|---|
| [`ros2_ws/`](ros2_ws/README.md) | ROS 2 Humble workspace — sensor drivers, perception pipeline, bringup. **Main README + full task tracker live here.** |
| [`training/`](training/README.md) | Off-robot training + export (distill backbone, train residual, ONNX → TensorRT). |
| `firmware-esp/` | ESP32-C6 firmware streaming TMF8829 ToF frames. |
| `tools/` | `calibrate_camera.py` (fisheye calibration) + ONNX/engine build scripts. |
| `CAD-files/` | Mechanical (Fusion/SLDASM). |
| `checkerboard_9x6_25mm.pdf` | Print-ready calibration target (100% scale). |

## Status

The **full perception pipeline runs live on the Orin at ~15 Hz** — real Network A backbone
(distilled → TensorRT FP16), **binary ToF firmware** (~16 Hz), and closed-form ToF→mono
anchoring. Camera is **calibrated** (fisheye → pinhole), so depth is metric. Validated
against a known target: Network-A metric depth agrees with the ToF to **~12 mm** (within the
ToF's ~20 mm accuracy) — see the demo montage [`docs/demo/pipeline_demo.png`](docs/demo/pipeline_demo.png).

**Network B (residual refiner): first real training run done — trained → exported → live**
(563 paired frames, `residual=residual_trt`), but **not yet a clean win.** It makes genuine
corrections (median 41 mm) yet has three open concerns: degenerate-pixel blow-ups (heavy tail
→ 10 km clamp), a rate drop (15 → ~7 Hz; its apply is on CPU), and coverage capped at ~0.88
(under-confident; additive-variance ceiling). Full write-up:
[training/README.md → First training run](training/README.md#first-training-run--results--concerns-2026-07-23).
Network A is pilot-quality (2,000 imgs, ρ 0.9962 vs teacher); a full ~15–20k re-distill is pending.

Live rates: [ros2_ws/README.md → Performance notes](ros2_ws/README.md#performance-notes-jetson-agx-orin).
Full breakdown + living checklist: [ros2_ws/README.md → Task tracker](ros2_ws/README.md#task-tracker).

## ▶ Do next (immediate) — Network B go/no-go

1. **Held-out-anchor accuracy eval** — does A+B predict held-out ToF zones *more accurately*
   than A-only? The objective keep-or-recollect decision (we have B's correction magnitude,
   not its sign yet).
2. **If B wins:** GPU-offload B's apply (restore ~15 Hz) + add a `da,db` stability clamp/regularizer
   (kill the 10 km tail).
3. **If not:** re-collect more/varied paired data (`paired_logger auto`) + retrain with the regularizer.

See [training/README.md → §2 Network B](training/README.md).

**✅ Done:** full pipeline live end-to-end (camera → rectify → ToF binary → Network A +
anchoring → `/cloud`); fisheye **calibrated** (cv2.fisheye, RMS 0.5406 px, real de-warp);
Network A **distilled (pilot) → TensorRT FP16/INT8 → running live**; ToF **binary firmware +
persistent assembler** (~8 → ~16 Hz); perception **GPU-offloaded** (~5.6 → ~27 Hz capable);
**Network B trained on 563 real frames → exported FP16 → integrated live** (concerns above).

## Output: point cloud → LiDAR / SLAM layer (planned)

The pipeline publishes `/cloud` (`sensor_msgs/PointCloud2`) — per-frame **metric 3D points**
(back-projected from the depth via the camera intrinsics), the same data type a LiDAR or
depth camera emits. So it feeds RViz, PCL, Nav2, and SLAM nodes directly.

**Coverage — how it compares to LiDAR.** We are **forward-facing**, not 360°: the cloud
covers the **camera's FOV cone** (denser than LiDAR — ~125k pts/frame — but narrower). True
all-around coverage needs the multi-module **ring** (future) or driving/turning to sweep.

**Depth confirmation coverage.** The ToF is a **32×32 = 1024-zone grid** over its central
~61°×45° cone (≈830 valid zones/frame in practice), so it ground-truths **many objects at
once across the center**, not a single point. The wider camera **periphery** is dense but
**mono-estimated** (scaled by the global fit, not ToF-confirmed) — which is exactly what
**Network B** refines and assigns calibrated uncertainty. Held-out ToF anchors are how we
*measure* depth accuracy at many non-center points (see training/README §2a). Over motion the
narrow ToF cone **sweeps** the scene, so an accumulated map ends up ToF-confirmed far beyond
any single frame's center.

**SLAM without wheel odometry — yes.** Like LiDAR SLAM (which also has no wheels), motion is
recovered **from the sensor data itself**: register consecutive frames — geometrically (ICP
on the clouds) and/or visually (RGB feature tracking + our metric depth = RGB-D odometry) —
then accumulate. Two advantages over the alternatives: the ToF gives **absolute scale every
frame** (no monocular-SLAM scale drift), and the RGB enables **visual loop closure** (harder
for bare LiDAR). The cost is our **narrow FOV** (less frame-to-frame overlap → more drift on
fast turns / blank walls / long corridors), where an optional **IMU or wheel-odom prior** adds
robustness but is **not required**. Fastest path to a working no-wheels map: `/depth` + RGB
into **RTAB-Map** (RGB-D SLAM). This is the last roadmap stage (post-Network B).

## Quick start (view the live sensors)

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch ringfusion_bringup feeds.launch.py   # camera + ToF heatmap; see ros2_ws/README.md
```

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

**Network B (residual refiner) is still the identity mock — training it is the current focus.**
Network A is pilot-quality (2,000 imgs, ρ 0.9962 vs teacher); a full ~15–20k re-distill is pending.

Live rates: [ros2_ws/README.md → Performance notes](ros2_ws/README.md#performance-notes-jetson-agx-orin).
Full breakdown + living checklist: [ros2_ws/README.md → Task tracker](ros2_ws/README.md#task-tracker).

## ▶ Do next (immediate) — Network B

1. **Collect paired `(rectified image, 32×32 ToF)` logs** with `paired_logger` (stationary
   stations across varied geometry).
2. **Train the residual via held-out anchors** — `train_residual.py --real` → coverage → 0.68.
3. **Export FP16, integrate** (`residual_engine:=…`) → the calibrated-uncertainty "fused depth".

See [training/README.md → §2a Network B](training/README.md).

**✅ Done:** full pipeline live end-to-end (camera → rectify → ToF binary → Network A +
anchoring → `/cloud`); fisheye **calibrated** (cv2.fisheye, RMS 0.5406 px, real de-warp);
Network A **distilled (pilot) → TensorRT FP16/INT8 → running live**; ToF **binary firmware +
persistent assembler** (~8 → ~16 Hz); perception **GPU-offloaded** (~5.6 → ~27 Hz capable).

## Quick start (view the live sensors)

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch ringfusion_bringup feeds.launch.py   # camera + ToF heatmap; see ros2_ws/README.md
```

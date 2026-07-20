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

The **sensor stack and full perception pipeline run today with mock networks**. The
two neural nets are written but **not trained**. Depth is **not metric until the
fisheye lens is calibrated**. The GPU (PyPI torch) currently has a broken cuBLAS —
training needs NVIDIA's JetPack torch first.

Full breakdown + living checklist:
**[ros2_ws/README.md → Task tracker](ros2_ws/README.md#task-tracker)**.

## ▶ Do next (immediate)

1. **Calibrate the lens.** Print `checkerboard_9x6_25mm.pdf` (100% scale, tape flat,
   measure a square), then run `tools/calibrate_camera.py` capture + calibrate and
   paste the result into
   [`ros2_ws/src/ringfusion_bringup/config/calibration.yaml`](ros2_ws/src/ringfusion_bringup/config/calibration.yaml).
   Turns the nominal de-warp into the accurate one.
2. **Fix GPU torch.** The `~/.local` PyPI `torch 2.11.0` has broken cuBLAS
   (`CUBLAS_STATUS_ALLOC_FAILED`). Replace with NVIDIA's JetPack wheel
   (L4T 36.5 / CUDA 12.6 / py3.10). Unblocks all of training (Section B).
3. **B1 — Step 0 sanity.** `python training/step0_sanity.py --image step0_raw_frame.png --raw`
   — eyeball Depth Anything V2 on a rectified frame before investing in distillation.
   (Instant once GPU is fixed; slow on CPU.)
4. **B2 — distillation.** Collect ~20k rectified images → `cache_teacher` →
   `distill_backbone` → `export_onnx` → `build_engine`, and measure Orin FPS.

## Quick start (view the live sensors)

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch ringfusion_bringup feeds.launch.py   # camera + ToF heatmap; see ros2_ws/README.md
```

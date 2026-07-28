# RingFusion

Real-time metric depth + point clouds from **one wide-angle camera and a sparse
multizone ToF sensor**, on an NVIDIA Jetson AGX Orin. A monocular depth network
supplies dense *structure*; the ToF supplies absolute *scale*; a closed-form
least-squares fit joins them with **no learned parameters**. Full design in
[`RingFusion_technical_reference_updateP2.md`](RingFusion_technical_reference_updateP2.md).

![RingFusion pipeline running live](docs/demo/gifs/pipeline_4panel.gif)

*Live on the Orin at 13.7 Hz. Left to right: rectified camera · 32×32 ToF (the only real
distance measurement) · fused metric depth · top-down obstacle map. Full 34 s clip and an
**interactive 3D flythrough** of the same drive:
[`docs/demo/network_b_v1_vs_v2.html`](docs/demo/network_b_v1_vs_v2.html) — open it in a browser,
it is fully self-contained.*

<table>
<tr>
<td width="50%"><img src="docs/demo/gifs/tof_heatmap.gif" alt="32x32 ToF heatmap"><br>
<sub><b>What the ToF actually sees</b> — 1024 zones, black where no return. This sparse grid
is the entire source of metric scale.</sub></td>
<td width="50%"><img src="docs/demo/gifs/topdown.gif" alt="Top-down obstacle map"><br>
<sub><b>Bird's-eye map</b> derived from the fused cloud — robot at bottom-centre, rings at
1/2/3 m, ground faint, obstacles coloured by distance.</sub></td>
</tr>
</table>

## Repository layout

| Path | What |
|---|---|
| [`ros2_ws/`](ros2_ws/README.md) | ROS 2 Humble workspace — sensor drivers, perception pipeline, bringup. **Main README + full task tracker live here.** |
| [`training/`](training/README.md) | Off-robot training + export (distill backbone, train residual, ONNX → TensorRT). |
| `firmware-esp/` | ESP32-C6 firmware streaming TMF8829 ToF frames. |
| `tools/` | `calibrate_camera.py` (fisheye calibration) + ONNX/engine build scripts. |
| `CAD-files/` | Mechanical (Fusion/SLDASM). |
| `checkerboard_9x6_25mm.pdf` | Print-ready calibration target (100% scale). |

## How it works

```
Arducam fisheye ──▶ rectify (fisheye → pinhole) ──▶ Network A  ──┐
                                                   (relative     │
                                                    disparity)   │
                                                                 ▼
ESP32-C6 ──▶ TMF8829 32×32 ToF ──▶ project zones to pixels ──▶ closed-form
             (binary, ~16 Hz)      (mirror + FOV corrected)    affine fit  ──▶ metric depth D₀
                                                                 │              (no learned
                                                                 │               parameters)
                                                                 ▼
                                            Network B (residual) ──▶ /depth  /depth_var  /cloud
```

**Network A** predicts *relative* disparity — good structure, no scale. **The ToF** supplies
scale but only 1024 sparse zones covering **7.5 % of the frame**. A robust 2-parameter fit
maps disparity → inverse depth using those zones as anchors; that step is pure geometry and
has no learned parameters, so the system produces valid metric depth even with Network B
switched off. **Network B** then applies a learned per-pixel correction plus a variance
estimate, supervised on held-out ToF zones.

## Status — 2026-07-28

**Running live end-to-end on the Orin with both real networks at 13.7 Hz.**

| | value |
|---|---|
| `/depth`, `/depth_var`, `/cloud` | **13.7 Hz** (ToF-limited) |
| Depth accuracy vs real ToF, driving | **0.199 m** mean absolute error |
| Uncertainty quality, `corr(σ, \|error\|)` | **0.943** |
| Backbone agreement with truth, ρ | **0.917** |
| Far-field blow-ups | **0** frames over 105 driving |

### A calibration bug dominated everything before this date

The ToF grid was **horizontally mirrored** relative to the camera *and* `fov_h`/`fov_v` were
**swapped**. Every anchor landed on the wrong pixel, corrupting the affine fit and every
metric downstream of it.

<table>
<tr>
<td width="50%"><img src="docs/demo/gifs/anchors_deployed_broken.jpg" alt="ToF anchors, broken calibration"><br>
<sub><b>Before</b> — ToF anchors coloured by measured distance, scrambled across the scene.
ρ 0.737.</sub></td>
<td width="50%"><img src="docs/demo/gifs/anchors_fixed.jpg" alt="ToF anchors, fixed calibration"><br>
<sub><b>After</b> — far/red on the back wall, near/blue on the floor, clean gradient.
ρ 0.917, depth MAE −21 %.</sub></td>
</tr>
</table>

It also overturned an earlier conclusion: the Depth Anything V2 teacher had scored no better
than the distilled student (ρ 0.750 vs 0.737), which read as "monocular depth is intrinsically
hard here". Both were simply being scored against misprojected anchors. **The backbone was
never the bottleneck.**

### Network B is now a clean win

`residual_v3`, retrained on the corrected geometry, is the first version that beats **doing
nothing** — v1 and v2 never did:

| | closed-form only | v1 | v2 | **v3** |
|---|---|---|---|---|
| depth MAE, driving | 0.294 m | — | 0.247 m | **0.199 m** |
| `corr(σ, \|error\|)` | — | 0.490 | 0.913 | **0.943** |
| frames hitting the 20 m clamp | 0 | 47/102 *(pre-fix)* | 0 | **0** |
| added frame-to-frame jitter | baseline | ~2× | +4 % | **+7 %** |

The uncertainty channel used to be saturated and *most confident where it was most wrong*; it
now tracks real error at ρ 0.943, staying tight at the anchors and widening where the network
extrapolates.

### Known limits

- **ToF covers 7.5 % of the frame.** The other 92.5 % is monocular extrapolation carrying the
  affine fit. A geometry limit, not a bug, but it bounds what can be trusted.
- **v3 still under-corrects** — mean |B−A| is only 0.019 m, so the structure-loss weight
  (0.15) likely has room to come down further.
- **Network A is pilot-quality** — 2,000 images; a full ~15–20 k re-distill is pending.
- **Checkpoint selection is noisy** — the validation split is ~62 samples against a
  heavy-tailed NLL loss.

Full detail, per-run numbers and the task tracker:
[ros2_ws/README.md](ros2_ws/README.md). Capture and diagnostic tooling:
[tools/diagnostics/](tools/diagnostics/README.md). All captures:
[docs/demo/](docs/demo/README.md).

## ▶ Do next

1. **Sweep `--struct-weight` below 0.15** — v3 improved when it dropped from 0.3, and it is
   still barely correcting, so the optimum is probably lower again.
2. **Control run at 0.3 on corrected geometry** — v3 changed the weight *and* the geometry
   together, so its win is not yet attributable to either.
3. **Widen the validation split** before checkpoint selection bites.
4. **Re-distill Network A** on the full set now that we know it was never the limiting factor.

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

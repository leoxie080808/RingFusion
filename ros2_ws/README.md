# RingFusion ROS 2 Workspace

ROS 2 Humble workspace for the RingFusion project (ToF hub + camera driver, perception, bringup, and message definitions).

## Project status (handoff)

Snapshot for anyone picking this up. The **sensor stack and the full perception
pipeline are code-complete and run today with mock networks.** The two neural
networks are written but **not yet trained**, and nothing has been **built or run on
the Jetson** yet. Depth is not metrically trustworthy until the fisheye lens is
calibrated. Design details live in `RingFusion_technical_reference_updateP2.md`;
the training/export workflow is in [../training/README.md](../training/README.md).

### Done

- **Sensors, MCU → ROS.** ESP32-C6 firmware streams TMF8829 ToF frames; `tof_driver`
  (serial → `ToFFrame`) and `camera` (Arducam IMX219 CSI) nodes publish live, plus
  `tof_heatmap` + `dual_view` for inspection.
- **Perception pipeline, structurally complete** (stages 2–8): backbone → zone
  projection → closed-form anchoring → analytic per-pixel variance → residual →
  unprojection. Publishes `/cloud`, `/depth`, `/depth_var`. The pure-numpy core
  ([src/ringfusion_perception/ringfusion_perception/pipeline.py](src/ringfusion_perception/ringfusion_perception/pipeline.py))
  has a PC test suite ([src/ringfusion_perception/test/](src/ringfusion_perception/test/), 5/5 passing — no ROS/CUDA needed). See [Perception](#perception).
- **Both networks are pluggable and default to mocks**, so the pipeline runs now and
  swaps to real engines with a launch arg (`backbone_engine:=…`, `residual_engine:=…`) —
  no code change. `MockResidual` is the exact identity, so output = the closed-form fit.
- **Training + export code written** ([../training/](../training/), [../tools/](../tools/)): Network A
  (student backbone), Network B (residual, measured ~0.46M params), distillation + NLL
  losses, datasets, teacher caching, ONNX + TensorRT build scripts. Torch-only parts
  smoke-tested; the residual reuses the deployed anchoring math so training matches inference.

### Not done / blocked

- **Fisheye calibration** — [src/ringfusion_bringup/config/calibration.yaml](src/ringfusion_bringup/config/calibration.yaml) still holds
  NOMINAL placeholder intrinsics. **Depth is not metrically trustworthy until this is
  done** (checkerboard, Kannala-Brandt). Blocking prerequisite.
- **Networks not trained** — no `student_best.pth` / `residual_best.pth` yet; the pipeline
  is running mocks.
- **No TensorRT engines** — must be built on the Orin (hardware-/version-specific). Never
  built or run on the Jetson yet.
- **Residual ground truth** — synthetic/LiDAR/OAK-D depth not collected (the schedule risk).
- **All paper numbers are placeholders** — FPS, accuracy, and param counts (student measured
  **3.66M**, residual measured 0.46M; the doc's "6.1M" student was a placeholder). Nothing is
  submittable until measured on real hardware.
- The dev PC (Windows) **cannot build the ROS workspace** (`rclpy` is Linux/ROS) — build and
  train on the Jetson or a Linux GPU box. The pure-numpy pipeline + training code do run on the PC.

### Next steps (in dependency order)

| # | Step | Needs | Note |
|---|---|---|---|
| 0 | Run Depth Anything V2 on real **rectified** Arducam frames (sanity) | camera | cheap; could change the plan |
| 1 | Fisheye calibration → real intrinsics in `calibration.yaml` | camera | **blocks trustworthy depth** |
| 2 | Collect ~20k rectified images | camera | diversity > volume |
| 3 | `cache_teacher.py` — cache DA V2 disparity targets | GPU | once; expensive |
| 4 | `distill_backbone.py` → student, validate vs teacher | 3 | retires MockBackbone |
| 5 | `export_onnx` → `build_engine` INT8, **measure Orin FPS** | 4, Jetson | headline number |
| 6 | Run DEPTHOR-Small on the same Orin | Jetson | efficiency claim |
| 7 | Ground-truth collection (synthetic first) | — | unblocks residual |
| 8 | `train_residual.py` with NLL, measure coverage (→0.68) | 7 | calibration claim |
| 9 | Export residual FP16, integrate, re-measure end-to-end | 8 | final numbers |

**Steps 0–6 need no ground truth and deliver the two headline results** (Orin throughput and
the DEPTHOR comparison) — do them first. Because the residual is zero-initialized to the
identity, the system ships and produces the closed-form result before Network B exists.

## Packages

| Package | Type | Purpose |
|---|---|---|
| `ringfusion_msgs` | ament_cmake | Custom message definitions (`ToFFrame.msg`) |
| `ringfusion_drivers` | ament_python | ToF hub (`tof_driver`), Arducam camera (`camera`), ToF heatmap colorizer (`tof_heatmap`), local combined viewer (`dual_view`) |
| `ringfusion_perception` | ament_python | Mono depth + ToF anchoring perception node (see [Perception](#perception)) |
| `ringfusion_bringup` | ament_cmake | Launch files and extrinsic calibration config |

> `ringfusion_msgs` must declare `<export><build_type>ament_cmake</build_type></export>` in its `package.xml`, or colcon misidentifies it as a plain `catkin` package and skips it during the build.

## First-time setup

```bash
source /opt/ros/humble/setup.bash
cd ~/RingFusion/ros2_ws
rosdep install --from-paths src --ignore-src -r -y   # install missing dependencies
```

## Building

```bash
# Build everything
colcon build --symlink-install

# Build a single package (and nothing else)
colcon build --symlink-install --packages-select ringfusion_msgs

# Build a package and everything that depends on it
colcon build --symlink-install --packages-up-to ringfusion_bringup

# Clean build (wipe generated artifacts, not src/)
rm -rf build/ install/ log/
colcon build --symlink-install
```

After building, source the overlay in every new terminal:

```bash
source install/setup.bash
```

## Running

```bash
# Just view both feeds (camera + ToF heatmap) — see "Viewing both feeds" below
ros2 launch ringfusion_bringup feeds.launch.py

# Full fusion module: ToF hub + camera + perception (-> /cloud, /depth)
ros2 launch ringfusion_bringup single_module.launch.py port:=/dev/ttyACM1

# Same, but feed a still image instead of the live CSI camera
ros2 launch ringfusion_bringup single_module.launch.py image:=/path/to/shot.jpg

# Run a single node directly
ros2 run ringfusion_drivers tof_driver
ros2 run ringfusion_drivers camera
ros2 run ringfusion_drivers tof_heatmap
ros2 run ringfusion_perception perception
```

View the output point cloud in `rviz2`: add a `PointCloud2` display on `/cloud` with fixed frame `cam_0`.

## Perception

`perception_node` caches the latest camera frame + ToF frame and runs the pure-numpy
pipeline (`pipeline.run`) whenever a ToF frame arrives (~5 Hz). It publishes:

| Topic | Type | Contents |
|---|---|---|
| `/cloud` | `sensor_msgs/PointCloud2` | metric point cloud (the goal) |
| `/depth` | `sensor_msgs/Image` `32FC1` | metric depth map |
| `/depth_var` | `sensor_msgs/Image` `32FC1` | per-pixel depth variance (calibrated uncertainty) |

The pipeline runs two neural networks, both **pluggable** so the workspace runs today
with mocks and swaps to real engines on the Jetson with no other changes:

- **Backbone** (Network A, `backbone.py`) — monocular relative disparity. `MockBackbone`
  by default; pass `backbone_engine:=student_int8.engine` to use `TensorRTBackbone`.
- **Residual** (Network B, `residual.py`) — per-pixel correction to the affine fit plus
  extra variance. `MockResidual` (the exact identity → output equals the closed-form
  fit) by default; pass `residual_engine:=residual_fp16.engine` to use `ResidualRefiner`.

The math between them (zone projection, closed-form anchoring, analytic covariance,
unprojection) has no learned parameters. See `RingFusion_technical_reference_updateP2.md`.

**Stage 1 rectification.** The lens is a ~155° fisheye, so `perception_node` remaps each
frame to a rectilinear (pinhole) image before the pipeline runs (`rectify.FisheyeRectifier`),
and everything downstream — zone projection *and* cloud unprojection — then uses one
consistent pinhole `K`. A zero-distortion fisheye is still an *equidistant* fisheye, and the
nominal focal length is close to its true value, so rectification is **active even with the
nominal calibration** (a rough but real de-warp); real calibration just refines the
lens-specific coefficients. It falls back to an identity passthrough only for a `pinhole`
model or if cv2 is missing.

**See it live:** `ros2 run ringfusion_perception rectify_view` publishes `/rectify_compare`
(raw | rectified, side by side). View it at
`http://<jetson-ip>:8080/stream?topic=/rectify_compare` (with `web_video_server` running).
Point at straight edges — door frames, floor tiles — to confirm the de-warp; tune
`rectify:` `fov_scale`/`balance` in `calibration.yaml` and relaunch to adjust the crop.

**Calibrating the lens** (fills in the real intrinsics that activate rectification):

```bash
# 1. Collect ~20 checkerboard views (headless auto-capture; move the board around,
#    especially into the corners where fisheye distortion is strongest)
PYTHONNOUSERSITE=1 python tools/calibrate_camera.py --capture calib_imgs --cols 9 --rows 6
# 2. Calibrate and print the yaml block to paste into calibration.yaml
python tools/calibrate_camera.py --images calib_imgs --cols 9 --rows 6 --square-mm 25
```

Paste the printed `camera:` + `rectify:` block into
[src/ringfusion_bringup/config/calibration.yaml](src/ringfusion_bringup/config/calibration.yaml).
Intrinsics are resolution-specific — calibrate at the resolution you deploy.

Parameters (`perception_node`): `calib`, `frame_id`, `backbone_engine`, `residual_engine`,
`min_confidence` (default `-1` = ignore ToF confidence and weight all zones equally; set
`>= 0` to reject weak zones and weight the fit by confidence).

**Testing on a dev PC (no ROS/CUDA/cv2 needed).** `pipeline.py`, `geometry.py`,
`anchoring.py`, `residual.MockResidual`, and `backbone.MockBackbone` are pure numpy, so
the whole pipeline runs and is unit-tested off-robot:

```bash
cd src/ringfusion_perception
python -m pytest test/ -v          # or: python test/test_pipeline.py
```

## Camera hardware (Arducam IMX219 on the B0472 CSI adapter)

The `camera` node (`ringfusion_drivers/camera.py`) drives the Arducam IMX219 wide-angle
module through `nvarguscamerasrc`, the native Jetson ISP path — this requires the Arducam
kernel driver to be installed first (it does not come with JetPack by default).

**One-time driver install** (Jetson AGX Orin, JetPack 6 / L4T 36.5):

```bash
cd ~
wget https://github.com/ArduCAM/MIPI_Camera/releases/download/v0.0.3/install_full.sh
chmod +x install_full.sh
./install_full.sh -m imx219      # downloads the .deb matching your exact kernel build
sudo reboot
```

Verify after reboot:

```bash
dpkg -l | grep arducam                 # arducam-nvidia-l4t-kernel should be listed
ls /dev/video0                         # should exist
v4l2-ctl --list-devices                # should show "imx219 ..." on /dev/video0
media-ctl -p                           # should show an imx219 sensor entity, not an empty topology
```

**Raw pipeline smoke test** (bypasses ROS entirely — good first check):

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=30 ! \
  "video/x-raw(memory:NVMM),width=1280,height=720,framerate=21/1" ! \
  nvvidconv ! fakesink
```

If this reports `Argus Correctable Error Status` / `CANCELLED` at the very end after
`Got EOS`, that's normal teardown noise for a fixed-buffer-count pipeline — not a failure.

**Rotation and color:** the module is mounted upside down on the ring, so `camera_node`
rotates 180° by default (`flip` param, `nvvidconv flip-method=2`; pass `flip:=0` to disable).
The IMX219 has no ISP color-tuning profile installed, so raw frames come out with a strong
color cast and crushed contrast — `camera.py`'s `ArducamCSI.read()` applies a gray-world
white-balance + contrast-stretch correction (via a LUT, ~8ms/frame) before publishing.

## Viewing both feeds (camera + ToF heatmap)

One command brings up the whole viewing stack — camera, ToF driver, heatmap colorizer,
the local on-monitor window, and the browser server:

```bash
sudo apt-get install -y ros-humble-web-video-server   # one-time

ros2 launch ringfusion_bringup feeds.launch.py
```

Arguments: `port` (ToF serial, default `/dev/ttyACM1`), `fps` (camera capture, default 30),
`view` (local window, default true), `web` (browser server, default true).

**Local viewing — lowest latency, recommended.** Run the launch from a terminal *inside the
Jetson's desktop session* (monitor plugged in) and the `dual_view` window opens automatically,
showing camera + ToF side by side with live Hz. It subscribes to the ROS topics directly —
no network hop, no JPEG re-encode, which are the two things that add lag. Over SSH with no
display, pass `view:=false`.

**Browser viewing — from another machine on the network:**

```
http://<jetson-ip>:8080/stream?topic=/image&quality=60
http://<jetson-ip>:8080/stream?topic=/tof_heatmap
```

(find `<jetson-ip>` with `hostname -I`). **On the `quality` param:** at the default quality
(95), a 1280x720 JPEG stream needs ~20Mbps sustained. If the viewing device's WiFi can't hold
that, the server's write queue backs up into a growing multi-second lag even though capture
(~8ms) and ROS transport (~11ms) stay fast. `quality=60` cuts it to ~4Mbps with no visible
quality loss; drop to `30`-`40` if lag persists. Local viewing avoids this entirely.

### Performance notes (Jetson AGX Orin)

- **Power mode.** Check with `nvpmodel -q`. If it's not MAXN, everything is throttled (fewer
  cores, lower clocks) — set it with `sudo nvpmodel -m 0` (applies immediately, no reboot;
  persists across reboots) then `sudo jetson_clocks` (re-run after each boot). The higher-res
  camera modes (1080p30) also appear to need MAXN's CSI/ISP bandwidth.
- **Camera rate.** The Arducam IMX219 tuning here only accepts **1280x720** via
  `nvarguscamerasrc`; higher-res modes report `INVALID_SETTINGS`. 720p runs up to ~30fps.
- **ToF rate.** Capped at 5Hz by the firmware (`MEASUREMENT_PERIOD_MS = 200` in
  `firmware-esp/main/main.c`). Lower it (e.g. `33` → 30Hz target; will self-limit to the
  sensor's real 48x32 max) and reflash the ESP32. The ROS side forwards whatever it emits.
- **`nvargus-daemon`.** If the camera starts failing with `INVALID_SETTINGS` intermittently,
  the Argus daemon is wedged (often from `kill -9`-ing a camera process). Fix:
  `sudo systemctl restart nvargus-daemon`. Stop camera nodes with SIGTERM, not `kill -9`.

## Sanity checks / useful `ros2` commands

```bash
# Confirm colcon sees all 4 packages with the right build type
colcon list

# Confirm packages are registered on the ROS 2 graph (after sourcing install/setup.bash)
ros2 pkg list | grep ringfusion
ros2 pkg prefix ringfusion_msgs        # should point into install/

# Confirm the custom message built correctly
ros2 interface show ringfusion_msgs/msg/ToFFrame

# While nodes are running:
ros2 node list                         # expect tof_driver, camera, perception, cam_to_tof
ros2 topic list                        # expect /cloud, plus driver/camera topics
ros2 topic hz /cloud                   # confirm perception is publishing
ros2 topic echo /cloud --once          # sanity-check one message
ros2 node info /perception             # see its subscriptions/publications/params
ros2 param list /tof_driver
ros2 run tf2_ros tf2_echo cam_0 tof_0  # confirm the static transform is being published

# Debugging
ros2 doctor                            # general environment/network health check
rqt_graph                              # visualize the node/topic graph
```

## Troubleshooting

- **A package is missing from `colcon list` / didn't build**: check its `package.xml` has the correct `<export><build_type>...</build_type></export>` tag (`ament_cmake` for CMake packages, `ament_python` for Python packages).
- **"failed to create symbolic link ... Is a directory"**: a stale `build/<pkg>` directory from an earlier failed build is conflicting with `--symlink-install`. Remove just that package's build folder and rebuild:
  ```bash
  rm -rf build/<pkg> install/<pkg>
  colcon build --symlink-install --packages-select <pkg>
  ```
- **Nodes can't find messages/executables after building**: make sure you `source install/setup.bash` in the terminal you're running from (each new terminal needs it).
- **`camera` node crashes with "Could not open CSI camera"**: check `dpkg -l | grep arducam` and `/dev/video0` exist first (see Camera hardware section above — the Arducam driver may not be installed). If those are fine, check `python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer` — if it says `NO`, a pip-installed `opencv-python` in `~/.local` is shadowing JetPack's GStreamer-enabled system `python3-opencv`. `single_module.launch.py` already sets `PYTHONNOUSERSITE=1` to work around this; if you run `ros2 run ringfusion_drivers camera` directly outside the launch file, prefix it with `PYTHONNOUSERSITE=1` too.

## Task tracker

Living checklist of remaining work, in dependency order. Scratch items off (`[x]`)
as they land. Detail on each in the technical reference and the sections above.

### ▶ Do next (immediate action items)

1. **Calibrate the lens (A2) — physical, you.** Print `checkerboard_9x6_25mm.pdf` at
   **100% scale**, tape it flat to something rigid, measure one square with a ruler.
   Then ping me to drive the capture + calibrate; I paste the result into
   `calibration.yaml`. This turns the nominal de-warp into the accurate one.
2. **Fix GPU torch — unblocks all of Section B.** The `~/.local` PyPI `torch 2.11.0`
   has broken cuBLAS (`CUBLAS_STATUS_ALLOC_FAILED`). Replace with NVIDIA's JetPack
   wheel (L4T 36.5 / CUDA 12.6 / py3.10). Required before B2; makes B1 instant. Say go
   and I'll look up the exact wheel and do the swap.
3. **B1 Step-0 sanity — eyeball the teacher.** `python training/step0_sanity.py
   --image step0_raw_frame.png --raw` (add `--long-side 640` on CPU; runs in seconds
   once the GPU is fixed). Looking for: near surfaces warm, far cool, crisp edges, no
   big smeared/flat blobs. A bad result changes the plan *before* B2.
4. **B2 distillation — after 1–3.** Collect ~20k rectified images (the camera +
   `rectify_view` can produce them), then `cache_teacher` → `distill_backbone` →
   `export_onnx` → `build_engine`, and measure Orin FPS.

**A — Camera / Arducam pipeline**
- [x] A1. `tools/calibrate_camera.py` — fisheye (Kannala-Brandt) checkerboard calibration tool
- [ ] A2. Run calibration on the real lens → replace nominal intrinsics in `calibration.yaml` *(needs a printed checkerboard; physical step)*
- [x] A3. Stage 1 rectification (fisheye → rectilinear) wired into perception (`rectify.py`); **active now** with the nominal equidistant model (a real de-warp), refined once A2 lands. Confirm live: `rectify_view` → `/rectify_compare`
- [x] A4. Capture resolution = **1640×1232** (IMX219 full-sensor 2×2-binned — full fisheye FOV; the 16:9 modes crop it). Wired into `calibration.yaml`, both launch files, `camera_node`, and the calib tool. Runs ~15 Hz capping a full core (color-correct at 2 MP; fine for the ~5 Hz fusion, see C3)

**B — Networks (need no ground truth for B1–B2)**
- [ ] B1. Step 0 sanity: run Depth Anything V2 on real **rectified** frames (cheap; could change the plan). *Script ready + validated: `training/step0_sanity.py` (rectifies inline, reuses `cache_teacher`'s teacher). `transformers` + `pillow>=10` installed; test frame `step0_raw_frame.png`. Runs but is slow on CPU (minutes even at `--long-side 640`) — GPU fix (above) makes it instant. Output: `[ RGB | inverse-depth ]` side-by-side.*
- [ ] B2. Distill backbone → ONNX → INT8 engine **on the Orin** → measure FPS (headline number). Backbone input size (default **384×288**, must be a **multiple of 32**) is the depth-detail lever, tunable here with a latency measurement — *not* the camera resolution. Built student is **3.66M params** (design doc's "6.1M" was a placeholder). **PREREQUISITE: fix GPU torch** — the `~/.local` PyPI `torch 2.11.0` has a broken cuBLAS (`CUBLAS_STATUS_ALLOC_FAILED`); replace with NVIDIA's JetPack wheel (L4T 36.5 / CUDA 12.6 / py3.10). CPU works for one-off B1 only
- [ ] B3. Ground-truth collection → train residual with NLL → measure calibration coverage (→0.68)

**C — Housekeeping**
- [ ] C1. `tof_driver` 500 Hz poll cleanup (threaded blocking read; verify ToF latency doesn't regress)
- [ ] C2. Fix stale "B0472 stitches 4 cameras into one frame" comment in `camera.py` (it's native per-camera)
- [ ] C3. Camera color-correction is CPU-bound (~one core/camera at 1640×1232) — won't scale to 4 cameras; move the LUT tone-correction to the GPU (nvvidconv/CUDA) or make it optional

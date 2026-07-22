# RingFusion ROS 2 Workspace

ROS 2 Humble workspace for the RingFusion project (ToF hub + camera driver, perception, bringup, and message definitions).

## Project status (handoff)

Snapshot for anyone picking this up. The **sensor stack and the full perception
pipeline are code-complete and run today with mock networks.** The **fisheye lens is
calibrated** (RMS 0.5406 px) and the **backbone (Network A) has been distilled on a
2000-image pilot** (val_ssi 3.51, **ρ 0.9962** vs the Depth Anything V2 teacher —
validated with `compare_student.py` / `eval_student.py`). The student is **exported to
ONNX, the FP16 engine is built, and the INT8 engine is building** on the Orin; the
TensorRT runtime (`trt_util.TRTRunner`) and the INT8 calibrator were **ported off pycuda
to torch** (pycuda isn't on the Jetson). **Next: an isolated `TRTRunner` smoke test, then
wire the engine into perception to measure FPS** (retiring `MockBackbone`). The residual
(Network B) is written but not yet trained. Design details live in
`RingFusion_technical_reference_updateP2.md`.

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

- **Full training set** — only a **2000-image pilot** has been collected so far. The
  full ~15–20k (deployment environment, via `collect_frames`) still needs gathering for
  the final-quality backbone; the pilot is enough to prove the pipeline, not to ship.
- **Live real backbone** — engines are building on the Orin (FP16 done, INT8 in progress),
  but none has been *run* live yet: `TRTRunner` smoke test + `backbone_engine:=…` launch
  still to do. Perception runs `MockBackbone` until an engine is passed in.
- **Residual (Network B) not trained** — no `residual_best.pth`; needs measured/synthetic
  ground-truth depth (the schedule risk). `MockResidual` (identity) until then.
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
| 0 | ~~Run Depth Anything V2 on real rectified Arducam frames (sanity)~~ | — | **DONE** (B1 passed → `step0.png`) |
| 1 | ~~Fisheye calibration → real intrinsics in `calibration.yaml`~~ | — | **DONE** (RMS 0.5406 px, `identity=False` verified) |
| 2 | Collect rectified images (`collect_frames`) | camera | **pilot done (2000)**; full ~15–20k still to gather |
| 3 | ~~`cache_teacher.py` — cache DA V2 disparity targets~~ | GPU | **DONE** (2000 cached) |
| 4 | ~~`distill_backbone.py` → student, validate vs teacher~~ | 3 | **DONE** (pilot: val_ssi 3.51, ρ 0.9962) |
| 5 | `export_onnx` → `build_engine` INT8, **measure Orin FPS** ← **you are here** | 4, Jetson | headline number |
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
ros2 run ringfusion_perception collect_frames   # collect rectified training images (see below)
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

**Running the real TensorRT backbone (retires the mock).** Once an engine is built on
the Orin (see [../training/README.md](../training/README.md) §3), pass it in and the
node loads `TensorRTBackbone` instead of `MockBackbone` — no code change:

```bash
ros2 launch ringfusion_bringup single_module.launch.py \
    backbone_engine:=$HOME/RingFusion/student_int8.engine port:=/dev/ttyACM1
ros2 topic hz /depth      # headline FPS; compare student_int8.engine vs student_fp16.engine
```

The runtime (`trt_util.TRTRunner`) uses **torch** for device memory + the CUDA stream,
**not pycuda** (pycuda isn't installed on the Jetson and is painful to build). So the
only runtime deps are `tensorrt` (JetPack) + `torch` (both present). Before the full
launch you can smoke-test the runtime in isolation: instantiate `TRTRunner` with an
engine and run one inference — a clear error beats a buried ROS failure.

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

## Collecting training images (backbone distillation — B2)

The backbone (Network A) is trained by **distillation**: Depth Anything V2 (the
"teacher") auto-generates the depth targets, so collection needs **images only — no
measured depth**. The `collect_frames` node banks **rectified** frames off `/image`
through the exact `color-correct → rectify` path the robot runs at inference (the
camera node has already white-balanced/contrast-corrected the frame; this node then
rectifies it with the same `FisheyeRectifier` + `calibration.yaml` the pipeline uses),
straight into a folder that `training/cache_teacher.py` reads as-is.

**Why the DEPLOYMENT ENVIRONMENT, not a generic/online dataset.** Distillation makes
the student copy the teacher's depth *on whatever image distribution you show it*, so
the student ends up good at scenes that look like its training set. Your deployed
frames have a specific look — this rectified 155° fisheye crop, the IMX219's
color/noise, your scenes and lighting — that internet photos (different camera, lens,
projection, content) do not share, and you **cannot rectify a normal photo to mimic
your lens**. So the bulk of the data must be your own rectified frames. External
images can be a **minority supplement** for diversity if your environment is very
homogeneous, but they never substitute for in-domain data — and since your own frames
are free to label (the teacher does it), there's little reason to. To stretch a
smaller set, prefer **augmentation of your own frames** (the distill trainer already
augments) over importing foreign images. Rule of thumb: **diversity > volume** —
~15–20k varied in-domain frames is the target, but you can pilot with ~5k and re-run
(the cached teacher makes re-distilling cheap).

**Lock `fov_scale`/`balance` before collecting.** The whole dataset *and* the deployed
pipeline must use the same `rectify:` settings in `calibration.yaml`, or the training
frames won't match what the robot sees. (Eyeball the crop first with `rectify_view`.)

### Run it (two terminals, both sourced)

```bash
# one-time: build so ros2 run sees the node
cd ~/RingFusion/ros2_ws
colcon build --symlink-install --packages-select ringfusion_perception
source install/setup.bash
```

```bash
# Terminal 1 — camera (publishes the color-corrected /image)
cd ~/RingFusion/ros2_ws && source install/setup.bash
PYTHONNOUSERSITE=1 ros2 run ringfusion_drivers camera
```

```bash
# Terminal 2 — collector (live preview window on the Jetson's monitor)
cd ~/RingFusion/ros2_ws && source install/setup.bash
ros2 run ringfusion_perception collect_frames --ros-args \
  -p calib:=$HOME/RingFusion/ros2_ws/src/ringfusion_bringup/config/calibration.yaml \
  -p out_dir:=$HOME/RingFusion/data/rect \
  -p target:=20000
```

Startup log should read `identity=False` (real de-warp active) and report how many
frames are already on disk. If `imshow` errors, prefix Terminal 2 with
`PYTHONNOUSERSITE=1` too (forces JetPack's GTK-enabled OpenCV). Needs the Jetson's
own monitor — not over SSH.

### Controls (shown in the preview window)

| Key | Action |
|---|---|
| **SPACE / y** | save the current rectified frame (manual) |
| **c** | toggle **continuous** auto-capture on/off |
| **q / ESC** | quit |

The overlay shows `[count/target]`, the mode, and a live `NEW`/`similar` + `sharp NNN`/`BLURRY NNN` status.

### Two automatic quality gates (continuous mode)

- **Dedup** — a frame counts as new only if it differs enough from the last *saved*
  frame (mean abs gray diff > `dedup_thresh`, default 8). Stops you banking 500 copies
  of the same wall as you stand still.
- **Sharpness / blur** — continuous mode saves only frames with variance-of-Laplacian
  ≥ `blur_thresh` (default 60), so **motion-blurred frames from walking are rejected**.
  Watch the live `sharp NNN` readout: stand still to see the sharp value, wave the
  camera to see it drop, set `blur_thresh` between the two (e.g. `-p blur_thresh:=100`).
  **Manual `y`-save always honours your keypress** (shows the BLURRY tag as a warning).

### Parameters

`out_dir` (default `data/rect`), `calib`, `target` (default 20000), `dedup_thresh`
(default 8.0), `min_interval` (seconds between auto-saves, default 0.3), `blur_thresh`
(default 60.0).

### Field-session tips

- **Cover the variety axes:** different areas/rooms, distances (close *and* far),
  angles/heights, lighting conditions. Diversity is what matters, not raw count.
- **Beat blur physically:** walk slowly or **pause-step** (a half-second stop lets a
  clean frame land, and continuous mode grabs it). Motion blur comes from long
  exposure, which comes from dim scenes — **brighter areas → sharper frames**, so move
  slowest where it's dark.
- **Manual for deliberate shots, continuous for bulk.** Hold `c` on while walking a
  route; tap it off and use `y` for specific poses you care about.
- **Stop and resume anytime:** quit with `q`; re-running **appends** (it continues
  numbering after the frames already in `out_dir`), so collect across several sessions.

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

1. **B2 — get the pilot engine live on the Orin + measure FPS (CURRENT).** Pilot done:
   2000 images → distilled (val_ssi **3.51**, **ρ 0.9962**). **Exported to ONNX; FP16
   engine built; INT8 building.** Remaining: (a) isolated `TRTRunner` smoke test against
   an engine, (b) `ros2 launch … single_module.launch.py backbone_engine:=…student_int8.engine`
   → `ros2 topic hz /depth` for the FPS number (retires `MockBackbone`). Export/engine
   gotchas (onnxscript, KeyError, pycuda→torch) are all fixed — see
   [../training/README.md](../training/README.md) §3.
2. **Then collect the full ~15–20k images and re-distill** for the deployable-quality
   student — the 2000-image pilot is intentionally small. Re-run `compare_student` /
   `eval_student` to quantify the gain (target ρ → ~0.998, d1.25 → >0.9).

**✅ Done recently:** GPU torch fixed — cuBLAS + cuDNN verified on the Orin (recipe in
[training/README.md](../training/README.md#gpu-torch-on-the-orin--working-recipe-resolved));
**B1 Step-0 passed** — Depth Anything V2 produces clean depth on our rectified fisheye
(`step0.png`), validating the distillation plan.

**A — Camera / Arducam pipeline**
- [x] A1. `tools/calibrate_camera.py` — fisheye (Kannala-Brandt) checkerboard calibration tool
- [x] A2. **DONE (2026-07-21).** Calibrated the real lens (cv2.fisheye, RMS **0.5406 px**); real intrinsics in `calibration.yaml`. Verified live: `rectify_view` logs `identity=False` and straight edges de-warp correctly.
- [x] A3. Stage 1 rectification (fisheye → rectilinear) wired into perception (`rectify.py`); **active now** with the nominal equidistant model (a real de-warp), refined once A2 lands. Confirm live: `rectify_view` → `/rectify_compare`
- [x] A4. Capture resolution = **1640×1232** (IMX219 full-sensor 2×2-binned — full fisheye FOV; the 16:9 modes crop it). Wired into `calibration.yaml`, both launch files, `camera_node`, and the calib tool. Runs ~15 Hz capping a full core (color-correct at 2 MP; fine for the ~5 Hz fusion, see C3)

**B — Networks (need no ground truth for B1–B2)**
- [x] B1. Step 0 sanity — **PASSED**. Depth Anything V2 on our rectified fisheye produces clean depth (near/far correct, crisp edges, objects separated) → distillation plan validated. `training/step0_sanity.py --image step0_raw_frame.png --raw` runs in ~14 s on GPU; result in `step0.png`.
- [ ] B2. Distill backbone → ONNX → INT8 engine **on the Orin** → measure FPS (headline number). Backbone input size (default **384×288**, must be a **multiple of 32**) is the depth-detail lever, tunable here with a latency measurement — *not* the camera resolution. Built student is **3.66M params** (design doc's "6.1M" was a placeholder). **GPU torch fixed** (cuBLAS/cuDNN verified — recipe in training/README). **Pilot done (2000 imgs):** collected (`collect_frames`) → cached → distilled → **val_ssi 3.51, ρ 0.9962** vs teacher (validated via `compare_student.py` / `eval_student.py`). **Now exporting to TensorRT on the Orin** (`export_onnx` legacy exporter → `build_engine` FP16→INT8 → `backbone_engine:=…` → `ros2 topic hz /depth`); then re-collect the full ~15–20k and re-distill for final quality.
- [ ] B3. Ground-truth collection → train residual with NLL → measure calibration coverage (→0.68)

**C — Housekeeping**
- [ ] C1. `tof_driver` 500 Hz poll cleanup (threaded blocking read; verify ToF latency doesn't regress)
- [ ] C2. Fix stale "B0472 stitches 4 cameras into one frame" comment in `camera.py` (it's native per-camera)
- [ ] C3. Camera color-correction is CPU-bound (~one core/camera at 1640×1232) — won't scale to 4 cameras; move the LUT tone-correction to the GPU (nvvidconv/CUDA) or make it optional

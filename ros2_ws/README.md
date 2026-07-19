# RingFusion ROS 2 Workspace

ROS 2 Humble workspace for the RingFusion project (ToF hub + camera driver, perception, bringup, and message definitions).

## Packages

| Package | Type | Purpose |
|---|---|---|
| `ringfusion_msgs` | ament_cmake | Custom message definitions (`ToFFrame.msg`) |
| `ringfusion_drivers` | ament_python | ToF hub (`tof_driver`), Arducam camera (`camera`), ToF heatmap colorizer (`tof_heatmap`), local combined viewer (`dual_view`) |
| `ringfusion_perception` | ament_python | Mono depth + ToF anchoring perception node |
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

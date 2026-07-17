# RingFusion ROS 2 Workspace

ROS 2 Humble workspace for the RingFusion project (ToF hub + camera driver, perception, bringup, and message definitions).

## Packages

| Package | Type | Purpose |
|---|---|---|
| `ringfusion_msgs` | ament_cmake | Custom message definitions (`ToFFrame.msg`) |
| `ringfusion_drivers` | ament_python | ToF hub (`tof_driver`) and Arducam camera (`camera`) driver nodes |
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
# Launch a full module: ToF hub + camera + perception
ros2 launch ringfusion_bringup single_module.launch.py port:=/dev/ttyACM0

# Same, but feed a still image instead of the live CSI camera
ros2 launch ringfusion_bringup single_module.launch.py image:=/path/to/shot.jpg

# Run a single node directly
ros2 run ringfusion_drivers tof_driver
ros2 run ringfusion_drivers camera
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

## Viewing the raw video feed

```bash
sudo apt-get install -y ros-humble-web-video-server   # one-time

ros2 launch ringfusion_bringup single_module.launch.py   # or: ros2 run ringfusion_drivers camera
ros2 run web_video_server web_video_server --ros-args -p port:=8080
```

Then open `http://<jetson-ip>:8080/stream?topic=/image&quality=60` in a browser on the same
network (find `<jetson-ip>` with `hostname -I`). Useful when there's no display attached to
the Jetson.

**On the `quality` param:** at the default quality (95), a 1280x720@~15Hz JPEG stream needs
~20Mbps sustained — measured steady at 2.5MB/s over loopback, no drops. If the viewing
device's WiFi link can't sustain that continuously, the server's write queue backs up and
never catches up, showing up as a growing, then roughly constant, multi-second lag even
though capture (~8ms) and ROS transport (~11ms) stay fast. `quality=60` cuts that to
~4Mbps (measured) with no visible quality loss and comfortably fits typical WiFi. Drop it
further (`quality=30`-`40`) if the lag is still there.

For a local display instead: `ros2 run rqt_image_view rqt_image_view` (subscribe to `/image`).

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

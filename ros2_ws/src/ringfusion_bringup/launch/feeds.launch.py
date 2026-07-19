"""Bring up BOTH live feeds — camera + ToF heatmap — in one command.

This is the viewing stack (no perception/fusion): camera driver, ToF driver,
ToF-heatmap colorizer, and two ways to watch them.

  ros2 launch ringfusion_bringup feeds.launch.py

Arguments (all optional):
  port    ToF serial device            (default /dev/ttyACM1  — the ESP32-C6
                                         native-USB port, idVendor 303a)
  fps     camera capture rate          (default 30; 1280x720 is the only
                                         resolution this Arducam tuning allows)
  view    open the local dual_view window on the Jetson's monitor (default true)
  web     start web_video_server for browser viewing              (default true)

LOCAL viewing (lowest latency — no network, no JPEG re-encode):
  the dual_view window opens automatically when view:=true. Run this launch
  from a terminal INSIDE the Jetson's desktop session (monitor plugged in),
  so a display is available. Over SSH with no display, pass view:=false.

BROWSER viewing (from another machine on the network):
  http://<jetson-ip>:8080/stream?topic=/image&quality=60
  http://<jetson-ip>:8080/stream?topic=/tof_heatmap
  (find <jetson-ip> with `hostname -I`)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port = LaunchConfiguration('port')
    fps = LaunchConfiguration('fps')
    view = LaunchConfiguration('view')
    web = LaunchConfiguration('web')

    return LaunchDescription([
        # A pip-installed opencv-python in ~/.local shadows JetPack's system
        # python3-opencv (the GStreamer-enabled one). Without this the camera
        # can't open nvarguscamerasrc. Both cv2 builds have GUI support, so
        # dual_view's imshow works under this too.
        SetEnvironmentVariable('PYTHONNOUSERSITE', '1'),

        DeclareLaunchArgument('port', default_value='/dev/ttyACM1'),
        DeclareLaunchArgument('fps', default_value='30'),
        DeclareLaunchArgument('view', default_value='true'),
        DeclareLaunchArgument('web', default_value='true'),

        # --- sensors ---
        Node(package='ringfusion_drivers', executable='camera',
             name='camera', output='screen',
             parameters=[{'sensor_id': 0, 'width': 1280, 'height': 720,
                          'fps': fps, 'rate': 30.0, 'frame_id': 'cam_0'}]),

        Node(package='ringfusion_drivers', executable='tof_driver',
             name='tof_driver', output='screen',
             parameters=[{'port': port, 'module_id': 0, 'frame_id': 'tof_0'}]),

        # --- ToF heatmap colorizer (/tof -> /tof_heatmap) ---
        Node(package='ringfusion_drivers', executable='tof_heatmap',
             name='tof_heatmap', output='screen'),

        # --- local combined window on the Jetson's monitor ---
        Node(package='ringfusion_drivers', executable='dual_view',
             name='dual_view', output='screen',
             condition=IfCondition(view)),

        # --- browser streaming server ---
        Node(package='web_video_server', executable='web_video_server',
             name='web_video_server', output='screen',
             parameters=[{'port': 8080}],
             condition=IfCondition(web)),
    ])

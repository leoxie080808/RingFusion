"""Bring up one RingFusion module: camera + ToF hub + perception.

  ros2 launch ringfusion_bringup single_module.launch.py \
       port:=/dev/ttyACM1

The ESP32-C6 hub enumerates as two ports; port must be its native-USB one
(idVendor 303a) where the app's TMF8829 frames actually stream, not the
UART-bridge one which only ever prints ROM bootloader text.

Use image:=/path/shot.jpg to test with a still image instead of the CSI camera.
View in rviz2: add PointCloud2 on /cloud (fixed frame: cam_0).
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    calib = os.path.join(
        get_package_share_directory('ringfusion_bringup'),
        'config', 'calibration.yaml')

    port = LaunchConfiguration('port')
    image = LaunchConfiguration('image')
    backbone_engine = LaunchConfiguration('backbone_engine')
    residual_engine = LaunchConfiguration('residual_engine')

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM1'),
        DeclareLaunchArgument('image', default_value='',
                              description='still image path; empty = CSI camera'),
        DeclareLaunchArgument('backbone_engine', default_value='',
                              description='backbone .engine path; empty = MockBackbone'),
        DeclareLaunchArgument('residual_engine', default_value='',
                              description='residual .engine path; empty = MockResidual'),

        Node(package='ringfusion_drivers', executable='tof_driver',
             name='tof_driver', output='screen',
             parameters=[{'port': port, 'module_id': 0, 'frame_id': 'tof_0'}]),

        # PYTHONNOUSERSITE=1 ONLY on the camera: it needs JetPack's system cv2 (built
        # with GStreamer) to open nvarguscamerasrc; a pip opencv-python in ~/.local
        # (no GStreamer) would shadow it. It must NOT be global -- the perception node
        # imports torch from ~/.local (user-site), which PYTHONNOUSERSITE=1 would hide.
        Node(package='ringfusion_drivers', executable='camera',
             name='camera', output='screen',
             additional_env={'PYTHONNOUSERSITE': '1'},
             parameters=[{'image': image, 'frame_id': 'cam_0',
                          'width': 1640, 'height': 1232}]),   # full-FOV binned mode

        Node(package='ringfusion_perception', executable='perception',
             name='perception', output='screen',
             parameters=[{'calib': calib, 'frame_id': 'cam_0',
                          'backbone_engine': backbone_engine,
                          'residual_engine': residual_engine}]),

        # static transform: ToF frame relative to camera (from the measured extrinsic)
        # translation in metres, lens above ToF so ToF is +y below camera.
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='cam_to_tof',
             arguments=['0', '0.020195', '0.00113', '0', '0', '0',
                        'cam_0', 'tof_0']),
    ])

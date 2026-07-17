"""Bring up one RingFusion module: camera + ToF hub + perception.

  ros2 launch ringfusion_bringup single_module.launch.py \
       port:=/dev/ttyACM0

Use image:=/path/shot.jpg to test with a still image instead of the CSI camera.
View in rviz2: add PointCloud2 on /cloud (fixed frame: cam_0).
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    calib = os.path.join(
        get_package_share_directory('ringfusion_bringup'),
        'config', 'calibration.yaml')

    port = LaunchConfiguration('port')
    image = LaunchConfiguration('image')

    return LaunchDescription([
        # A pip-installed opencv-python in ~/.local shadows JetPack's system
        # python3-opencv, which is the one built with GStreamer support.
        # Without this, cv2.VideoCapture(..., cv2.CAP_GSTREAMER) can't open
        # nvarguscamerasrc and ArducamCSI fails to start.
        SetEnvironmentVariable('PYTHONNOUSERSITE', '1'),

        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('image', default_value='',
                              description='still image path; empty = CSI camera'),

        Node(package='ringfusion_drivers', executable='tof_driver',
             name='tof_driver', output='screen',
             parameters=[{'port': port, 'module_id': 0, 'frame_id': 'tof_0'}]),

        Node(package='ringfusion_drivers', executable='camera',
             name='camera', output='screen',
             parameters=[{'image': image, 'frame_id': 'cam_0',
                          'width': 1280, 'height': 720}]),

        Node(package='ringfusion_perception', executable='perception',
             name='perception', output='screen',
             parameters=[{'calib': calib, 'frame_id': 'cam_0'}]),

        # static transform: ToF frame relative to camera (from the measured extrinsic)
        # translation in metres, lens above ToF so ToF is +y below camera.
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='cam_to_tof',
             arguments=['0', '0.020195', '0.00113', '0', '0', '0',
                        'cam_0', 'tof_0']),
    ])

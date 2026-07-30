"""Bring up one RingFusion module: camera + ToF hub + perception.

  ros2 launch ringfusion_bringup single_module.launch.py \
       port:=/dev/ttyACM1

The ESP32-C6 hub enumerates as two ports; port must be its native-USB one
(idVendor 303a) where the app's TMF8829 frames actually stream, not the
UART-bridge one which only ever prints ROM bootloader text.

Use image:=/path/shot.jpg to test with a still image instead of the CSI camera.
View in rviz2: add PointCloud2 on /cloud (fixed frame: cam_0).

For the live blend A/B, relaunch with blend:=false -- the node reads the parameter once at
construction, so `ros2 param set` on a running node will NOT switch it.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    calib = os.path.join(
        get_package_share_directory('ringfusion_bringup'),
        'config', 'calibration.yaml')

    port = LaunchConfiguration('port')
    image = LaunchConfiguration('image')
    backbone_engine = LaunchConfiguration('backbone_engine')
    residual_engine = LaunchConfiguration('residual_engine')
    # A bare LaunchConfiguration substitutes as a STRING, which fails to match the node's
    # bool/int parameter declarations ("Wrong parameter type"). ParameterValue with an
    # explicit value_type is the supported way to pass a typed override from the CLI.
    blend = ParameterValue(LaunchConfiguration('blend'), value_type=bool)
    roi_enable = ParameterValue(LaunchConfiguration('roi_enable'), value_type=bool)
    plane_refit_every = ParameterValue(LaunchConfiguration('plane_refit_every'),
                                       value_type=int)

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM1'),
        DeclareLaunchArgument('image', default_value='',
                              description='still image path; empty = CSI camera'),
        DeclareLaunchArgument('backbone_engine', default_value='',
                              description='backbone .engine path; empty = MockBackbone'),
        DeclareLaunchArgument('residual_engine', default_value='',
                              description='residual .engine path; empty = MockResidual'),
        # Stage 7c / 4b+7d. Both default ON, matching pipeline.run's own defaults. Exposed
        # so the live A/B needs a relaunch and not a rebuild -- the node reads `blend` once
        # at construction, so `ros2 param set` after startup has no effect.
        DeclareLaunchArgument('blend', default_value='true',
                              description='Stage 7c ToF/network blend'),
        DeclareLaunchArgument('roi_enable', default_value='true',
                              description='Stage 4b/7d geometric ROI'),
        DeclareLaunchArgument('plane_refit_every', default_value='1',
                              description='ground-plane RANSAC cadence, in frames'),

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
                          'width': 1640, 'height': 1232,   # full-FOV binned mode
                          'rate': 30.0}]),                 # uncap publish (node does ~27 Hz; was 15)

        Node(package='ringfusion_perception', executable='perception',
             name='perception', output='screen',
             parameters=[{'calib': calib, 'frame_id': 'cam_0',
                          'backbone_engine': backbone_engine,
                          'residual_engine': residual_engine,
                          'blend': blend, 'roi_enable': roi_enable,
                          'plane_refit_every': plane_refit_every}]),

        # static transform: ToF frame relative to camera (from the measured extrinsic)
        # translation in metres, lens above ToF so ToF is +y below camera.
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='cam_to_tof',
             arguments=['0', '0.020195', '0.00113', '0', '0', '0',
                        'cam_0', 'tof_0']),
    ])

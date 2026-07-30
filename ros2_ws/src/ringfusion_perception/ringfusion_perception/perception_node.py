"""RingFusion single-module perception node.

Subscribes:  image     (sensor_msgs/Image, rgb8)   from camera_node
             tof       (ringfusion_msgs/ToFFrame)  from tof_driver_node
Publishes:   cloud     (sensor_msgs/PointCloud2)   metric point cloud  <-- the goal
             depth     (sensor_msgs/Image, 32FC1)  metric depth map
             depth_var (sensor_msgs/Image, 32FC1)  per-pixel depth variance

Thin ROS wrapper: it caches the latest image + ToF frame and calls the pure-numpy
pipeline.run() whenever a fresh ToF frame arrives (ToF is the slower stream at
~5 Hz). All the heavy math lives in pipeline/geometry/anchoring, which are unit
tested off-robot.

Backbone and residual are pluggable. Ship with MockBackbone + MockResidual (the
mock residual is the exact identity, so output = the closed-form fit); swap in
TensorRTBackbone(engine) and ResidualRefiner(engine) once the engines are built
on the Jetson -- a one-line change each, nothing downstream changes.

Parameters:
  calib (str)            path to calibration.yaml
  frame_id (str)         frame the cloud/depth are expressed in (camera optical)
  backbone_engine (str)  path to the backbone .engine; '' -> MockBackbone
  residual_engine (str)  path to the residual .engine; '' -> MockResidual
  min_confidence (int)   reject ToF zones below this confidence and weight the fit
                         by confidence; -1 (default) = ignore confidence (uniform)
"""
import array
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from ringfusion_msgs.msg import ToFFrame

from . import geometry as geo
from . import pipeline
from . import blend as blend_mod
from . import roi
from .backbone import MockBackbone
from .residual import MockResidual
from .rectify import FisheyeRectifier
from .cloud_util import xyz_to_pointcloud2


def load_calib(path):
    import yaml
    with open(path) as f:
        c = yaml.safe_load(f)
    cam = c['camera']
    ext = c['extrinsics_cam_tof']
    rect = c.get('rectify', {})
    return {
        'K': (cam['fx'], cam['fy'], cam['cx'], cam['cy']),
        'dist': np.asarray(cam.get('dist', [0, 0, 0, 0]), float),
        'model': cam.get('model', 'fisheye'),
        'T_cam_tof': geo.make_T_cam_tof(ext['translation_mm'], ext['rotation_rpy_deg']),
        'img_w': cam['width'], 'img_h': cam['height'],
        'fov_h': c['tof']['fov_h_deg'], 'fov_v': c['tof']['fov_v_deg'],
        'rectify': {
            'width': rect.get('width', cam['width']),
            'height': rect.get('height', cam['height']),
            'balance': rect.get('balance', 0.0),
            'fov_scale': rect.get('fov_scale', 1.0),
        },
    }


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception')
        self.declare_parameter('calib', 'calibration.yaml')
        self.declare_parameter('frame_id', 'cam_0')
        self.declare_parameter('backbone_engine', '')
        self.declare_parameter('residual_engine', '')
        self.declare_parameter('min_confidence', -1)
        # Stage 7c blend (see blend.py). On by default: measured 2026-07-28 it beats both
        # sources it mixes. Exposed so it can be A/B'd live without a rebuild.
        self.declare_parameter('blend', True)
        self.declare_parameter('blend_near_deg', blend_mod.NEAR_DEG)
        self.declare_parameter('blend_far_deg', blend_mod.FAR_DEG)
        # Stage 4b/7d ROI. pipeline.run defaults roi_enable=True, so this was already ON in
        # deployment -- but it was not exposed, so there was no way to turn it off from a
        # launch file despite the README documenting `roi_enable:=false` as the fallback.
        self.declare_parameter('roi_enable', True)
        # refit_every for the ground-plane tracker; see the PlaneTracker note below.
        self.declare_parameter('plane_refit_every', 1)
        raw = load_calib(self.get_parameter('calib').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.min_confidence = int(self.get_parameter('min_confidence').value)
        self.blend = bool(self.get_parameter('blend').value)
        self.blend_near = float(self.get_parameter('blend_near_deg').value)
        self.blend_far = float(self.get_parameter('blend_far_deg').value)
        self.roi_enable = bool(self.get_parameter('roi_enable').value)
        # The node used to pass no plane_tracker at all, so pipeline.run fell through to its
        # `plane_tracker=None` default and re-RANSAC'd the ground plane from scratch on EVERY
        # frame. The 80.7 ms / 12.4 Hz offline benchmark was measured WITH a tracker, so the
        # deployed node was carrying a cost the benchmark never saw -- the timing figures did
        # not describe the shipped configuration. Hold one tracker for the node's lifetime,
        # which is also what makes the EMA and the jump-rejection in PlaneTracker do anything.
        self.plane_tracker = roi.PlaneTracker(
            refit_every=int(self.get_parameter('plane_refit_every').value))

        # Stage 1 rectifier: fisheye raw -> rectilinear. Identity until the lens
        # is calibrated (dist all zeros), so the pipeline runs unchanged today.
        r = raw['rectify']
        self.rectifier = FisheyeRectifier(
            raw['K'], raw['dist'], raw['model'],
            size_in=(raw['img_w'], raw['img_h']),
            size_out=(r['width'], r['height']),
            balance=r['balance'], fov_scale=r['fov_scale'])
        # Everything downstream runs on the rectified (pinhole) image, so the
        # pipeline's zone projection and cloud unprojection use one consistent
        # model -- the whole point of Stage 1.
        self.calib = dict(raw)
        self.calib['K'] = self.rectifier.K_rect
        self.calib['model'] = 'pinhole'
        self.calib['dist'] = np.zeros(4)
        self.calib['img_w'], self.calib['img_h'] = self.rectifier.size

        self.backbone = self._load_backbone()
        self.residual = self._load_residual()
        self.get_logger().info(
            "rectify: " + ("IDENTITY (lens not calibrated -- raw frame used as-is; "
                           "run tools/calibrate_camera.py)"
                           if self.rectifier.is_identity
                           else f"fisheye->pinhole {self.rectifier.size}"))

        self.last_image = None
        self.create_subscription(Image, 'image', self.on_image, 5)
        self.create_subscription(ToFFrame, 'tof', self.on_tof, 5)
        self.cloud_pub = self.create_publisher(PointCloud2, 'cloud', 5)
        self.depth_pub = self.create_publisher(Image, 'depth', 5)
        self.var_pub = self.create_publisher(Image, 'depth_var', 5)
        self.get_logger().info(
            f"Perception up: backbone={self.backbone.name} "
            f"residual={self.residual.name} -> /cloud, /depth, /depth_var")

    def _load_backbone(self):
        path = self.get_parameter('backbone_engine').value
        if path:
            from .backbone import TensorRTBackbone
            self.get_logger().info(f"loading backbone engine: {path}")
            return TensorRTBackbone(path)
        return MockBackbone()

    def _load_residual(self):
        path = self.get_parameter('residual_engine').value
        if path:
            from .residual import ResidualRefiner
            self.get_logger().info(f"loading residual engine: {path}")
            return ResidualRefiner(path)
        return MockResidual()

    def on_image(self, msg):
        if msg.encoding != 'rgb8':
            self.get_logger().warn(f"expected rgb8, got {msg.encoding}")
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.last_image = self.rectifier.rectify(img)   # Stage 1

    def on_tof(self, msg):
        if self.last_image is None:
            return  # wait for first camera frame
        dist_m = np.asarray(msg.dist_m, dtype=np.float32).reshape(msg.rows, msg.cols)
        valid = np.isfinite(dist_m)
        confidence = None
        if len(msg.confidence) == msg.rows * msg.cols:
            confidence = np.asarray(msg.confidence, np.uint8).reshape(msg.rows, msg.cols)

        res = pipeline.run(self.last_image, dist_m, valid, self.calib,
                           self.backbone, self.residual,
                           confidence=confidence, min_confidence=self.min_confidence,
                           blend=self.blend, blend_near=self.blend_near,
                           blend_far=self.blend_far, roi_enable=self.roi_enable,
                           plane_tracker=self.plane_tracker)
        if not res['ok']:
            self.get_logger().warn(
                f"anchoring failed ({res['n_anchors']} anchors this frame)")
            return

        # publish cloud (the goal)
        cloud_msg = xyz_to_pointcloud2(res['cloud'], self.frame_id, msg.header.stamp)
        self.cloud_pub.publish(cloud_msg)
        # publish metric depth and per-pixel variance as 32FC1
        self.depth_pub.publish(self._float_image(res['metric'], msg.header.stamp))
        if res['var'] is not None:
            self.var_pub.publish(self._float_image(res['var'], msg.header.stamp))

    def _float_image(self, arr, stamp):
        """Pack an HxW float array into a 32FC1 sensor_msgs/Image."""
        d = np.ascontiguousarray(arr, dtype=np.float32)
        m = Image()
        m.header.stamp = stamp
        m.header.frame_id = self.frame_id
        m.height, m.width = d.shape
        m.encoding = '32FC1'
        m.is_bigendian = 0
        m.step = d.shape[1] * 4
        # array.array hits the message data setter's fast path; raw bytes would be
        # validated element-by-element (~600ms for this 3.7MB map). See cloud_util.
        m.data = array.array('B', d.tobytes())
        return m


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

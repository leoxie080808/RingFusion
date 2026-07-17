"""RingFusion single-module perception node.

Subscribes:  image (sensor_msgs/Image, rgb8)   from camera_node
             tof   (ringfusion_msgs/ToFFrame)  from tof_driver_node
Publishes:   cloud (sensor_msgs/PointCloud2)   metric point cloud  <-- the goal
             depth (sensor_msgs/Image, 32FC1)  metric depth map

Caches the latest image and ToF frame and runs pipeline.run() whenever a fresh
ToF frame arrives (ToF is the slower stream at ~5 Hz). The heavy math is the
tested geometry/anchoring/backbone imported unchanged.

Parameters:
  calib (str)   path to calibration.yaml
  frame_id (str) frame the cloud is expressed in (the camera optical frame)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ringfusion_msgs.msg import ToFFrame

from . import geometry as geo
from . import anchoring as anc
from .backbone import MockBackbone
from .cloud_util import xyz_to_pointcloud2


def load_calib(path):
    import yaml
    with open(path) as f:
        c = yaml.safe_load(f)
    cam = c['camera']
    ext = c['extrinsics_cam_tof']
    return {
        'K': (cam['fx'], cam['fy'], cam['cx'], cam['cy']),
        'dist': np.asarray(cam.get('dist', [0, 0, 0, 0]), float),
        'model': cam.get('model', 'fisheye'),
        'T_cam_tof': geo.make_T_cam_tof(ext['translation_mm'], ext['rotation_rpy_deg']),
        'img_w': cam['width'], 'img_h': cam['height'],
        'fov_h': c['tof']['fov_h_deg'], 'fov_v': c['tof']['fov_v_deg'],
    }


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception')
        self.declare_parameter('calib', 'calibration.yaml')
        self.declare_parameter('frame_id', 'cam_0')
        self.calib = load_calib(self.get_parameter('calib').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.backbone = MockBackbone()   # swap TensorRTBackbone once engine exists

        self.last_image = None
        self.create_subscription(Image, 'image', self.on_image, 5)
        self.create_subscription(ToFFrame, 'tof', self.on_tof, 5)
        self.cloud_pub = self.create_publisher(__import__(
            'sensor_msgs.msg', fromlist=['PointCloud2']).PointCloud2, 'cloud', 5)
        self.depth_pub = self.create_publisher(Image, 'depth', 5)
        self.get_logger().info("Perception: image + tof -> /cloud, /depth")

    def on_image(self, msg):
        if msg.encoding != 'rgb8':
            self.get_logger().warn(f"expected rgb8, got {msg.encoding}")
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.last_image = img

    def on_tof(self, msg):
        if self.last_image is None:
            return  # wait for first camera frame
        dist_m = np.asarray(msg.dist_m, dtype=np.float32).reshape(msg.rows, msg.cols)
        valid = np.isfinite(dist_m)
        res = self.run_pipeline(self.last_image, dist_m, valid)
        if not res['ok']:
            self.get_logger().warn(
                f"anchoring failed ({res['n_anchors']} anchors this frame)")
            return
        # publish cloud (the goal)
        cloud_msg = xyz_to_pointcloud2(res['cloud'], self.frame_id, msg.header.stamp)
        self.cloud_pub.publish(cloud_msg)
        # publish metric depth as 32FC1
        d = res['metric'].astype(np.float32)
        dm = Image()
        dm.header = msg.header
        dm.header.frame_id = self.frame_id
        dm.height, dm.width = d.shape
        dm.encoding = '32FC1'
        dm.is_bigendian = 0
        dm.step = d.shape[1] * 4
        dm.data = d.tobytes()
        self.depth_pub.publish(dm)

    def run_pipeline(self, rgb, dist_m, valid):
        """Inlined pipeline.run() using the imported tested math."""
        h, w = rgb.shape[:2]
        K = self.calib['K']
        rows, cols = dist_m.shape
        disp = self.backbone.infer(rgb)
        proj = geo.project_zone_to_pixel(
            dist_m, valid, cols, rows, self.calib['fov_h'], self.calib['fov_v'],
            self.calib['T_cam_tof'], K, self.calib['dist'], model=self.calib['model'])
        uv = proj['uv']; z = proj['z_cam']; ok = proj['valid']
        finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z)
        u = np.round(np.where(finite, uv[:, 0], -1)).astype(int)
        v = np.round(np.where(finite, uv[:, 1], -1)).astype(int)
        inb = ok & finite & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
        du = disp[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)]
        disp_at = du[inb]; inv_depth = 1.0 / z[inb]
        weights = np.ones_like(inv_depth)
        fit = anc.solve_robust(disp_at, inv_depth, weights, iters=1)
        if fit is None:
            return {'ok': False, 'n_anchors': int(inb.sum())}
        a, b = fit
        metric = anc.to_metric_depth(disp, a, b)
        cloud = geo.unproject_depth_to_cloud(metric, K, model='pinhole', stride=2)
        return {'ok': True, 'n_anchors': int(inb.sum()),
                'metric': metric, 'cloud': cloud, 'a': a, 'b': b}


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

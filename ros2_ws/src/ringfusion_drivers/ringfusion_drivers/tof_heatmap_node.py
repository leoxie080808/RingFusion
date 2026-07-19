"""Publishes a colorized heatmap image from ringfusion_msgs/ToFFrame, for live viewing.

Subscribes:  tof         (ringfusion_msgs/ToFFrame)
Publishes:   tof_heatmap (sensor_msgs/Image, bgr8)

Parameters:
  scale                     upscale factor per zone (48x32 -> cols*scale x rows*scale)
  min_range_m, max_range_m  distance range mapped across the colormap
"""
import array
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from ringfusion_msgs.msg import ToFFrame

WEAK_CONFIDENCE = 6


class ToFHeatmapNode(Node):
    def __init__(self):
        super().__init__('tof_heatmap')
        self.declare_parameter('scale', 20)
        self.declare_parameter('min_range_m', 0.1)
        self.declare_parameter('max_range_m', 3.0)
        self.scale = int(self.get_parameter('scale').value)
        self.min_range = float(self.get_parameter('min_range_m').value)
        self.max_range = float(self.get_parameter('max_range_m').value)

        self.pub = self.create_publisher(Image, 'tof_heatmap', 10)
        self.create_subscription(ToFFrame, 'tof', self.on_tof, 5)
        self.get_logger().info("ToF heatmap: /tof -> /tof_heatmap")

    def on_tof(self, msg):
        dist_m = np.asarray(msg.dist_m, dtype=np.float32).reshape(msg.rows, msg.cols)
        conf = np.asarray(msg.confidence, dtype=np.uint8).reshape(msg.rows, msg.cols)

        span = self.max_range - self.min_range
        norm = np.clip((dist_m - self.min_range) / span, 0, 1)
        invalid = ~np.isfinite(dist_m)
        gray = (norm * 255).astype(np.uint8)
        gray[invalid] = 0
        color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        color[invalid] = (0, 0, 0)
        weak = (conf < WEAK_CONFIDENCE) & ~invalid
        color[weak] = (color[weak] * 0.5).astype(np.uint8)

        big = cv2.resize(color, (msg.cols * self.scale, msg.rows * self.scale),
                          interpolation=cv2.INTER_NEAREST)

        h, w = big.shape[:2]
        out = Image()
        out.header = Header()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.height = h
        out.width = w
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = w * 3
        out.data = array.array('B', big.tobytes())
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ToFHeatmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

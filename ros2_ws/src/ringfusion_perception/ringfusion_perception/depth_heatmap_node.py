"""Colorized heatmap of the perception depth output, for live viewing.

/depth is published as raw 32FC1 metric depth (metres) -- image viewers and
web_video_server can't show that directly. This node normalizes it over a distance
range and applies a colormap, mirroring tof_heatmap for the dense depth map.

Subscribes:  depth         (sensor_msgs/Image, 32FC1)  metric depth from perception
Publishes:   depth_heatmap (sensor_msgs/Image, bgr8)   colorized, viewable

View it:
  - LOCAL window on the Jetson's monitor:  -p show:=true
  - BROWSER: http://<jetson-ip>:8080/stream?topic=/depth_heatmap  (web_video_server)

Parameters:
  min_range_m, max_range_m  distance range mapped across the colormap (default 0.1..5.0)
  show                      also open a local cv2 window (needs the Jetson's display)
"""
import array
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

WIN = 'depth_heatmap'


class DepthHeatmapNode(Node):
    def __init__(self):
        super().__init__('depth_heatmap')
        self.declare_parameter('min_range_m', 0.1)
        self.declare_parameter('max_range_m', 5.0)
        self.declare_parameter('show', False)
        self.min_range = float(self.get_parameter('min_range_m').value)
        self.max_range = float(self.get_parameter('max_range_m').value)
        self.show = bool(self.get_parameter('show').value)
        self._latest = None
        if self.show:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            # GUI on a timer, not in the ~5 Hz depth callback (else the window starves
            # and the WM shows "not responding").
            self.create_timer(1.0 / 30.0, self._display)

        self.pub = self.create_publisher(Image, 'depth_heatmap', 5)
        self.create_subscription(Image, 'depth', self.on_depth, 5)
        self.get_logger().info(
            f"depth heatmap: /depth -> /depth_heatmap "
            f"[{self.min_range:.2f}..{self.max_range:.2f} m] show={self.show}")

    def on_depth(self, msg):
        if msg.encoding != '32FC1':
            self.get_logger().warn(f"expected 32FC1, got {msg.encoding}")
            return
        depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)

        span = max(self.max_range - self.min_range, 1e-6)
        norm = np.clip((depth - self.min_range) / span, 0, 1)
        invalid = ~np.isfinite(depth) | (depth <= 0)
        gray = (norm * 255).astype(np.uint8)
        gray[invalid] = 0
        color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        color[invalid] = (0, 0, 0)                       # black where no depth

        out = Image()
        out.header = msg.header
        out.height, out.width = color.shape[:2]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = out.width * 3
        out.data = array.array('B', color.tobytes())
        self.pub.publish(out)
        self._latest = color

    def _display(self):
        frame = self._latest
        if frame is None:
            frame = np.zeros((240, 480, 3), np.uint8)
            cv2.putText(frame, 'waiting for /depth ...', (24, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DepthHeatmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if node.show:
            cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

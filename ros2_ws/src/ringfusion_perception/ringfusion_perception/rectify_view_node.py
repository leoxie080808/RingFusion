"""Live rectification check: publishes raw | rectified side by side so you can
confirm Stage 1 visually (point the camera at straight edges -- a door frame,
floor tiles -- and watch them straighten on the right).

Subscribes: image           (sensor_msgs/Image, rgb8)  raw camera frame
Publishes:  rectify_compare  (sensor_msgs/Image, bgr8)  [ raw | rectified ]

Two ways to watch it:
  - LOCAL window on the Jetson's monitor (lowest latency, full refresh -- the
    composite is 2560x720, i.e. 2x bandwidth, so this is much smoother than the
    browser over WiFi):
        ros2 run ringfusion_perception rectify_view --ros-args -p show:=true
    Run it in the Jetson's desktop session (monitor attached). q/ESC to quit.
  - BROWSER via web_video_server (works headless, but heavier over WiFi):
        http://<jetson-ip>:8080/stream?topic=/rectify_compare&quality=40

Uses the SAME FisheyeRectifier + calibration.yaml the perception node uses, so
what you see here is exactly the Stage 1 the pipeline runs. Tune `rectify:`
(fov_scale / balance) in calibration.yaml and relaunch to compare.

Parameters:
  calib (path to calibration.yaml)
  show  (bool) also open a local cv2 window (needs the Jetson's display)
"""
import array
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from .rectify import FisheyeRectifier
from .perception_node import load_calib

FONT = cv2.FONT_HERSHEY_SIMPLEX


class RectifyView(Node):
    def __init__(self):
        super().__init__('rectify_view')
        self.declare_parameter('calib', 'calibration.yaml')
        self.declare_parameter('show', False)
        self.show = bool(self.get_parameter('show').value)
        if self.show:
            cv2.namedWindow('rectify_compare', cv2.WINDOW_NORMAL)
        c = load_calib(self.get_parameter('calib').value)
        r = c['rectify']
        self.rect = FisheyeRectifier(
            c['K'], c['dist'], c['model'], size_in=(c['img_w'], c['img_h']),
            size_out=(r['width'], r['height']), balance=r['balance'],
            fov_scale=r['fov_scale'])
        self._label = ('RECTIFIED (identity - pinhole/no cv2)'
                       if self.rect.is_identity else 'RECTIFIED')
        self.pub = self.create_publisher(Image, 'rectify_compare', 5)
        self.create_subscription(Image, 'image', self.on_image, 5)
        self.get_logger().info(
            f"rectify_view: /image -> /rectify_compare | identity={self.rect.is_identity} "
            f"K_rect={tuple(round(v,1) for v in self.rect.K_rect)}")

    def on_image(self, msg):
        if msg.encoding != 'rgb8':
            self.get_logger().warn(f"expected rgb8, got {msg.encoding}")
            return
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        raw = rgb[:, :, ::-1].copy()               # -> bgr for cv2
        rect = self.rect.rectify(raw)
        # match heights so hstack works even if the rectified size differs
        if rect.shape[0] != raw.shape[0]:
            s = raw.shape[0] / rect.shape[0]
            rect = cv2.resize(rect, (int(rect.shape[1] * s), raw.shape[0]))
        cv2.putText(raw, 'RAW fisheye', (24, 44), FONT, 1.1, (0, 255, 0), 3)
        cv2.putText(rect, self._label, (24, 44), FONT, 1.1, (0, 255, 0), 3)
        sbs = np.ascontiguousarray(np.hstack([raw, rect]))

        out = Image()
        out.header = msg.header
        out.height, out.width = sbs.shape[:2]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = sbs.shape[1] * 3
        out.data = array.array('B', sbs.tobytes())
        self.pub.publish(out)

        if self.show:
            cv2.imshow('rectify_compare', sbs)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RectifyView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

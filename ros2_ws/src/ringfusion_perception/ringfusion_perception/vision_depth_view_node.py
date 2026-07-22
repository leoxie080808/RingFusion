"""Side-by-side [ rectified camera | colorized depth ] for validating perception.

The pipeline computes depth on the RECTIFIED image, so rectifying /image here with
the SAME calibration lines it up pixel-for-pixel with /depth. That makes this the
real "is the depth correct?" check: a near object in the camera should appear
near/red at the SAME place in the depth, and left/right must not be swapped (which
would reveal a ToF-orientation or extrinsics mistake).

Subscribes: image (Image rgb8), depth (Image 32FC1)
Publishes:  vision_depth (Image bgr8)   [ rectified cam | depth heatmap ]

View: -p show:=true for a local window, or web_video_server on /vision_depth.
Params: calib, min_range_m, max_range_m, show
"""
import array
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from .rectify import FisheyeRectifier
from .perception_node import load_calib

WIN = 'vision_depth'
FONT = cv2.FONT_HERSHEY_SIMPLEX


class VisionDepthView(Node):
    def __init__(self):
        super().__init__('vision_depth_view')
        self.declare_parameter('calib', 'calibration.yaml')
        self.declare_parameter('min_range_m', 0.1)
        self.declare_parameter('max_range_m', 5.0)
        self.declare_parameter('show', False)
        self.min_range = float(self.get_parameter('min_range_m').value)
        self.max_range = float(self.get_parameter('max_range_m').value)
        self.show = bool(self.get_parameter('show').value)

        c = load_calib(self.get_parameter('calib').value)
        r = c['rectify']
        self.rect = FisheyeRectifier(
            c['K'], c['dist'], c['model'], size_in=(c['img_w'], c['img_h']),
            size_out=(r['width'], r['height']), balance=r['balance'],
            fov_scale=r['fov_scale'])
        self._bgr = None                       # latest rectified-ready camera frame
        self._latest = None                    # latest composed side-by-side to show

        if self.show:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            # Pump the GUI on a timer, NOT in the data callback: /depth is only ~5 Hz,
            # so calling waitKey there starves the window event loop and the WM marks it
            # "not responding" (and it freezes if /depth pauses). 30 Hz keeps it live.
            self.create_timer(1.0 / 30.0, self._display)
        self.pub = self.create_publisher(Image, 'vision_depth', 5)
        self.create_subscription(Image, 'image', self.on_image, 5)
        self.create_subscription(Image, 'depth', self.on_depth, 5)
        self.get_logger().info("vision+depth: /image + /depth -> /vision_depth")

    def on_image(self, msg):
        if msg.encoding != 'rgb8':
            return
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        self._bgr = rgb[:, :, ::-1].copy()     # rgb -> bgr for cv2

    def _colorize(self, depth):
        span = max(self.max_range - self.min_range, 1e-6)
        norm = np.clip((depth - self.min_range) / span, 0, 1)
        invalid = ~np.isfinite(depth) | (depth <= 0)
        gray = (norm * 255).astype(np.uint8)
        gray[invalid] = 0
        color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        color[invalid] = (0, 0, 0)
        return color

    def on_depth(self, msg):
        if msg.encoding != '32FC1' or self._bgr is None:
            return
        depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        dcolor = self._colorize(depth)
        rect = self.rect.rectify(self._bgr)
        if rect.shape[:2] != dcolor.shape[:2]:                  # match for hstack
            rect = cv2.resize(rect, (dcolor.shape[1], dcolor.shape[0]))
        cv2.putText(rect, 'RECTIFIED CAM', (16, 34), FONT, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(dcolor, 'DEPTH (near=red)', (16, 34), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        sbs = np.ascontiguousarray(np.hstack([rect, dcolor]))

        out = Image()
        out.header = msg.header
        out.height, out.width = sbs.shape[:2]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = out.width * 3
        out.data = array.array('B', sbs.tobytes())
        self.pub.publish(out)
        self._latest = sbs                     # timer displays it; keeps GUI responsive

    def _display(self):
        frame = self._latest
        if frame is None:
            frame = np.zeros((240, 760, 3), np.uint8)
            cv2.putText(frame, 'waiting for /image + /depth ...', (24, 130),
                        FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = VisionDepthView()
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

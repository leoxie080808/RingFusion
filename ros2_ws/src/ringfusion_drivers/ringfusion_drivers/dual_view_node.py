"""Local combined viewer: camera + ToF heatmap in ONE window, on the Jetson's
own display. This is the lowest-latency way to watch both feeds — it subscribes
to the ROS topics directly and shows them with cv2.imshow, with no web_video_server
JPEG re-encode and no network hop (the two things that add the visible lag when
viewing over WiFi in a browser).

Run it IN the Jetson's desktop session (a monitor plugged in, logged in), so a
display is available:

    ros2 run ringfusion_drivers dual_view

Press 'q' or ESC in the window to quit.

Subscribes: image (sensor_msgs/Image, rgb8), tof_heatmap (sensor_msgs/Image, bgr8)
Parameters:
  panel_height  each panel is scaled to this height (px); panels sit side by side
"""
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

WINDOW = 'RingFusion — camera | ToF'


class _Rate:
    """Rolling FPS estimate from message arrival times."""
    def __init__(self, n=30):
        self.t = []
        self.n = n
    def tick(self):
        now = time.time()
        self.t.append(now)
        if len(self.t) > self.n:
            self.t.pop(0)
    @property
    def fps(self):
        return (len(self.t) - 1) / (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


class DualView(Node):
    def __init__(self):
        super().__init__('dual_view')
        self.declare_parameter('panel_height', 540)
        self.ph = int(self.get_parameter('panel_height').value)

        self.cam = None
        self.tof = None
        self.cam_rate = _Rate()
        self.tof_rate = _Rate()

        self.create_subscription(Image, 'image', self.on_cam, 5)
        self.create_subscription(Image, 'tof_heatmap', self.on_tof, 5)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        # render on a wall-clock timer so the window stays responsive even if
        # one stream stalls
        self.create_timer(1.0 / 30.0, self.render)
        self.get_logger().info("dual_view: image + tof_heatmap -> window (press q/ESC to quit)")

    def on_cam(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        # camera publishes rgb8; cv2 shows bgr
        self.cam = img[:, :, ::-1].copy() if msg.encoding == 'rgb8' else img.copy()
        self.cam_rate.tick()

    def on_tof(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.tof = img.copy()   # already bgr8
        self.tof_rate.tick()

    def _panel(self, img, title, fps):
        if img is None:
            p = np.full((self.ph, int(self.ph * 16 / 9), 3), 40, np.uint8)
            cv2.putText(p, title + ' (waiting...)', (20, self.ph // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            return p
        h, w = img.shape[:2]
        p = cv2.resize(img, (int(w * self.ph / h), self.ph))
        cv2.putText(p, f'{title}  {fps:4.1f} Hz', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return p

    def render(self):
        left = self._panel(self.cam, 'camera', self.cam_rate.fps)
        right = self._panel(self.tof, 'ToF', self.tof_rate.fps)
        cv2.imshow(WINDOW, np.hstack([left, right]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DualView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

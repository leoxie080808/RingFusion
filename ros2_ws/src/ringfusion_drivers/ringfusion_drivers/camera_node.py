"""Publishes sensor_msgs/Image from an Arducam CSI camera (or a still image).

Wraps camera.ArducamCSI (nvarguscamerasrc) / camera.ImageFile. Publishes raw
RGB8 without cv_bridge (hand-packed) to avoid an extra dependency.
Parameters:
  sensor_id (int)  CSI sensor id on the B0472
  width,height,fps
  flip      (int)  nvvidconv flip-method; 2 = rotate 180 (default, matches
                    the ring mount), 0 = no rotation
  image     (str)  if set, serve this still image instead of the camera
  frame_id  (str)  tf frame, e.g. cam_0
  rate      (float) publish rate when using a still image
"""
import array
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from . import camera as camlib


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera')
        self.declare_parameter('sensor_id', 0)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 15)
        self.declare_parameter('flip', camlib.FLIP_180)
        self.declare_parameter('image', '')
        self.declare_parameter('frame_id', 'cam_0')
        self.declare_parameter('rate', 15.0)

        w = int(self.get_parameter('width').value)
        h = int(self.get_parameter('height').value)
        self.frame_id = self.get_parameter('frame_id').value
        image = self.get_parameter('image').value

        self.pub = self.create_publisher(Image, 'image', 10)
        if image:
            self.src = camlib.ImageFile(image)
            self.get_logger().info(f"Camera serving still image {image} -> /image")
        else:
            self.src = camlib.ArducamCSI(
                sensor_id=int(self.get_parameter('sensor_id').value),
                width=w, height=h, fps=int(self.get_parameter('fps').value),
                flip=int(self.get_parameter('flip').value))
            self.get_logger().info("Arducam CSI -> /image")
        period = 1.0 / float(self.get_parameter('rate').value)
        self.timer = self.create_timer(period, self.tick)

    def tick(self):
        rgb = self.src.read()
        if rgb is None:
            return
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = h
        msg.width = w
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = w * 3
        # array.array hits the generated setter's fast path; assigning bytes/list
        # instead makes it validate every element in a Python loop (~450ms/frame here).
        msg.data = array.array('B', rgb.tobytes())
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self.src.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

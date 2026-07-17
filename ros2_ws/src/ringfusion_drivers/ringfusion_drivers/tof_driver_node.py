"""Publishes ToFFrame messages by reading the ESP32-C6 hub over USB serial.

Wraps the tested tof_source.SerialToFSource (real firmware CSV/subframe format).
Parameters:
  port      (str)  serial device, e.g. /dev/ttyACM0
  baud      (int)  115200 (native USB CDC; value is nominal)
  module_id (int)  0..3, tags which ring module this hub serves
  frame_id  (str)  tf frame for this module's ToF, e.g. tof_0
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from ringfusion_msgs.msg import ToFFrame

from .tof_source import SerialToFSource


class ToFDriverNode(Node):
    def __init__(self):
        super().__init__('tof_driver')
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('module_id', 0)
        self.declare_parameter('frame_id', 'tof_0')

        port = self.get_parameter('port').value
        baud = int(self.get_parameter('baud').value)
        self.module_id = int(self.get_parameter('module_id').value)
        self.frame_id = self.get_parameter('frame_id').value

        self.pub = self.create_publisher(ToFFrame, 'tof', 10)
        try:
            self.src = SerialToFSource(port, baud)
        except Exception as e:
            self.get_logger().error(
                f"Could not open {port}: {e}. Is the ESP32 connected and the "
                f"port free (no idf.py monitor / serial viewer holding it)?")
            raise
        self.get_logger().info(
            f"ToF driver module {self.module_id} reading {port} -> /tof")
        # poll fast; frames only publish when a full 48x32 is assembled (~5 Hz)
        self.timer = self.create_timer(0.002, self.poll)

    def poll(self):
        frame = self.src.read()
        if frame is None:
            return
        msg = ToFFrame()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.module_id = self.module_id
        msg.rows, msg.cols = frame.dist_mm.shape
        # NaN survives in float32[]; consumers use it as the invalid marker
        msg.dist_m = frame.dist_m.reshape(-1).astype(np.float32).tolist()
        msg.confidence = frame.confidence.reshape(-1).astype(np.uint8).tolist()
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self.src.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ToFDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

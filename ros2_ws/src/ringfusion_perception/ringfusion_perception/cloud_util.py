"""Build sensor_msgs/PointCloud2 from an (N,3) float array (+ optional rgb)."""
import array
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def xyz_to_pointcloud2(points, frame_id, stamp, rgb=None):
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    if rgb is None:
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 12
        # array.array('B', ...) hits the message data setter's fast path;
        # assigning raw bytes makes rclpy validate all ~2.8MB element-by-element
        # in Python (~450ms/frame here), which alone pins a core at 5Hz.
        data = array.array('B', pts.tobytes())
    else:
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        point_step = 16
        c = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        rgb_packed = ((c[:, 0].astype(np.uint32) << 16) |
                      (c[:, 1].astype(np.uint32) << 8) |
                       c[:, 2].astype(np.uint32))
        arr = np.zeros(n, dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')])
        arr['x'] = pts[:, 0]; arr['y'] = pts[:, 1]; arr['z'] = pts[:, 2]
        arr['rgb'] = rgb_packed
        data = array.array('B', arr.tobytes())

    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.is_dense = True
    msg.data = data
    return msg

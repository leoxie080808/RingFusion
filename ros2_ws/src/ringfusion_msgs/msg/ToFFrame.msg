# A single reassembled TMF8829 depth frame from one ring module.
std_msgs/Header header      # stamp = capture time, frame_id = module tof frame
uint8  module_id            # 0..3, which side of the ring
uint16 cols                 # zone grid width  (48)
uint16 rows                 # zone grid height (32)
float32[] dist_m            # rows*cols, row-major; NaN where no return
uint8[]   confidence        # rows*cols, per-zone confidence

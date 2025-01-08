import sys

print(sys.executable)

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from image_transport_py import ImageTransport
from cv_bridge import CvBridge
import rerun as rr

from sensor_msgs.msg import Image as ImageMsg

from pdb import set_trace as bp
import numpy as np


class RerunForwarder(Node):
    def __init__(self):
        super().__init__("rerun_forwarder")
        rr.init("zoo-vision-py")
        rr.serve_web(open_browser=True, server_memory_limit="1GB")

        self.cv_bridge = CvBridge()

        image_transport = ImageTransport("imagetransport_sub", image_transport="raw")
        image_transport.subscribe("input_camera/image", 1, self.image_callback)

    def image_callback(self, msg: ImageMsg):
        time = Time.from_msg(msg.header.stamp)
        # self.get_logger().info(
        #     f"got a new image from time:={time.nanoseconds / 1e9:.3f}, id={msg.header.frame_id}"
        # )
        rr.set_time_nanos("ros_time", time.nanoseconds)

        rr.log(
            "camera/img",
            rr.Image(
                bytes=np.ndarray(
                    shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data
                ),
                width=msg.width,
                height=msg.height,
                color_model="bgr",
                datatype="U8",
            ),
        )


def main():
    print("Starting zoo_vision_py package...")
    rclpy.init()

    rerun_forwarder = RerunForwarder()

    print("Listening for events...")
    rclpy.spin(rerun_forwarder)

    print("Shutting down...")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

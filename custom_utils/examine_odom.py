# !/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
"""

# ros2 topic echo /a200_0648/platform/odom --field pose.pose
"""

class OdomDegreeConverter(Node):
    def __init__(self):
        super().__init__('odom_degree_converter')
        self.subscription = self.create_subscription(
            Odometry,
            '/a200_0648/platform/odom',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        # 1. Extract translation (Position)
        pos = msg.pose.pose.position
        x_m = pos.x
        y_m = pos.y
        z_m = pos.z

        # 2. Extract rotation (Orientation)
        q = msg.pose.pose.orientation

        # Pure math conversion from Quaternion to Euler (Roll, Pitch, Yaw)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        # Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.degrees(math.copysign(math.pi / 2, sinp))  # use 90 degrees if out of range
        else:
            pitch = math.degrees(math.asin(sinp))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        # Print translation and rotation side-by-side cleanly
        self.get_logger().info(
            f"\n=== Odometry Status ===\n"
            f"Position (Meters):\n"
            f"  X: {x_m:6.2f} | Y: {y_m:6.2f} | Z: {z_m:6.2f}\n"
            f"Orientation (Degrees):\n"
            f"  Roll: {roll:5.1f}° | Pitch: {pitch:5.1f}° | Yaw: {yaw:5.1f}°\n"
            f"======================="
        )


def main(args=None):
    rclpy.init(args=args)
    node = OdomDegreeConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

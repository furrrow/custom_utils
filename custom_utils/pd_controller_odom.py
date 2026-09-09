import time
import numpy as np
import yaml
from typing import Tuple
import argparse
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy
# Message types
from std_msgs.msg import Empty
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

from pd_controller_naive import ROSData, pd_controller
"""
combined pd_controller fro flownav into the path_manager pipeline:
https://github.com/utn-air/flownav/blob/main/deployment/src/pd_controller.py
also see the planner_dwa_ros2.py

"""

class PDControllerNode(Node):
    def __init__(self, config_path: str, robot_name: str):
        super().__init__("pd_controller")
        self.vel_msg = Twist()
        WAYPOINT_TIMEOUT = 1  # seconds
        self.waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
        self.reached_goal = False
        self.reverse_mode = False
        self.robot_name = robot_name
        # CONSTS
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.rate = config["frame_rate"]

        robot_config = config[robot_name]
        print(f"using robot config for: {robot_name}")
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]

        # ROS Topics
        WAYPOINT_TOPIC = robot_config['waypoint_topic']
        ODOM_TOPIC = robot_config['odom_topic']
        REACHED_GOAL_TOPIC = robot_config['reached_goal_topic']
        VEL_TOPIC = robot_config['vel_topic']
        GOAL_TOPIC = robot_config['goal_topic']
        print("VEL_TOPIC", VEL_TOPIC)
        self.robot_radius = robot_config['robot_radius']
        self.dt = 1 / self.rate

        self.x = None
        self.y = None
        self.yaw = None
        self.v_x = 0.0
        self.w_z = 0.0

        # State space representation
        self.X = np.array([self.x, self.y, self.yaw, self.v_x, self.w_z])
        self.U = np.array([self.v_x, self.w_z])

        self.odom_assigned = False
        self.goalX = None
        self.goalY = None
        self._goal_req_sent = False

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.qos_profile_r = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.waypoint_sub = self.create_subscription(Float32MultiArray, 
                                                     WAYPOINT_TOPIC, 
                                                     self.callback_drive, 
                                                     qos_profile = self.qos_profile_r)
        self.req_goal_pub = self.create_publisher(Empty, "/req_goal", 10)
        self.reached_goal_sub = self.create_subscription(Bool,  REACHED_GOAL_TOPIC,
                                                         self.callback_reached_goal, 10)
        self.vel_out = self.create_publisher(Twist, VEL_TOPIC, 10)
        self.sub_goal = self.create_subscription(PoseStamped, GOAL_TOPIC, self.on_goal_cartesian_wf, self.qos_profile)
        self.sub_odom = self.create_subscription(Odometry, ODOM_TOPIC, self.on_odom, self.qos_profile)
        self.timer = self.create_timer(1.0 / self.rate, self.main_loop)
        self.get_logger().info("Registered with master node. Waiting for waypoints...")

    def callback_drive(self, waypoint_msg: Float32MultiArray):
        """Callback function for the waypoint subscriber"""
        self.get_logger().info("Setting waypoint")
        self.waypoint.set(waypoint_msg.data)

    def callback_reached_goal(self, reached_goal_msg: Bool):
        """Callback function for the reached goal subscriber"""
        self.reached_goal = reached_goal_msg.data

    def goalDefined(self):
        if self.goalX is not None and self.goalY is not None:
            return True

        if not self._goal_req_sent:
            self.req_goal_pub.publish(Empty())
            self._goal_req_sent = True
        return False

    def atGoal(self):
        if not self.odom_assigned:
            return False
        elif self.goalX is None or self.goalY is None:
            return False
        elif np.linalg.norm(self.X[:2] - np.array([self.goalX, self.goalY])) <= self.robot_radius:
            if not self._goal_req_sent:
                self.req_goal_pub.publish(Empty())
                self._goal_req_sent = True
            return True
        return False

    # Callback for Odometry
    def on_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        roll, pitch, yaw = R.from_quat([rot_q.x, rot_q.y, rot_q.z, rot_q.w]).as_euler('xyz')
        self.yaw = yaw

        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z

        self.X[0] = self.x
        self.X[1] = self.y
        self.X[2] = self.yaw
        self.X[3] = self.v_x
        self.X[4] = self.w_z
        self.odom_assigned = True

    def on_goal_cartesian_wf(self, msg):
        """
        Goals defined wrt odom frame in cartesian coordinates
            msg.linear.x: x     (m)
            msg.linear.y: y     (m)
        """
        self.goalX = msg.pose.position.x
        self.goalY = msg.pose.position.y
        self._goal_req_sent = False

    def main_loop(self):
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0
        if self.reached_goal:
            self.vel_out.publish(Twist()) # Clean stop
            self.get_logger().info("model called reached_goal topic! Stopping...")
            self.timer.cancel()
            return
        if not self.odom_assigned:
            return

        if not self.goalDefined():
            self.get_logger().info("Goal not defined!")
        elif self.atGoal() and self.goalX is not None and self.goalY is not None:
            self.get_logger().info("Subgoal reached!")
            self.goalX = None
            self.goalY = None
        elif self.waypoint.is_valid(verbose=True):
            t1 = time.time()
            v, w = pd_controller(self.waypoint.get(), self.max_v, self.max_w, self.dt, eps=1e-8)
            if self.reverse_mode:
                v *= -1
            self.vel_msg.linear.x = v
            self.vel_msg.angular.z = w
            t2 = time.time()
            self.get_logger().info(f"PD controller: v {v} w  {w}"
                               f"Time taken: {t2 - t1:.4f} seconds")
        else:
            self.get_logger().info("waypoint not valid!")
        self.vel_out.publish(self.vel_msg)

def main():
    parser = argparse.ArgumentParser(description="Run the PD Controller")
    parser.add_argument("-r", "--robot", type=str, help="Robot Name",
                        default="husky")
    parser.add_argument("--config", type=str, help="yaml config file",
                        default="./config/robot.yaml")
    args = parser.parse_args()
    print("robot name: ", args.robot)
    rclpy.init()
    pd_controller_node = PDControllerNode(robot_name=args.robot, config_path=args.config)
    try:
        rclpy.spin(pd_controller_node)
    except KeyboardInterrupt:
        pass
    finally:
        pd_controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import argparse
import math
import threading
from typing import List, Optional, Tuple
import numpy as np
import cv2

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Empty
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R
from sympy.codegen.ast import continue_

from planning_utils import start_to_current
from io_utils import load_calibration, overlay_path
"""
PathManagerNode from CHOP:
https://github.com/gershom96/CHOP/blob/main/deployment/path_manager.py
The path manager maintains the latest predicted waypoint sequence as the active path. 
When a new path is received, the robot pose at the time of prediction is stored so that
waypoints can be transformed consistently from the local planning frame into the current 
odometry frame. During execution, the path manager removes waypoints that hve already been
 reached or have fallen behind the robot, then publishes the next valid waypoint as the 
 current navigation target.
inputs:
- /path topic type Path of any trajectories from your policies
- /started type Empty when the path has started. 
- /req_goal the planner requests next waypoint once the current one is reached.
- odom from your robot
- camera input from image topic, for runtime visual overlay
outputs:
- /next_goal, the next waypoint on the current trajectory
- /active_path, of tyep Path, current path that is being executed.
"""
class PathManagerNode(Node):
    def __init__(self, config_path: str, robot_name: str, visualize: bool=False):
        super().__init__("path_manager")

        self.qos_profile  = QoSProfile(
                        reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST,  
                        depth=15  
                    )
        
        self.qos_profile_r  = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,  
                depth=15  
            )
        self.robot_name = robot_name
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        robot_config = config[robot_name]
        self.odom_topic = robot_config['odom_topic']
        self.image_topic = robot_config['image_topic']
        # ---- Params ----
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("behind_margin", 0.09)    # pop if x < -margin
        self.declare_parameter("reach_radius", 0.1)     # pop if dist < radius (extra robustness)
        self.declare_parameter("overlay_enabled", visualize)
        self.declare_parameter("overlay_topic", "/path_overlay")

        self.base_frame = self.get_parameter("base_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.behind_margin = float(self.get_parameter("behind_margin").value)
        self.reach_radius = float(self.get_parameter("reach_radius").value)
        self.overlay_enabled = bool(self.get_parameter("overlay_enabled").value)
        self.overlay_topic = self.get_parameter("overlay_topic").value

        self.cam_matrix, self.dist_coeffs, self.T_base_from_cam = load_calibration(config['cam_matrix'])
        self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)

        # ---- State ----
        self._lock = threading.Lock()

        # latest homogeneous from base to world
        self._current_T_w : Optional[np.ndarray] = None
        # snapshot homogeneous from base to world
        self._start_T_w: Optional[np.ndarray] = None

        # path points stored in START robot frame (what model outputs)
        self._path_start_xy: np.ndarray = np.empty((0, 2))
        self._pts_w: np.ndarray = np.empty((0, 2))
        self._image: Optional[np.ndarray] = None

        # ---- ROS I/O ----
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, self.qos_profile)
        self.create_subscription(Empty, "/started", self.on_started, self.qos_profile)
        self.create_subscription(Path, "/path", self.on_path, self.qos_profile)
        self.create_subscription(Empty, "/req_goal", self.on_req_goal, self.qos_profile)

        if self.overlay_enabled:
            self.bridge = CvBridge()
            self.create_subscription(CompressedImage, self.image_topic, self.on_image, 10)
            self.pub_overlay = self.create_publisher(Image, self.overlay_topic, 10)

        self.pub_next_goal = self.create_publisher(PoseStamped, "/next_goal", 10)
        self.pub_active_path = self.create_publisher(Path, "/active_path", 10)

        self.get_logger().info(
            f"PathManagerNode running. base_frame={self.base_frame}, "
            f"behind_margin={self.behind_margin}, reach_radius={self.reach_radius}"
        )

    # ---------------- callbacks ----------------

    def on_odom(self, msg: Odometry):
        # gets transformation matrix from world to base as current_T_w.
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        c, s = np.cos(yaw), np.sin(yaw)
        with self._lock:
            self._current_T_w = np.eye(3)
            self._current_T_w[:2, :2] = [[c, -s], [s, c]]
            self._current_T_w[:2, 2]  = [p.x, p.y]

    def on_started(self, _msg: Empty):
        with self._lock:
            if self._current_T_w is None:
                self.get_logger().warn("Got /started but no odom yet; cannot snapshot start pose.")
                return
            self._start_T_w = self._current_T_w.copy()
        self.get_logger().info("Saved start odom pose from /started.")

    def on_path(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn("Received empty /path.")
            with self._lock:
                self._path_start_xy = np.empty((0, 2))
            return

        with self._lock:
            if self._start_T_w is None:
                self.get_logger().warn("Received /path but no /started pose saved; ignoring.")
                return
            # store points in start frame (model frame)
            self._path_start_xy = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses])

        # Rebase to current frame + drop behind + publish
        self._drop_behind_and_publish()

    def on_req_goal(self, _msg: Empty):
        # Planner says it reached the current goal -> pop front and publish next
        with self._lock:
            if self._path_start_xy.size != 0:
                self._path_start_xy = self._path_start_xy[1:]
        self._drop_behind_and_publish()

    def on_image(self, msg: CompressedImage):
        # publishes image overlay for visualization
        if not self.overlay_enabled:
            return
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._image = img.copy()
            if self._pts_w is not None and self._current_T_w is not None:
                if len(self._pts_w) > 0:
                    _pts_w = self._pts_w.copy()
                    T_c = self._current_T_w.copy()
                else:
                    T_c = None
                    _pts_w = None
            else:
                T_c = None
                _pts_w = None
        if _pts_w is not None and T_c is not None:
            pts_w_h = np.vstack([_pts_w.T, np.ones(_pts_w.shape[0])])  # (3,N)
            pts_cur = (np.linalg.inv(T_c) @ pts_w_h)[:2, :].T  # (N,2)
            overlay_img = overlay_path(pts_cur, img, self.cam_matrix, self.T_cam_from_base)
            if overlay_img is not None:
                out_msg = self.bridge.cv2_to_imgmsg(overlay_img, encoding="bgr8")
                out_msg.header.stamp = msg.header.stamp
                out_msg.header.frame_id = self.world_frame
                self.pub_overlay.publish(out_msg)
            else:
                out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                out_msg.header.stamp = msg.header.stamp
                out_msg.header.frame_id = self.world_frame
                self.pub_overlay.publish(out_msg)

    
    # ---------------- core logic ----------------

    def _drop_behind_and_publish(self):
        with self._lock:
            if self._path_start_xy.size == 0:
                return
            T_s = self._start_T_w
            T_c = self._current_T_w
            pts_start = self._path_start_xy.copy()

        if T_s is None or T_c is None:
            return
        
        pts_cur = start_to_current(T_s, T_c, pts_start)  # (N,2)

        dists = np.linalg.norm(pts_cur, axis=1)
        behind = pts_cur[:, 0] < -self.behind_margin
        reached = dists < self.reach_radius
        keep = ~(behind | reached)

        with self._lock:
            self._path_start_xy = self._path_start_xy[keep, :]
        self.get_logger().info(f"keeping {len(keep)} out of {len(pts_cur)} points.")
        pts_cur = pts_cur[keep, :]

        # current frame -> world frame
        pts_w = []
        T_w_c = T_c
        pts_c_h = np.vstack([pts_cur.T, np.ones(pts_cur.shape[0])])  # (3,N)
        pts_w   = (T_w_c @ pts_c_h)[:2, :].T  # (N,2)

        self._pts_w = pts_w

        self._publish_if_available(pts_w)
        if self.overlay_enabled and pts_w.size != 0 and self._image is not None:
            overlay_img = overlay_path(pts_cur, self._image, self.cam_matrix, self.T_cam_from_base)
            if overlay_img is not None:
                out_msg = self.bridge.cv2_to_imgmsg(overlay_img, encoding="bgr8")
                out_msg.header.stamp = self.get_clock().now().to_msg()
                out_msg.header.frame_id = self.world_frame
                self.pub_overlay.publish(out_msg)

    def _publish_if_available(self, pts_world: np.ndarray):

        if pts_world.size == 0:
            return
        gx, gy = pts_world[1]

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.world_frame
        goal.pose.position.x = float(gx)
        goal.pose.position.y = float(gy)
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.pub_next_goal.publish(goal)

        path_msg = Path()
        path_msg.header = goal.header
        for x_w, y_w in pts_world:
            ps = PoseStamped()
            ps.header = goal.header
            ps.pose.position.x = float(x_w)
            ps.pose.position.y = float(y_w)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_active_path.publish(path_msg)
        self.get_logger().info(f"getting pt {gx, gy}, path_len {len(pts_world)}")

def main():

    parser = argparse.ArgumentParser(description="Run the Path Manager")
    parser.add_argument("-r", "--robot", type=str, help="Robot Name",
                        default="husky")
    parser.add_argument("--config", type=str, help="yaml config file",
                        # default="./config/robot.yaml")
                        default="/home/gamma-nav/Documents/Projects/git_repos/steernav/steernav/config/robot.yaml")

    args, ros_args = parser.parse_known_args()
    
    rclpy.init()
    node = PathManagerNode(robot_name=args.robot, config_path=args.config)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

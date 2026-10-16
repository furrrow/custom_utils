import json

import numpy as np
import yaml
from typing import Tuple
import argparse
from pathlib import Path

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool, String
from rclpy.qos import QoSProfile
from rclpy.qos import QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

def clip_angle(theta) -> float:
    """Clip angle to [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi

def pd_controller(waypoint: np.ndarray, max_v, max_w, dt, eps=1e-8) -> Tuple[float]:
    """PD controller for the robot"""

    # if waypoint[0] != 0.0:
    #     pdb.set_trace()
    assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
        print(f"dx: {dx}, dy: {dy}")
    # this controller only uses the predicted heading if dx and dy are near zero
    if len(waypoint) == 4 and np.abs(dx) < eps and np.abs(dy) < eps:
        v = 0
        w = clip_angle(np.arctan2(hy, hx)) / dt
    elif np.abs(dx) < eps:
        v = 0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan(dy / dx) / dt
    print(f"before clipping v: {v}, w: {w}")
    v = np.clip(v, 0, max_v)
    w = np.clip(w, -max_w, max_w)
    return v, w

class ROSData:
    def __init__(self, timeout: int = 3, queue_size: int = 1, name: str = ""):
        self.timeout = timeout
        self.last_time_received = None
        self.queue_size = queue_size
        self.data = None
        self.name = name
        self.phantom = False

    def get(self):
        return self.data

    def clear(self):
        self.data = None
        self.last_time_received = None

    def set(self, data):
        current_time = rclpy.clock.Clock().now()

        time_waited = float('inf')
        if self.last_time_received is not None:
            # Update time_waited with the actual time waited, otherwise stick to a huge value. This resets the data
            time_waited = (current_time - Time(seconds=self.last_time_received)).nanoseconds / 1e9  # Convert to seconds

        if self.queue_size == 1:
            self.data = data
        else:
            if self.data is None or time_waited > self.timeout:  # Reset queue if timeout
                self.data = []
            if len(self.data) == self.queue_size:
                self.data.pop(0)
            self.data.append(data)
        self.last_time_received = current_time.nanoseconds / 1e9  # Store time in seconds

    def is_valid(self, verbose: bool = False):
        current_time = rclpy.clock.Clock().now()

        time_waited = float('inf')
        if self.last_time_received is not None:
            # Update time_waited with the actual time waited, otherwise stick to a huge value. This resets the data
            time_waited = (current_time - Time(seconds=self.last_time_received)).nanoseconds / 1e9  # Convert to seconds

        valid = time_waited < self.timeout
        if self.queue_size > 1:
            valid = valid and len(self.data) == self.queue_size
        if verbose and not valid:
            print(f"Not receiving {self.name} data for {time_waited:.2f} seconds (timeout: {self.timeout} seconds)")
        return valid

class PDControllerNode(Node):
    def __init__(self, config_path: str, robot_name: str, default_dt:float=0.0,
                 steer:bool = False, simulation_episode_control: bool = False):
        super().__init__("pd_controller")
        WAYPOINT_TIMEOUT = 1  # seconds
        self.vel_msg = Twist()
        self.waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
        self.reached_goal = False
        self.robot_name = robot_name
        self.simulation_episode_control = simulation_episode_control
        self.current_episode_id = None
        self.ready_for_motion = not simulation_episode_control
        with open(config_path, "r") as f:
            deploy_config = yaml.safe_load(f)
        self.rate = deploy_config["controller_rate"]
        self.waypoint_idx = deploy_config['waypoint_idx']
        robot_config = deploy_config[robot_name]
        print(f"using robot config for: {robot_name}")
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]
        print("maxv:", self.max_v, "maxw", self.max_w)

        # ROS Topics
        IMAGE_TOPIC = robot_config['image_topic']
        print(f"IMAGE_TOPIC: {IMAGE_TOPIC}")
        if steer:
            WAYPOINT_TOPIC = robot_config['steered_waypoint_topic']
        else:
            WAYPOINT_TOPIC = robot_config['waypoint_topic']
        REACHED_GOAL_TOPIC = robot_config['reached_goal_topic']
        VEL_TOPIC = robot_config['vel_topic']
        print("VEL_TOPIC", VEL_TOPIC)
        print("subscribing to WAYPOINT_TOPIC", WAYPOINT_TOPIC)
        if default_dt > 0:  # how long is the controller expected to reach next waypoint
            self.dt = default_dt
        else:
            self.dt = 1 / self.rate
        self.get_logger().info(f"setting dt value of controller to {self.dt}")

        self.waypoint_sub = self.create_subscription(Float32MultiArray, 
                                                     WAYPOINT_TOPIC,
                                                     self.callback_drive, 
                                                     qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                                                              history=QoSHistoryPolicy.KEEP_LAST,
                                                                              depth=10))
        self.reached_goal_sub = self.create_subscription(Bool, 
                                                         REACHED_GOAL_TOPIC, 
                                                         self.callback_reached_goal, 
                                                         10)
        self.vel_out = self.create_publisher(Twist, 
                                             VEL_TOPIC,
                                             10)

        self.episode_sub = None
        if self.simulation_episode_control:
            reliable_transient_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.episode_sub = self.create_subscription(
                String,
                "/isaacsim/episode",
                self.callback_episode,
                qos_profile=reliable_transient_qos,
            )
            self.publish_zero("waiting for a simulation episode")

        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)
        self.get_logger().info("Registered with master node. Waiting for waypoints...")

    def callback_drive(self, waypoint_msg: Float32MultiArray):
        """Callback function for the waypoint subscriber"""
        if self.simulation_episode_control:
            if self.current_episode_id is None:
                self.get_logger().warning("Ignoring waypoints while waiting for simulation episode start")
                return
            else:
                self.ready_for_motion = True
        self.get_logger().info("Setting waypoint")
        self.waypoint.set(waypoint_msg.data)

    def callback_episode(self, episode_msg: String):
        """Reset cached control state when a new simulation episode starts."""
        try:
            episode = json.loads(episode_msg.data)
            episode_id = episode["episode_id"]
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("episode_id must be a nonempty string")
            if episode.get("state") != "started":
                return
            if episode.get("robot_type") != "nova_carter":
                raise ValueError(f"unexpected robot type {episode.get('robot_type')} only supporting nova_carter")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"callback_episode error: {exc}")
            return
        if episode_id != self.current_episode_id:
            self.current_episode_id = episode_id
            self.reached_goal = False
            self.waypoint.clear()
            self.ready_for_motion = False
            self.publish_zero(f"reset for new episode {episode_id}")

    def publish_zero(self, reason: str = ""):
        """Clear and publish zero vel commands"""
        self.vel_msg = Twist()
        self.vel_out.publish(self.vel_msg)
        if reason:
            self.get_logger().error(f"publish_zero error: {reason}")

    def callback_reached_goal(self, reached_goal_msg: Bool):
        """Callback function for the reached goal subscriber"""
        self.reached_goal = reached_goal_msg.data

    def timer_callback(self):
        if self.reached_goal:
            self.vel_out.publish(self.vel_msg)
            self.get_logger().info("Reached goal! Stopping...")
            rclpy.shutdown()
            return

        if self.simulation_episode_control and not self.ready_for_motion:
            self.publish_zero()
            return

        if self.waypoint.is_valid(verbose=True):
            v, w = pd_controller(self.waypoint.get(), self.max_v, self.max_w, self.dt, eps=1e-8)
            self.vel_msg.linear.x = v
            self.vel_msg.angular.z = w
            self.get_logger().info(f"Publishing new velocity: {v}, {w}")
        else:
            self.vel_msg = Twist() # no velocity
            self.waypoint.clear()
        self.vel_out.publish(self.vel_msg)

def main():
    parser = argparse.ArgumentParser(description="Run the PD Controller")
    parser.add_argument("-r", "--robot", type=str, help="Robot Name",
                        default="husky")
    parser.add_argument("--config", type=str, help="yaml config file",
                        default=str(Path(__file__).resolve().parents[2] / "configs/robot.yaml"))
    parser.add_argument("--steer", action="store_true",
                        help="listen to steered waypoint topic")
    parser.add_argument("--dt", type=float, help="dt for pd controller",
                        default=1.5)
    parser.add_argument(
        "--simulation-episode-control",
        action="store_true",
        help="enable episode reset and safe-zero behavior for the persistent simulation controller",
    )
    args = parser.parse_args()
    print("robot name: ", args.robot)
    rclpy.init()
    pd_controller_node = PDControllerNode(
        robot_name=args.robot,
        config_path=args.config,
        default_dt=args.dt,
        steer=args.steer,
        simulation_episode_control=args.simulation_episode_control,
    )
    try:
        rclpy.spin(pd_controller_node)
    except KeyboardInterrupt:
        pass
    finally:
        pd_controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

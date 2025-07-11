import math
from collections import defaultdict, deque
import rclpy
from rclpy.node import Node
import torch
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from tidup_move_base_msgs.msg import (
    AgentStatesPrediction,
    AgentStatePrediction,
    PoseWith2DCovariance,
)
from rclpy.time import Time as RclpyTime
from geometry_msgs.msg import Twist


class HumanTrajPredictor(Node):
    def __init__(self):
        super().__init__("san_navistar_node")

        self.max_human_num_ = 5

        self.agents_data_ = None
        self.odom_data_ = None

        # ! SUBSCRIBERS
        self.agent_states_sub_ = self.create_subscription(
            AgentStates,
            "/pedsim_simulator/simulated_agents",
            self.agent_states_callback,
            1,
        )

        self.odom_sub_ = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            1,
        )

        # ! PUBLISHERS
        self.cmd_vel_pub_ = self.create_publisher(Twist, "cmd_vel", 10)

        # ! DATA HISTORY FOR MODEL

        self.command_timer = self.create_timer(0.1, self.command_callback)

    def command_callback(self):

        pass

    def agent_states_callback(self, msg: AgentStates):
        self.max_human_num_ = len(msg.agent_states)
        self.agents_data_ = msg.agent_states

    def odom_callback(self, msg: Odometry):
        self.odom_data_ = msg


def main(args=None):
    rclpy.init(args=args)
    node = HumanTrajPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

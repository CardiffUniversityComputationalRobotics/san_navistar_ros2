import math
import rclpy
from rclpy.node import Node
import torch
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
import torch
import torch.nn as nn
import os
from san_navistar_ros2.models.model import Policy
import numpy as np
from san_navistar_ros2.data.navigation.star.configs.config import Config
from tf_transformations import euler_from_quaternion
from pedsim_msgs.msg import AgentStates, AgentState
from esc_move_base_msgs.msg import Path2D
import gym
from launch_ros.substitutions import FindPackageShare
from san_navistar_ros2.utils import check_reverse


class SANNaviStarNode(Node):
    def __init__(self):
        super().__init__("san_navistar_node")

        self.san_navistar_ros2_dir = FindPackageShare("san_navistar_ros2").find(
            "san_navistar_ros2"
        )

        # ! PARAMS DECLARATION
        self.declare_parameter("v_pref", 0.4)
        self.declare_parameter("robot_radius", 0.3)
        self.declare_parameter("human_radius", 0.3)
        self.declare_parameter("robot_fov", 3.14)
        self.declare_parameter("human_fov", 3.14)
        self.declare_parameter("visible_dis", 10.0)
        self.declare_parameter("command_timer_period", 0.25)

        # topics
        self.declare_parameter("goal_path_topic", "/solution_path")
        self.declare_parameter("social_agents_topic", "/pedsim_simulator/simulated_agents")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.max_human_num_ = 5

        self.agents_data_ = None
        self.odom_data_ = None

        # ! ROBOT PARAMS
        self.robot_radius_ = self.get_parameter("robot_radius").value
        self.human_radius_ = self.get_parameter("human_radius").value
        self.robot_fov_ = self.get_parameter("robot_fov").value
        self.human_fov_ = self.get_parameter("human_fov").value
        self.v_pref_ = self.get_parameter("v_pref").value

        # goal
        self.global_goal_x_ = 0.0
        self.global_goal_y_ = 0.0

        self.local_goal_x_ = 0.0
        self.local_goal_y_ = 0.0

        self.goal_available_ = False

        self.visible_dis = self.get_parameter("visible_dis").value

        self.waypoints = []

        self.actor_critic_init_ = False
        self.actor_critic = None

        high = np.inf * np.ones(
            [
                2,
            ]
        )
        self.action_space_ = gym.spaces.Box(-high, high, dtype=np.float32)

        # topic configs
        self.global_plan_topic = self.get_parameter("goal_path_topic").value
        self.agent_states_topic = self.get_parameter("social_agents_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        # ! SUBSCRIBERS
        self.agent_states_sub_ = self.create_subscription(
            AgentStates,
            self.agent_states_topic,
            self.agent_states_callback,
            1,
        )

        self.odom_sub_ = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            1,
        )

        self.goal_subs = self.create_subscription(
            PoseStamped, "/goal_pose", self.global_goal_callback, 5
        )

        self.global_plan_sub = self.create_subscription(
            Path2D, self.global_plan_topic, self.global_plan_callback, 10
        )

        # ! PUBLISHERS
        self.cmd_vel_pub_ = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # ! INITIALISE MODEL

        model_dir = "data/navigation/star"
        test_model = "00500.pt"

        self.config = Config()

        torch.manual_seed(self.config.env.seed)
        torch.cuda.manual_seed_all(self.config.env.seed)
        if self.config.training.cuda:
            if self.config.training.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            else:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

        torch.set_num_threads(1)

        self.load_path = os.path.join(model_dir, "checkpoints", test_model)
        print("load path is:", self.load_path)

        # ! TIMERS
        self.command_timer_period_ = self.get_parameter("command_timer_period").value
        self.command_timer = self.create_timer(
            self.command_timer_period_, self.command_callback
        )

    def agent_states_callback(self, msg: AgentStates):
        self.max_human_num_ = len(msg.agent_states)
        self.agents_data_ = msg.agent_states

    def odom_callback(self, msg: Odometry):
        self.odom_data_ = msg

    def global_goal_callback(self, msg: PoseStamped):
        self.global_goal_x_ = msg.pose.position.x
        self.global_goal_y_ = msg.pose.position.y
        self.goal_available_ = True

    def global_plan_callback(self, msg: Path2D):
        self.waypoints = []

        if len(msg.waypoints) > 1:

            got_waypoints = msg.waypoints

            got_waypoints.reverse()

            for pos in got_waypoints:
                if (
                    math.sqrt(
                        (pos.x - self.odom_data_.pose.pose.position.x) ** 2
                        + (pos.y - self.odom_data_.pose.pose.position.y) ** 2
                    )
                    > 0.2
                ):
                    self.waypoints.append([pos.x, pos.y])
                else:
                    break

            self.waypoints.reverse()
        self.local_goal_x_ = self.waypoints[0][0]
        self.local_goal_y_ = self.waypoints[0][1]

        self.global_goal_x_ = self.waypoints[-1][0]
        self.global_goal_y_ = self.waypoints[-1][1]

        self.goal_available_ = True

    def generate_ob(self, reset):
        ob = {}

        orientation_list = [
            self.odom_data_.pose.pose.orientation.x,
            self.odom_data_.pose.pose.orientation.y,
            self.odom_data_.pose.pose.orientation.z,
            self.odom_data_.pose.pose.orientation.w,
        ]

        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        # nodes
        visible_humans, num_visibles, human_visibility = self.get_num_human_in_fov()

        self.update_last_human_states(human_visibility, reset=reset)

        ob["robot_node"] = np.array(
            [
                [
                    [
                        self.odom_data_.pose.pose.position.x,
                        self.odom_data_.pose.pose.position.y,
                        self.robot_radius_,
                        self.local_goal_x_,
                        self.local_goal_y_,
                        self.v_pref_,
                        yaw,
                    ]
                ]
            ]
        )
        # edges
        # temporal edge: robot's velocity
        temporal_edges = np.array(
            [self.odom_data_.twist.twist.linear.x, self.odom_data_.twist.twist.linear.y]
        ).reshape(1, -1)
        ob["temporal_edges"] = np.expand_dims(temporal_edges, axis=0)
        # spatial edges: the vector pointing from the robot position to each human's position
        spatial_edges = np.zeros((self.max_human_num_, 2))

        for i in range(self.max_human_num_):
            relative_pos = np.array(
                [
                    self.last_human_states_[i, 0]
                    - self.odom_data_.pose.pose.position.x,
                    self.last_human_states_[i, 1]
                    - self.odom_data_.pose.pose.position.y,
                ]
            )
            spatial_edges[i] = relative_pos
        ob["spatial_edges"] = np.array(np.expand_dims(spatial_edges, axis=0))
        flatten_obs = np.concatenate(
            (
                spatial_edges.reshape(
                    -1,
                ),
                [
                    self.odom_data_.pose.pose.position.x,
                    self.odom_data_.pose.pose.position.y,
                    self.robot_radius_,
                    self.local_goal_x_,
                    self.local_goal_y_,
                    self.v_pref_,
                    yaw,
                ],
            ),
            axis=-1,
        )
        ob["flatten_obs"] = np.expand_dims(flatten_obs, axis=0)

        goal_relative_pos = np.array(
            [
                self.local_goal_x_ - self.odom_data_.pose.pose.position.x,
                self.local_goal_y_ - self.odom_data_.pose.pose.position.y,
                1,
                0,
            ]
        ).reshape(1, -1)
        one_hot = np.array([0, 1] * self.max_human_num_).reshape(
            self.max_human_num_, -1
        )
        spatial_edges = np.concatenate((spatial_edges, one_hot), axis=-1)
        spatial_edges = np.concatenate((spatial_edges, goal_relative_pos), axis=0)
        dis = np.linalg.norm(spatial_edges, axis=-1)
        visible_mask = (dis < self.visible_dis).astype(int)
        visible_mask[-1] = 1

        ob["spatial_edges_transformer"] = np.expand_dims(spatial_edges, axis=0)
        ob["visible_masks"] = np.expand_dims(visible_mask, axis=0)
        robot_pos = np.array(
            [
                self.odom_data_.pose.pose.position.x,
                self.odom_data_.pose.pose.position.y,
                1,
                0,
            ]
        ).reshape(1, -1)
        ob["robot_pos"] = np.expand_dims(robot_pos, axis=0)

        return ob

    def update_last_human_states(self, human_visibility, reset):
        """
        update the self.last_human_states array
        human_visibility: list of booleans returned by get_human_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        # keep the order of 5 humans at each timestep
        for i in range(self.max_human_num_):
            if human_visibility[i]:
                # humanS = np.array(self.humans[i].get_observable_state_list())
                humanS = np.array(
                    [
                        self.agents_data_[i].pose.position.x,
                        self.agents_data_[i].pose.position.y,
                        self.agents_data_[i].twist.linear.x,
                        self.agents_data_[i].twist.linear.y,
                        self.human_radius_,
                    ]
                )
                self.last_human_states_[i, :] = humanS

            else:
                if reset:
                    humanS = np.array([15.0, 15.0, 0.0, 0.0, 0.3])
                    self.last_human_states_[i, :] = humanS

                else:
                    px, py, vx, vy, theta = self.last_human_states_[i, :]
                    # Plan A: linear approximation of human's next position
                    px = px + vx * self.command_timer_period_
                    py = py + vy * self.command_timer_period_
                    self.last_human_states_[i, :] = np.array([px, py, vx, vy, theta])

                    # Plan B: assume the human doesn't move, use last observation
                    # self.last_human_states[i, :] = np.array([px, py, 0., 0., r])

    def get_num_human_in_fov(self):
        human_ids = []
        humans_in_view = []
        num_humans_in_view = 0

        for i in range(self.max_human_num_):
            visible = self.detect_visible(self.agents_data_[i], robot1=True)
            if visible:
                humans_in_view.append(self.agents_data_[i])
                num_humans_in_view = num_humans_in_view + 1
                human_ids.append(True)
            else:
                human_ids.append(False)

        return humans_in_view, num_humans_in_view, human_ids

    def detect_visible(self, state2: AgentState, robot1=False, custom_fov=None):

        orientation_list = [
            self.odom_data_.pose.pose.orientation.x,
            self.odom_data_.pose.pose.orientation.y,
            self.odom_data_.pose.pose.orientation.z,
            self.odom_data_.pose.pose.orientation.w,
        ]

        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        real_theta = yaw

        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [
            state2.pose.position.x - self.odom_data_.pose.pose.position.x,
            state2.pose.position.y - self.odom_data_.pose.pose.position.y,
        ]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        else:
            if robot1:
                fov = self.robot_fov_
            else:
                fov = self.human_fov_

        return np.abs(offset) <= fov / 2

    def command_callback(self):

        if self.goal_available_ and len(self.waypoints) > 0:
            if not self.actor_critic_init_:
                device = torch.device("cuda" if self.config.training.cuda else "cpu")
                self.last_human_states_ = np.zeros((self.max_human_num_, 5))
                self.actor_critic = Policy(
                    self.generate_ob(True),
                    self.action_space_,
                    base_kwargs=self.config,
                    base=self.config.robot.policy,
                    device=device,
                )
                self.actor_critic.load_state_dict(
                    torch.load(
                        self.san_navistar_ros2_dir
                        + "/san_navistar_ros2/data/navigation/star/checkpoints/00500.pt",
                        map_location=device,
                    )
                )
                self.actor_critic.base.nenv = self.config.testing.num_processes

                nn.DataParallel(self.actor_critic).to(device)
                self.actor_critic_init_ = True
                print("defined actor critic")

            recurrent_cell = "GRU"
            double_rnn_size = 2 if recurrent_cell == "LSTM" else 1

            eval_recurrent_hidden_states = {}
            eval_recurrent_hidden_states["human_node_rnn"] = np.zeros(
                (
                    self.config.testing.num_processes,
                    1,
                    self.config.SRNN.human_node_rnn_size * double_rnn_size,
                )
            )
            eval_recurrent_hidden_states["human_human_edge_rnn"] = np.zeros(
                (
                    self.config.testing.num_processes,
                    self.config.sim.human_num + 1,
                    self.config.SRNN.human_human_edge_rnn_size * double_rnn_size,
                )
            )
            eval_masks = np.zeros((self.config.testing.num_processes, 1))

            obs = self.generate_ob(False)

            _, action, _, eval_recurrent_hidden_states = self.actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
            )

            action = check_reverse(action)

            cmd_vel = Twist()
            cmd_vel.linear.x = float(action[0][0])
            cmd_vel.angular.z = float(action[0][1])

            self.cmd_vel_pub_.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    node = SANNaviStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

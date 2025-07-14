import math
import rclpy
from rclpy.node import Node
import torch
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
from geometry_msgs.msg import Twist
import torch
import torch.nn as nn
import os
from san_navistar_ros2.models.model import Policy
import numpy as np


class SANNaviStarNode(Node):
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

        # ! INITIALISE MODEL

        model_dir = "data/navigation/star"
        test_model = "00500.pt"

        # from importlib import import_module

        # if model_dir.endswith("/"):
        #     model_dir = model_dir[:-1]
        #     # try:
        # model_dir_string = model_dir.replace("/", ".") + ".configs.config"
        # print(model_dir_string)
        # model_arguments = import_module(model_dir_string)
        # Config = getattr(model_arguments, "Config")
        # # except:
        #     print(
        #         "Failed to get Config function from ", test_args.model_dir, "/config.py"
        #     )
        #     from crowd_nav.configs.config import Config

        from san_navistar_ros2.data.navigation.star.configs.config import Config

        config = Config()

        log_file = os.path.join(model_dir, "test")
        if not os.path.exists(log_file):
            os.mkdir(log_file)

        torch.manual_seed(config.env.seed)
        torch.cuda.manual_seed_all(config.env.seed)
        if config.training.cuda:
            if config.training.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            else:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

        torch.set_num_threads(1)
        device = torch.device("cuda" if config.training.cuda else "cpu")

        load_path = os.path.join(model_dir, "checkpoints", test_model)
        print("load path is:", load_path)

        eval_dir = os.path.join(model_dir, "eval")
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)

        # actor_critic = Policy(
        #     envs.observation_space.spaces,
        #     envs.action_space,
        #     base_kwargs=config,
        #     base=config.robot.policy,
        #     device=device,
        # )

        # ! TIMERS
        self.command_timer = self.create_timer(0.1, self.command_callback)

        print("ready")

        self.last_human_states_ = np.zeros((self.max_human_num_, 5))

    def agent_states_callback(self, msg: AgentStates):
        self.max_human_num_ = len(msg.agent_states)
        self.agents_data_ = msg.agent_states

    def odom_callback(self, msg: Odometry):
        self.odom_data_ = msg

    def generate_ob(self, reset):
        ob = {}

        # nodes
        visible_humans, num_visibles, human_visibility = self.get_num_human_in_fov()

        self.update_last_human_states(human_visibility, reset=reset)

        ob["robot_node"] = [self.robot.get_full_state_list_noV()]
        # edges
        # temporal edge: robot's velocity
        ob["temporal_edges"] = np.array([self.robot.vx, self.robot.vy]).reshape(1, -1)
        # spatial edges: the vector pointing from the robot position to each human's position
        spatial_edges = np.zeros((self.human_num, 2))

        for i in range(self.human_num):
            relative_pos = np.array(
                [
                    self.last_human_states[i, 0] - self.robot.px,
                    self.last_human_states[i, 1] - self.robot.py,
                ]
            )
            spatial_edges[i] = relative_pos
        ob["spatial_edges"] = spatial_edges
        ob["flatten_obs"] = np.concatenate(
            (
                spatial_edges.reshape(
                    -1,
                ),
                self.robot.get_full_state_list(),
            ),
            axis=-1,
        )

        goal_relative_pos = np.array(
            [self.robot.gx - self.robot.px, self.robot.gy - self.robot.py, 1, 0]
        ).reshape(1, -1)
        one_hot = np.array([0, 1] * self.max_human_num_).reshape(
            self.max_human_num_, -1
        )
        spatial_edges = np.concatenate((spatial_edges, one_hot), axis=-1)
        spatial_edges = np.concatenate((spatial_edges, goal_relative_pos), axis=0)
        dis = np.linalg.norm(spatial_edges, axis=-1)
        visible_mask = (dis < self.visible_dis).astype(int)
        visible_mask[-1] = 1

        ob["spatial_edges_transformer"] = spatial_edges
        ob["visible_masks"] = visible_mask
        ob["robot_pos"] = np.array([self.robot.px, self.robot.py, 1, 0]).reshape(1, -1)

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
                humanS = np.array(self.humans[i].get_observable_state_list())
                self.last_human_states[i, :] = humanS

            else:
                if reset:
                    humanS = np.array([15.0, 15.0, 0.0, 0.0, 0.3])
                    self.last_human_states[i, :] = humanS

                else:
                    px, py, vx, vy, theta = self.last_human_states[i, :]
                    # Plan A: linear approximation of human's next position
                    px = px + vx * self.time_step
                    py = py + vy * self.time_step
                    self.last_human_states[i, :] = np.array([px, py, vx, vy, theta])

                    # Plan B: assume the human doesn't move, use last observation
                    # self.last_human_states[i, :] = np.array([px, py, 0., 0., r])

    def command_callback(self):

        pass


def main(args=None):
    rclpy.init(args=args)
    node = SANNaviStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

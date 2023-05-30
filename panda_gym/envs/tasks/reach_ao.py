import itertools
import json
import logging
import random
import sys
from math import exp
from pathlib import Path
from typing import Any, Dict

import gymnasium
import numpy as np
import pybullet
import pybullet as p
from pyb_utils.collision import NamedCollisionObject, CollisionDetector

from panda_gym.envs.core import Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.utils import distance

sys.modules["gym"] = gymnasium

ROOT_DIR = Path(__file__).parent.parent.parent
SCENARIO_DIR = f"{ROOT_DIR}\\assets\\scenarios"
NARROW_TUNNEL_DIR = f"{SCENARIO_DIR}\\narrow_tunnel"
LIBRARY_DIR = f"{SCENARIO_DIR}\\library"
WORKSHOP_DIR = f"{SCENARIO_DIR}\\workshop"
KASYS_DIR = f"{SCENARIO_DIR}\\kasys"


class ReachAO(Task):
    def __init__(
            self,
            sim,
            robot,
            get_ee_position,
            reward_type="sparse",
            goal_distance_threshold=0.05,
            show_goal_space=False,
            joint_obstacle_observation="all",
            scenario="cube_3",
            randomize_robot_pose=False,
            truncate_episode_on_collision=True,
            show_debug_labels=True,
            fixed_target=None,
            # adjustment factors for dense rewards
            factor_punish_distance=10e-3,  # def: 10.0
            factor_punish_collision=1.0,  # def: 1.0
            factor_punish_action_magnitude=0.000,  # def: 0.005
            factor_punish_action_difference_magnitude=0.0,  # def: ?
            factor_punish_obstacle_proximity=0.0,  # def: ?
            collision_reward=-100

    ) -> None:
        super().__init__(sim)

        self.sim_id = self.sim.physics_client._client
        # if self.sim.dummy_collision_client is not None:
        #     self.dummy_sim_id = self.sim.dummy_collision_client._client

        self.robot: Panda = robot
        self.obstacles = {}
        # self.dummy_obstacles = {}
        # self.dummy_obstacle_id = {}
        self.joint_obstacle_observation = joint_obstacle_observation

        # general reward configuration
        self.collision_reward = collision_reward

        # dense reward configuration
        self.factor_punish_distance = factor_punish_distance
        self.factor_punish_collision = factor_punish_collision
        self.factor_punish_effort = factor_punish_action_magnitude
        self.factor_punish_action_difference_magnitude = factor_punish_action_difference_magnitude
        self.factor_punish_obstacle_proximity = factor_punish_obstacle_proximity

        # if target is fixed, it won't be randomly sampled on each episode
        self.fixed_target = fixed_target

        self.scenario = scenario
        self.randomize_robot_pose = randomize_robot_pose
        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(self.sample_inside_torus)#lambda: self.set_robot_random_joint_position_ik_sphere(0.45, 0.5)

        # self.robot_params = self.create_robot_debug_params()
        self.cube_size_large = np.array([0.05, 0.05, 0.05])
        self.cube_size_medium = np.array([0.03, 0.03, 0.03])
        self.cube_size_small = np.array([0.02, 0.02, 0.02])
        self.cube_size_mini = np.array([0.01, 0.01, 0.01])

        self.reward_type = reward_type
        self.distance_threshold = goal_distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range = 0.3
        self.obstacle_count = 0

        self._sample_goal = self._sample_obstacle = self.sample_from_goal_range
        # set default goal range
        self.x_offset = 0.6
        self.goal_range_low = np.array([-self.goal_range / 2.5 + self.x_offset, -self.goal_range / 1.5, 0])
        self.goal_range_high = np.array([self.goal_range / 2.5 + self.x_offset, self.goal_range / 1.5, self.goal_range])

        self.truncate_episode_on_collision = truncate_episode_on_collision
        if not self.truncate_episode_on_collision:
            def no_truncation():
                pass

            self.is_truncated = no_truncation

        # # init pose generator env
        # self.reach_checker_env = gym.make("PandaReachChecker", control_type = "js", render=False)

        # # load pose checker model
        # path = pathlib.Path(__file__).parent.resolve()
        # self.pose_generator_model = TD3.load(fr"{path}/pose_generator_model.zip",
        #                                      env=self.reach_checker_env)

        self.bodies = {"robot": self.robot.id}

        exclude_links = ["panda_grasptarget", "panda_leftfinger", "panda_rightfinger"]  # env has no grasptarget
        self.collision_links = [i for i in self.robot.link_names if i not in exclude_links]
        self.collision_objects = []

        # set default obstacle mode
        self.randomize_obstacle_position = False  # Randomize obstacle placement
        self.randomize_obstacle_velocity = False
        self.random_num_obs = False
        self.sample_size_obs = [0, 0]

        # extra observations
        self.distances_links_to_closest_obstacle = np.zeros(len(self.collision_links))
        self.is_collided = False
        self.action_magnitude = 0
        self.action_diff = 0
        self.manipulability = 0

        # set scene
        with self.sim.no_rendering():
            self._create_scene()

            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

            if self.scenario:
                self._create_scenario(scenario)()

                if self.randomize_robot_pose and self.robot_pose_randomizer is not None:
                    self.robot.reset = self.robot_pose_randomizer

                # register collision objects for collision detector
                for obstacle_name, obstacle_id in self.obstacles.items():
                    self.collision_objects.append(NamedCollisionObject(obstacle_name))
                    self.bodies[obstacle_name] = obstacle_id

            # set velocity range
            self.velocity_range_low = np.array([-0.2, -0.2, -0.2])
            self.velocity_range_high = np.array([0.2, 0.2, 0.2])

        if show_goal_space:
            self.show_goal_space()

        # add collision detector for robot
        self.named_collision_pairs_rob_obs = []
        self.named_collision_links = []
        for link_name in self.collision_links:
            link = NamedCollisionObject("robot", link_name=link_name)
            self.named_collision_links.append(link)
            for obstacle in self.collision_objects:
                self.named_collision_pairs_rob_obs.append((link, obstacle))
        self.collision_detector = CollisionDetector(col_id=self.sim_id, bodies=self.bodies,
                                                    named_collision_pairs=self.named_collision_pairs_rob_obs)

        # get collision pairs for self collision check of robot, prevent repeating pairs
        combinations = list(itertools.combinations(self.named_collision_links, 2))
        named_collision_pairs_robot = [(a, b) for a, b in combinations]

        self.self_collision_detector = CollisionDetector(col_id=self.sim_id, bodies=self.bodies,
                                                         named_collision_pairs=named_collision_pairs_robot)

        self.show_debug_labels = show_debug_labels
        if self.show_debug_labels:
            self.debug_manip_label_name = "manip"
            self.debug_manip_label_base_text = "Manipulability Score:"
            self.debug_dist_label_name = "dist"
            self.debug_dist_label_base_text = "Distance:"
            self.debug_obs_label_name = "obs"
            self.debug_obs_label_base_text = "Closest Obstacle Distance:"
            self.debug_action_difference_label_name = "action_diff"
            self.debug_action_difference_base_text = "Action Difference:"
            self.debug_effort_label_name = "effort"
            self.debug_effort_base_text = "Effort:"

            self.debug_reward_label_name = "reward"
            self.debug_reward_base_text = "Reward:"

            # self.sim.create_debug_text(self.debug_manip_label_name, f"{self.debug_manip_label_base_text} 0")
            # self.sim.create_debug_text(self.debug_dist_label_name, f"{self.debug_dist_label_base_text} 0")
            # self.sim.create_debug_text(self.debug_obs_label_name, f"{self.debug_obs_label_base_text} 0")
            # self.sim.create_debug_text(self.debug_action_difference_label_name,
            #                            f"{self.debug_action_difference_base_text} 0")
            # self.sim.create_debug_text(self.debug_action_smallness_label_name,
            #                            f"{self.debug_action_smallness_base_text} 0")

    def _create_scenario(self, scenario: str):
        found_scenario = False

        match scenario:
            case scenario if "cube" in scenario.split(sep="_"):
                found_scenario = self.create_scenario_cube
            case scenario if "wang" in scenario.split(sep="_"):
                found_scenario = self.create_scenario_wang
            case scenario if "wangexp" in scenario.split(sep="_"):
                found_scenario = self.create_scenario_wang_experimental
            case "narrow_tunnel":
                found_scenario = self.create_scenario_narrow_tunnel
            case "library":
                found_scenario = self.create_scenario_library
            case "library1":
                found_scenario = self.create_scenario_library
            case "library2":
                found_scenario = self.create_scenario_library
            case "workshop":
                found_scenario = self.create_scenario_workshop
            case "kasys":
                found_scenario = self.create_scenario_kasys
            case "wall":
                found_scenario = self.create_stage_wall
            case "box":
                found_scenario = self.create_stage_box
            case "base1":
                found_scenario = self.create_scenario_base1
            case "base2":
                found_scenario = self.create_scenario_base2
            case "base3":
                found_scenario = self.create_scenario_base3
            case "base3_randshape":
                found_scenario = self.create_scenario_base3_randshape
            case "base4":
                found_scenario = self.create_scenario_base4
            case "base2_5":
                found_scenario = self.create_scenario_base2_5

            case "reach1":
                found_scenario = self.create_scenario_reach1
            case "reach2":
                found_scenario = self.create_scenario_reach2
            case "reach3":
                found_scenario = self.create_scenario_reach3
            case "reach4":
                found_scenario = self.create_scenario_reach4

            case "showcase":
                found_scenario = self.create_showcase

        if not found_scenario:
            logging.warning("Scenario not found. Aborting")
            raise Exception("Scenario not found!")

        return found_scenario

    def _create_scene(self):

        self.sim.create_plane(z_offset=-0.4)
        # self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=0.3)
        self.bodies["table"] = self.sim.create_table(length=3.0, width=3.0, height=0.4, x_offset=0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.8]),
        )

        self.bodies["dummy_sphere"] = self.sim.create_sphere(
            body_name="dummy_sphere",
            radius=0.05,  # dummy target is intentionally bigger to ensure safety distance to obstacle
            mass=0.0,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.0]),
        )

    def create_scenario_narrow_tunnel(self):

        # self.robot.neutral_joint_values = np.array(
        #     [-2.34477029,  1.69617261,  1.81619755, - 1.98816377, - 1.58805049,  1.2963265,
        #      0.41092735]
        #     )

        self.robot.neutral_joint_values = np.array([-1.0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])

        self.robot.set_joint_neutral()
        self.goal_range_low = np.array([0.2, 0.2, 0])
        self.goal_range_high = np.array([0.5, 0.4, 0.6])

        with open(f"{NARROW_TUNNEL_DIR}\\narrow_tunnel.json") as t:
            urdfs = json.load(t)

        # append path
        for urdf in urdfs.values():
            fileName = urdf["fileName"]
            urdf["fileName"] = f"{NARROW_TUNNEL_DIR}\\urdf\\{fileName}"

        indexes = self.sim.load_scenario(urdfs)
        for idx, urdf in zip(indexes, urdfs.values()):
            body_name = urdf["bodyName"]
            name = f"narrow_tunnel_obj_{body_name}"
            self.obstacles[f"{name}"] = idx

    def create_scenario_workshop(self):

        # self.robot.neutral_joint_values = np.array([-2.13728387, - 0.02008786,  0.55133245, - 1.18430162,  0.07185752,  1.35811325,
        #      0.79608991])

        self.robot.neutral_joint_values[0] = -1.5

        self.robot.set_joint_neutral()
        self.goal_range_low = np.array([0.4, -0.15, 0.45])
        self.goal_range_high = np.array([0.7, 0.12, 0.6])
        self.goal = self.fixed_target

        with open(f"{WORKSHOP_DIR}\\workshop.json") as t:
            urdfs = json.load(t)

        # append path
        for urdf in urdfs.values():
            fileName = urdf["fileName"]
            urdf["fileName"] = f"{WORKSHOP_DIR}\\urdf\\{fileName}"

        indexes = self.sim.load_scenario(urdfs)
        for idx, urdf in zip(indexes, urdfs.values()):
            body_name = urdf["bodyName"]
            name = f"workshop_obj_{body_name}"
            self.obstacles[f"{name}"] = idx

    def create_scenario_kasys(self):

        # self.robot.neutral_joint_values = np.array([-2.13728387, - 0.02008786,  0.55133245, - 1.18430162,  0.07185752,  1.35811325,
        #      0.79608991])

        # self.robot.neutral_joint_values[0] = -1.5

        self.robot.set_joint_neutral()
        self.goal_range_low = np.array([1.4, -0.15, 0.45])
        self.goal_range_high = np.array([1.7, 0.12, 0.6])
        self.goal = self.fixed_target

        with open(f"{KASYS_DIR}\\kasys.json") as t:
            urdfs = json.load(t)

        # append path
        for urdf in urdfs.values():
            fileName = urdf["fileName"]
            urdf["fileName"] = f"{KASYS_DIR}\\urdf\\{fileName}"

        indexes = self.sim.load_scenario(urdfs)
        for idx, urdf in zip(indexes, urdfs.values()):
            body_name = urdf["bodyName"]
            name = f"kasys_obj_{body_name}"
            self.obstacles[f"{name}"] = idx

    def create_scenario_library(self):

        self.robot.neutral_joint_values = [0.0, 0.12001979, 0.0, -1.64029458, 0.02081271, 3.1,
                                           0.77979846]  # above table
        # self.fixed_target = np.array([0.7, 0.0, 0.2]) # below table
        # self.goal = self.fixed_target
        with open(f"{LIBRARY_DIR}\\library.json") as t:
            urdfs = json.load(t)

        # append path
        for urdf in urdfs.values():
            fileName = urdf["fileName"]
            urdf["fileName"] = f"{LIBRARY_DIR}\\urdf\\{fileName}"

        indexes = self.sim.load_scenario(urdfs)
        for idx, urdf in zip(indexes, urdfs.values()):
            body_name = urdf["bodyName"]
            name = f"library_obj_{body_name}"
            self.obstacles[f"{name}"] = idx

        # Create custom goal space
        if self.scenario == "library1":
            self.robot.neutral_joint_values = [-2.9595587, -0.51728982, -0.0226481, -2.07870811, 0.05155275, 3.09642384,
                                               0.82259363]
            self.goal_range_low = np.array([0.5, -0.3, 0])
            self.goal_range_high = np.array([0.85, 0.3, 0.3])
        elif self.scenario == "library1.5":
            self.robot.neutral_joint_values = [-2.96090532, -0.0434537, -0.20340835, -1.62954942, 0.02795931,
                                               3.08670391,
                                               0.77425641]
            self.goal_range_low = np.array([0.5, -0.3, 0])
            self.goal_range_high = np.array([0.85, 0.3, 0.3])
        elif self.scenario == "library2":
            self.goal_range_low = np.array([-0.7, -0.4, 0.4])
            self.goal_range_high = np.array([-0.55, 0.4, 0.85])
        else:
            self.goal_range_low = np.array([0.2, -0.3, 0])
            self.goal_range_high = np.array([0.7, 0.3, 0.6])

    def create_stage_3(self):
        """Two big obstacles surrounding end effector, big goal range"""
        self.goal_range = 0.5
        spacing = 2.5
        spacing_x = 5
        x_fix = -0.025
        y_fix = -0.15
        z_fix = 0.15
        for x in range(1):
            for y in range(2):
                for z in range(1):
                    self.create_obstacle_cuboid(
                        np.array([x / spacing_x + x_fix, y / spacing + y_fix, z / spacing + z_fix]),
                        size=np.array([0.05, 0.05, 0.05]))

    def create_stage_shelf_1(self):
        """Shelf."""
        self.goal_range = 0.4

        self.create_obstacle_cuboid(
            np.array([0.2, 0.0, 0.25]),
            size=np.array([0.1, 0.4, 0.001]))

    def create_stage_wall(self):
        """wall parkour."""

        self.goal_range_low = np.array([0.6, -0.6, 0.1])
        self.goal_range_high = np.array([0.7, -0.4, 0.3])

        self.robot.neutral_joint_values = [0.94551719, 0.65262327, 0.12742699, -1.74347465, -0.16996126, 1.97424632,
                                           0.88058222]

        self.create_obstacle_cuboid(
            np.array([0.0, -0.05, 0.1]),
            size=np.array([0.2, 0.05, 0.3]))

        # self.create_obstacle_cuboid(
        #     np.array([0.2, 0.0, 0.25]),
        #     size=np.array([0.1, 0.4, 0.001]))
        # self.create_obstacle_cuboid(
        #     np.array([0.2, 0.0, 0.25]),
        #     size=np.array([0.1, 0.4, 0.001]))

    def create_stage_box(self):
        """box."""
        self.goal_range = 0.4

        self.create_obstacle_cuboid(
            np.array([0.0, 0.26, 0.1]),
            size=np.array([0.18, 0.001, 0.2]))

        self.create_obstacle_cuboid(
            np.array([0.0, -0.26, 0.1]),
            size=np.array([0.18, 0.001, 0.2]))

    def create_scenario_cube(self):
        num_cubes = int(self.scenario.split(sep="_")[1])

        self.goal_range_low = np.array([-0.7, -0.3, 0])
        self.goal_range_high = np.array([0.7, 0.3, 0.6])

        self.randomize_obstacle_position = True
        self.random_num_obs = False
        self.sample_size_obs = [1, 3]
        for i in range(num_cubes):
            self.create_obstacle_cuboid(size=self.cube_size_medium)

    def sample_random_joint_position_within_workspace(self):
        random_joint_conf = self.np_random.uniform(low=np.array(self.robot.joint_lim_min),
                                                   high=np.array(self.robot.joint_lim_max))
        with self.sim.no_rendering():
            self.robot.set_joint_angles(random_joint_conf)
            goal = np.array(self.get_ee_position())
            self.robot.set_joint_neutral()

            return goal

    def set_robot_random_joint_position(self):
        joint_positions = np.random.uniform(low=np.array(self.robot.joint_lim_min),
                                            high=np.array(self.robot.joint_lim_max))
        self.robot.set_joint_angles(joint_positions)

    def set_robot_random_joint_position_ik(self):
        ee_target = self.sample_random_joint_position_within_workspace()  # self.sample_sphere(0.3, 0.6, upper_half=True)
        joint_positions = self.robot.inverse_kinematics(link=11, position=ee_target)[:7]

        self.robot.set_joint_angles(joint_positions)

    def set_robot_random_joint_position_ik_sphere(self, radius_minor, radius_major):
        ee_target = self.sample_inside_hollow_sphere(radius_minor, radius_major, upper_half=True)
        joint_positions = self.robot.rtb_ik(ee_target)
        self.robot.set_joint_angles(joint_positions)

    def set_robot_random_joint_position_ik_goal_space(self, goal_space):
        ee_target = goal_space()
        joint_positions = self.robot.rtb_ik(ee_target)
        self.robot.set_joint_angles(joint_positions)

    def set_robot_random_pose(self, target_func):
        z_upper_limit = 0.6
        z_lower_limit = 0.4
        valid_pose_found = False
        with self.sim.no_rendering():
            while not valid_pose_found:
                ee_target = target_func()
                joint_positions = self.robot.rtb_ik(ee_target)
                self.robot.set_joint_angles(joint_positions)
                # limits prevent unwanted robot poses
                if z_upper_limit >= self.robot.get_link_position(11)[2] >= z_lower_limit:
                    valid_pose_found = True



    def create_scenario_reach1(self):
        self.goal_range_low = np.array([-0.4 / 2 + 0.6, -0.4 / 2, 0])
        self.goal_range_high = np.array([0.4 / 2 + 0.6, 0.4 / 2, 0.4])
        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(lambda: self.sample_inside_torus(front_half=True))

    def create_scenario_reach2(self):
        goal_radius_minor = 0.5
        goal_radius_major = 0.85
        self._sample_goal = lambda: self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major,
                                                                     upper_half=True, three_quarter_front_half=True)
        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(lambda: self.sample_inside_torus(front_half=True))

    def create_scenario_reach3(self):
        goal_radius_minor = 0.5
        goal_radius_major = 0.85
        self._sample_goal = lambda: self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half=True,
                                                             three_quarter_front_half=True)
        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(self.sample_inside_torus)

    def create_scenario_reach4(self):
        goal_radius_minor = 0.5
        goal_radius_major = 0.85
        self._sample_goal = lambda: self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half=True)
        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(self.sample_inside_torus)

    def create_scenario_base1(self):
        self.create_scenario_reach1()
        self.randomize_obstacle_position = True

        self.create_obstacle_sphere(radius=0.04)

    def create_scenario_base2(self):
        self.randomize_obstacle_position = True
        self.random_num_obs = False
        goal_radius_minor = 0.5
        goal_radius_major = 0.8

        # PandaReach with 1 obstacle
        def sample_base_2_goal():
            return self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half=True,
                                                    front_half=True)

        self._sample_goal = sample_base_2_goal
        self._sample_obstacle = sample_base_2_goal

        for i in range(2):
            self.create_obstacle_sphere(radius=0.05)

        self.robot_pose_randomizer = lambda: self.set_robot_random_joint_position_ik_goal_space(sample_base_2_goal)

    def create_scenario_base2_5(self):
        self.randomize_obstacle_position = True
        self.random_num_obs = False
        goal_radius_minor = 0.5
        goal_radius_major = 0.8

        # PandaReach with 1 obstacle
        def sample_base_2_goal():
            return self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half=True,
                                                    three_quarter_front_half=True)

        self._sample_goal = sample_base_2_goal
        self._sample_obstacle = sample_base_2_goal

        for i in range(2):
            self.create_obstacle_cuboid(self.cube_size_large)

    def create_scenario_base3(self):
        self.randomize_obstacle_position = True
        self.random_num_obs = False
        goal_radius_minor = 0.5
        goal_radius_major = 0.85

        # PandaReach with 1 obstacle
        def sample_base_3_goal():
            return self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half=True)

        def sample_base_3_obstacle():
            sample = self.sample_inside_hollow_sphere(0.1, 0.5)
            return sample + self.goal

        self._sample_goal = sample_base_3_goal
        self._sample_obstacle = sample_base_3_obstacle
        for i in range(3):
            self.create_obstacle_sphere(radius=0.05)

        self.robot_pose_randomizer = lambda: self.set_robot_random_joint_position_ik_goal_space(sample_base_3_goal)

    def create_scenario_base3_randshape(self):
        self.randomize_obstacle_position = True
        self.random_num_obs = True
        goal_radius_minor = 0.5
        goal_radius_major = 0.8

        # PandaReach with 1 obstacle
        def sample_base_3_goal():
            return self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half=True)

        def sample_base_3_obstacle():
            sample = self.sample_inside_hollow_sphere(0.1, 0.5)
            return sample + self.goal

        self._sample_goal = sample_base_3_goal
        self._sample_obstacle = sample_base_3_obstacle
        for i in range(3):
            self.create_obstacle_cuboid(size=self.cube_size_large)
            self.create_obstacle_sphere(radius=0.05)

        self.sample_size_obs = [3, 3]

    def create_scenario_base4(self):
        self.scenario = "wang_3"
        self.create_scenario_wang()
        self.randomize_robot_pose = True
        self.robot_pose_randomizer = lambda: self.set_robot_random_joint_position_ik()

    def create_scenario_wang(self):
        goal_radius_minor = 0.4
        goal_radius_major = 0.95

        def sample_wang_obstacle():

            if np.random.rand() > 0.3:
                # move to goal
                sample = self.sample_inside_hollow_sphere(0.2, 0.6)
                return sample + self.goal
            else:
                sample = self.sample_inside_hollow_sphere(0.2, 0.4)
                return self.robot.get_ee_position() + sample
            # else:
            #     # sample near base
            #     sample = self.sample_sphere(0.3,0.5, True)
            #     return sample + self.robot.get_link_position(0)

        num_spheres = int(self.scenario.split(sep="_")[1])

        def sample_wang_goal():
            return self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half=True)

        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(self.sample_inside_torus)

        self._sample_obstacle = lambda: sample_wang_obstacle()
        self._sample_goal = sample_wang_goal

        self.randomize_obstacle_position = True
        self.random_num_obs = False

        for i in range(num_spheres):
            self.create_obstacle_sphere(radius=0.05)


        self.sim.create_sphere(
                body_name=f"a",
                radius=goal_radius_minor,
                mass=0.0,
                position=np.array([0,0,0]),
                rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
                physics_client=self.sim.physics_client,
                ghost=True
            )


        # major sphere
        self.sim.create_sphere(
                body_name=f"b",
                radius=goal_radius_major,
                mass=0.0,
                position=np.array([0,0,0]),
                rgba_color=np.array([1.0, 1.0, 1.0, 0.2]),
                physics_client=self.sim.physics_client,
                ghost=True
            )

    def create_scenario_wang_experimental(self):
        """Scenario for trying out different variations of wang"""
        goal_radius_minor = 0.5
        goal_radius_major = 0.8

        def sample_wang_obstacle():
            rand = np.random.rand()
            if rand > 0.3:
                # move to goal
                sample = self.sample_inside_hollow_sphere(0.1, 0.5)
                return sample + self.goal
            elif rand > 0.1:
                sample = self.sample_inside_hollow_sphere(0.1, 0.4)
                return self.robot.get_ee_position() + sample
            # else:
            #     return self.sample_sphere(0.3,0.5)
            else:
                # sample near base
                sample = self.sample_inside_hollow_sphere(0.3, 0.6, True)
                return sample + self.robot.get_link_position(0)

        num_obstacles = int(self.scenario.split(sep="_")[1])

        def sample_wang_goal():
            return self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half=True)

        self.robot_pose_randomizer = lambda: self.set_random_robot_base()

        self._sample_obstacle = lambda: sample_wang_obstacle()
        self._sample_goal = sample_wang_goal

        self.randomize_obstacle_position = True
        self.sample_size_obs = [num_obstacles, num_obstacles]
        self.random_num_obs = False

        for i in range(num_obstacles):
            # self.create_obstacle_cuboid(size=self.cube_size_large)
            self.create_obstacle_sphere(radius=0.05)

    def create_showcase(self):
        goal_radius_minor = 0.4
        goal_radius_major = 0.95

        # show goal space
        obstacle_name = "goal_space"
        # position[0] += 0.6

        ids = []
        # minor sphere

        # for i in range(10_000):
        #     self.sim.create_sphere(
        #             body_name=f"showcase_{i}",
        #             radius=0.05,
        #             mass=0.0,
        #             position=self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major),
        #             rgba_color=np.array([1.0, 1.0, 1.0, 0.01]),
        #             physics_client=self.sim.physics_client,
        #             ghost=True
        #         )
        for i in range(3):

            self.create_obstacle_sphere(self.sample_inside_hollow_sphere(goal_radius_minor, goal_radius_major))

        self.sim.create_sphere(
                body_name=f"{obstacle_name}_{len(self.obstacles)}",
                radius=goal_radius_minor,
                mass=0.0,
                position=np.array([0,0,0]),
                rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
                physics_client=self.sim.physics_client,
                ghost=True
            )


        # major sphere
        self.sim.create_sphere(
                body_name=f"{obstacle_name}_{len(self.obstacles)}",
                radius=goal_radius_major,
                mass=0.0,
                position=np.array([0,0,0]),
                rgba_color=np.array([1.0, 1.0, 1.0, 0.2]),
                physics_client=self.sim.physics_client,
                ghost=True
            )




    def sample_from_robot_workspace(self):
        return self.sample_random_joint_position_within_workspace()

    def create_obstacle_sphere(self, position=np.array([0.1, 0, 0.1]), radius=0.02, velocity=np.array([0, 0, 0]),
                               alpha=0.8):
        obstacle_name = "obstacle"
        # position[0] += 0.6

        ids = []
        for physics_client in (self.sim.physics_client,):
            ids.append(self.sim.create_sphere(
                body_name=f"{obstacle_name}_{len(self.obstacles)}",
                radius=radius,
                mass=0.0,
                position=position,
                rgba_color=np.array([0.5, 0, 0, alpha]),
                physics_client=physics_client
            ))

        self.obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = ids[0]
        # self.dummy_obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = ids[1]
        # self.dummy_obstacle_id[ids[0]] = ids[1]

    def create_obstacle_cuboid(self, position=np.array([0.1, 0, 0.1]),
                               size=np.array([0.01, 0.01, 0.01])):
        obstacle_name = "obstacle"
        position[0] += 0.6
        ids = []
        for physics_client in (self.sim.physics_client,):
            ids.append(self.sim.create_box(
                body_name=f"{obstacle_name}_{len(self.obstacles)}",
                half_extents=size,
                mass=0.0,
                position=position,
                rgba_color=np.array([1.0, 0.5, 0.0, 1]),
                physics_client=physics_client,
            ))

        self.obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = ids[0]
        # self.dummy_obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = ids[1]
        # self.dummy_obstacle_id[ids[0]] = ids[1]

    def create_robot_debug_params(self):
        """Create debug params to set the robot joint positions from the GUI."""
        params = {}
        for i in range(pybullet.getNumJoints(self.robot.id, physicsClientId=self.sim_id)):
            joint_name = pybullet.getJointInfo(self.robot.id, i)[1].decode("ascii")
            params[joint_name] = pybullet.addUserDebugParameter(
                joint_name,
                rangeMin=-2 * np.pi,
                rangeMax=2 * np.pi,
                startValue=0,
                physicsClientId=self.sim_id,
            )
        return params

    # def get_obs(self) -> np.ndarray:
    #     if self.obstacles:
    #         q = self.robot.get_joint_angles(self.robot.joint_indices[:7])
    #         obs_per_link = self.collision_detector.compute_distances_per_link(q, self.robot.joint_indices[:7],
    #                                                                           max_distance=10.0)
    #
    #         if self.joint_obstacle_observation == "all":
    #             self.distances_links_to_closest_obstacle = np.array([min(i) for i in obs_per_link.values()])
    #         elif self.joint_obstacle_observation == "closest":
    #             self.distances_links_to_closest_obstacle = min(obs_per_link.values())
    #
    #         self.is_collided = min(self.distances_links_to_closest_obstacle) <= 0.0
    #
    #     return self.distances_links_to_closest_obstacle

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def get_obs(self) -> np.ndarray:
        obstacle_obs = np.zeros(9)
        # check robot collision
        # robot_collision_obs, _ = self.self_collision_detector.compute_distances_per_link(max_distance=999.0)
        # robot_collision_obs = np.array([min(i) for i in robot_collision_obs.values()])
        if self.obstacles:
            # q = self.robot.get_joint_angles(self.robot.joint_indices[:7])
            obs_per_link, info = self.collision_detector.compute_distances_per_link(max_distance=999.0)

            self.distances_links_to_closest_obstacle = np.array([min(i) for i in obs_per_link.values()])
            if self.joint_obstacle_observation == "all":
                obstacle_obs = self.distances_links_to_closest_obstacle
            elif self.joint_obstacle_observation == "all2":
                sorted_obs_per_link = [sorted(i) for i in obs_per_link.values()]
                obstacle_obs = np.array([i[:2] for i in sorted_obs_per_link]).flatten()
            elif self.joint_obstacle_observation == "all3":
                sorted_obs_per_link = [sorted(i) for i in obs_per_link.values()]
                obstacle_obs = np.array([i[:3] for i in sorted_obs_per_link]).flatten()
            elif self.joint_obstacle_observation == "all_close":
                obstacle_obs = np.array([i if i < 0.4 else 1.0 for i in self.distances_links_to_closest_obstacle])
            elif self.joint_obstacle_observation == "closest":
                obstacle_obs = np.array(min(obs_per_link.values()))
            elif self.joint_obstacle_observation == "vectors":
                obstacle_obs = self.get_vector_obs(obs_per_link, info)
                # for a,b in closest_pairs:
                #     p.addUserDebugLine(a, b, physicsClientId=0, lineColorRGB=np.array([0,1,0]))
                # p.removeAllUserDebugItems(physicsClientId=0)
            elif self.joint_obstacle_observation == "vectors+all":
                closest_distances_vectors = self.get_vector_obs(obs_per_link, info)
                closest_distances_all = self.distances_links_to_closest_obstacle
                obstacle_obs = np.concatenate([closest_distances_all, closest_distances_vectors])


            self.is_collided = min(
                self.distances_links_to_closest_obstacle) <= 0.0  # and min(robot_collision_obs) <= 0.0
        else:
            # no obstacles
            if self.joint_obstacle_observation == "vectors":
                obstacle_obs = np.ones(27)

            # self.is_collided = min(robot_collision_obs) <= -0.05
            # print(robot_collision_obs)
        # todo: get end effector error

        distance_obs = np.array([distance(self.robot.get_ee_position(), self.goal)])
        np.concatenate([obstacle_obs, distance_obs])


        return obstacle_obs

    def get_vector_obs(self, obs_per_link, info):
        closest_points = list(info["closest_points"].values())
        obs_distances = list(obs_per_link.values())
        closest_per_obstacle = [sorted(zip(i, y)) for i, y in zip(obs_distances, closest_points)]
        closest_pairs = np.array([x[0][1] for x in closest_per_obstacle])
        closest_distances = np.array([i[0] - i[1] for i in closest_pairs]).flatten()
        return closest_distances

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:

        if self.fixed_target is None:
            self.set_coll_free_goal(["table", "robot"])
        # sample new (collision free) obstacles
        if self.randomize_obstacle_position:
            self.set_coll_free_obs()
        else:
            coll_obj = [i for i in self.obstacles.keys()]
            coll_obj.extend(["table", "robot"])
            self.set_coll_free_goal(coll_obj, margin=0.1)

        self.sim.set_base_pose("dummy_sphere", np.array([0.0, 0.0, -5.0]),
                               np.array([0.0, 0.0, 0.0, 1.0]))  # move dummy away

        if self.randomize_obstacle_velocity:
            self.set_random_obs_velocity()

        if self.random_num_obs:
            self.set_random_num_obs()

        # use rl policy to generate joint angles (to get final pose)
        # arrived = False
        # self.reach_checker_env.task.set_fixed_target(self.goal)
        # obs, _ = self.reach_checker_env.reset()
        #
        # for i in range(50):
        #     action, _ = self.pose_generator_model.predict(obs)
        #     obs, reward, done, truncated, info, = self.reach_checker_env.step(action)
        #     if done and info["is_success"]:
        #         arrived = True
        #         break
        #     elif done:
        #         break
        #
        # if arrived:
        #     optimal_angles = self.reach_checker_env.robot.get_joint_angles(self.robot.joint_indices[:7])
        #     optimal_angles[6] = 0.0
        #     self.robot.optimal_pose = optimal_angles
        # else:
        #     self.robot.optimal_pose = None

        # reset robot actions
        self.robot.recent_action = None
        self.robot.previous_action = None

    def set_random_num_obs(self):
        num_obs = random.randint(self.sample_size_obs[0], self.sample_size_obs[1])
        obs_to_move = abs(num_obs - len(self.obstacles))
        # shuffle obstacle order
        keys = list(self.obstacles.keys())
        np.random.shuffle(keys)

        for obstacle in keys:
            if obs_to_move <= 0:
                break
            obstacle_id = self.obstacles[obstacle]
            # move obstacle far away from work space
            self.sim.set_base_pose(obstacle, np.array([99.9, 99.9, -99.9]), np.array([0.0, 0.0, 0.0, 1.0]))
            # self.sim.set_base_pose_dummy(self.dummy_obstacle_id[obstacle_id], np.array([99.9, 99.9, -99.9]),
            #                        np.array([0.0, 0.0, 0.0, 1.0]),
            #                        physics_client=self.sim.dummy_collision_client)
            obs_to_move -= 1

    def set_random_obs_velocity(self):
        for obstacle in self.obstacles.keys():
            obstacle_id = self.obstacles[obstacle]
            velocity = np.random.uniform(self.velocity_range_low, self.velocity_range_high)
            self.sim.set_base_velocity(obstacle, velocity=velocity)
            # self.sim.set_base_velocity_dummy(self.dummy_obstacle_id[obstacle_id],
            #                                  velocity=velocity,
            #                                  physics_client=self.sim.dummy_collision_client
            #                                  )

    def set_coll_free_goal(self, coll_obj, margin=0.05):
        collision = [True]
        # get collision free goal within goal space
        i = 0
        while any(collision):
            self.goal = self._sample_goal()
            if i > 9999:
                raise StopIteration("Couldn't find collision free goal!")
            else:
                i += 1

            self.sim.set_base_pose("dummy_sphere", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
            # self.sim.physics_client.performCollisionDetection()
            collision = [self.check_collision("dummy_sphere", obj, margin=margin) for obj in coll_obj]
            # collision = [self.check_collision("dummy_sphere", "robot", margin=0.05),
            #              self.check_collision("dummy_sphere", "table", margin=0.05)]

            # for obstacle in self.obstacles:
            #     collision.append(self.check_collision("dummy_sphere", obstacle))
            #     if any(collision):
            #         self.goal = self._sample_goal()
            #         break
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def set_coll_free_obs(self):
        # get randomized obstacle position within goal space
        for obstacle in self.obstacles.keys():
            # get collision free obstacle position
            collision = [True]
            pos = None
            obstacle_id = self.obstacles[obstacle]

            # sample new obstacle position until it doesn't obstruct robot
            i = 0
            while any(collision):
                if i > 9999:
                    raise StopIteration("Couldn't find collision free obstacle!")
                else:
                    i += 1
                pos = self._sample_obstacle()
                self.sim.set_base_pose(obstacle, pos, np.array([0.0, 0.0, 0.0, 1.0]))
                collision = [self.check_collision("robot", obstacle, margin=0.05),
                             self.check_collision("table", obstacle, margin=0.05),
                             self.check_collision("dummy_sphere", obstacle, margin=0.05)]
                # avoid collision with other obstacles
                obs_collision = [self.check_collision(i, obstacle) if i is not obstacle else False
                                 for i in self.obstacles.keys()]
                collision.extend(obs_collision)
                # check if out of boundaries
                self.sim.set_base_pose("dummy_sphere", np.array([0.0, 0.0, 0.0]),
                                       np.array([0.0, 0.0, 0.0, 1.0]))  # move dummy to origin
                collision.append(not self.check_collision("dummy_sphere", obstacle, margin=1.0))

            # else:
            #     if pos is not None:
            #         self.sim.set_base_pose_dummy(self.dummy_obstacle_id[obstacle_id], pos,
            #                                      np.array([0.0, 0.0, 0.0, 1.0]),
            #                                      physics_client=self.sim.dummy_collision_client)

    def check_collision(self, obstacle_1, obstacle_2, margin=0.0):
        """Check if given bodies collide."""
        # margin in which overlapping counts as a collision
        # self.sim.physics_client.performCollisionDetection()
        closest_points = p.getClosestPoints(bodyA=self.bodies[obstacle_1], bodyB=self.bodies[obstacle_2],
                                            distance=10.0,
                                            physicsClientId=self.sim_id)
        if closest_points:
            contact_distance = np.min([pt[8] for pt in closest_points])
            collision = margin >= contact_distance
        else:
            collision = False
        return collision

    def sample_from_goal_range(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def sample_inside_hollow_sphere(self, radius_minor, radius_major, upper_half=False, front_half=False,
                                    three_quarter_front_half=False) -> np.ndarray:

        phi = np.random.uniform(0, 2 * np.pi)
        if upper_half:
            theta = np.random.uniform(0, 0.5 * np.pi)
        else:
            theta = np.random.uniform(0, np.pi)

        if front_half:
            phi = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)

        if three_quarter_front_half:
            phi = np.random.uniform(-0.75 * np.pi, 0.75 * np.pi)

        r = np.cbrt(np.random.uniform(radius_minor ** 3, radius_major ** 3))

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        sample = np.array([x, y, z])

        return sample

    def sample_inside_torus(self, R=0.5, r=0.05, front_half=False):
        """
        R : float
            Distance from the center of the tube to the center of the torus.
        r : float
            Radius of the tube (i.e., cross-sectional radius of the torus).
        """

        # Generate random angles theta and phi
        theta = 2.0 * np.pi * np.random.rand()
        if front_half:
            theta = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)

        phi = 2.0 * np.pi * np.random.rand()

        # Generate a random radius
        rad = r * np.sqrt(np.random.rand())

        # Convert to Cartesian coordinates
        x = (R + rad * np.cos(phi)) * np.cos(theta)
        y = (R + rad * np.cos(phi)) * np.sin(theta)
        z = rad * np.sin(phi)

        return np.array([x, y, z+0.5])

    def set_random_robot_base(self):
        self.robot.neutral_joint_values[0] = np.random.uniform(self.robot.joint_lim_min[0], self.robot.joint_lim_max[0])
        self.robot.set_joint_neutral()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def is_truncated(self) -> np.ndarray:
        return np.array(self.is_collided, dtype=np.bool8)

    def update_labels(self, manipulability, distance, obstacle_distances, action_difference, action_magnitude, reward):
        with self.sim.no_rendering():
            # self.sim.remove_all_debug_text()
            # self.sim.remove_debug_text(self.debug_dist_label_name)
            # self.sim.remove_debug_text(self.debug_manip_label_name)

            try:
                self.sim.create_debug_text(self.debug_dist_label_name,
                                           f"{self.debug_dist_label_base_text} {np.round(distance, 3)}")
                self.sim.create_debug_text(self.debug_manip_label_name,
                                           f"{self.debug_manip_label_base_text} {np.round(manipulability, 5)}")
                self.sim.create_debug_text(self.debug_obs_label_name,
                                           f"{self.debug_obs_label_base_text} {np.round(np.min(obstacle_distances), 5)}")
                self.sim.create_debug_text(self.debug_action_difference_label_name,
                                           f"{self.debug_action_difference_base_text} {np.round(action_difference, 5)}")
                self.sim.create_debug_text(self.debug_effort_label_name,
                                           f"{self.debug_effort_base_text} {np.round(action_magnitude, 5)}")
                self.sim.create_debug_text(self.debug_reward_label_name,
                                           f"{self.debug_reward_base_text} {np.round(reward, 5)}")
                pass
            except BaseException:
                pass

    def get_norm_action_difference(self) -> float:
        """Calculate the magnitude of deviation between the recent and the previous action."""
        if self.robot.previous_action is None:
            # There has not been any recorded action yet
            return 0.0

        action_diff = self.robot.recent_action - self.robot.previous_action
        return np.linalg.norm(action_diff)

    def get_norm_effort(self) -> float:
        """Calculate the magnitude of the action, i.e. calculate the square of the norm of the action."""

        return np.linalg.norm(self.robot.current_acceleration)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        ee_error = distance(achieved_goal, desired_goal)
        manipulability = self.robot.get_manipulability()
        obs_distance = self.distances_links_to_closest_obstacle
        action_diff = self.get_norm_action_difference()
        effort = self.get_norm_effort()

        if self.reward_type == "sparse":
            reward = -np.array((ee_error > self.distance_threshold), dtype=np.float32)
        elif self.reward_type == "wang":
            weight_distance = 10e-3
            weight_obs = 0.1
            tolerance_distance = 10e-4
            # IPPO reward
            distance_reward = weight_distance * np.square(ee_error) + np.log(
                np.square(ee_error) + tolerance_distance)
            obstacle_reward = weight_obs * np.sum([np.max([0, 1 - d / 0.05]) for d in obs_distance])
            reward = -np.array(distance_reward + obstacle_reward, dtype=np.float32)
        elif self.reward_type == "kumar":
            weight_distance = 20
            weight_effort = 0.005
            weight_obs = 0.1

            distance_reward = np.exp(-weight_distance * np.square(ee_error))
            effort_reward = - weight_effort * effort
            obstacle_reward = - weight_obs * np.sum([np.max([0, 1 - d / 0.05]) for d in obs_distance])
            reward = np.array(distance_reward + effort_reward + obstacle_reward, dtype=np.float32)

        else:
            # calculate dense rewards
            reward = -np.array(effort * self.factor_punish_effort, dtype=np.float32)

            ee_error *= self.factor_punish_distance
            if self.is_collided:
                reward += -np.array(ee_error + 100 * self.factor_punish_collision, dtype=np.float32)
            else:
                # punish being close to obstacles
                reward += -np.array(ee_error + exp(-self.is_collided) * self.factor_punish_obstacle_proximity,
                                    dtype=np.float32)

        if self.sim.render_env and self.show_debug_labels:
            self.update_labels(manipulability, ee_error, obs_distance, action_diff, effort, reward)

        if self.truncate_episode_on_collision:
            reward -= self.is_collided * -self.collision_reward  # prevent double negative in collision reward

        self.action_diff = action_diff
        self.action_magnitude = effort
        self.manipulability = manipulability

        return reward

    def show_goal_space(self):

        x = (self.goal_range_high[0] - self.goal_range_low[0]) / 2
        y = (self.goal_range_high[1] - self.goal_range_low[1]) / 2
        z = (self.goal_range_high[2] - self.goal_range_low[2]) / 2

        d = [self.goal_range_low, self.goal_range_high]

        d.append(np.array([self.goal_range_low[0], self.goal_range_high[1], self.goal_range_high[2]]))
        d.append(np.array([self.goal_range_low[0], self.goal_range_low[1], self.goal_range_high[2]]))
        d.append(np.array([self.goal_range_low[0], self.goal_range_high[1], self.goal_range_low[2]]))
        d.append(np.array([self.goal_range_high[0], self.goal_range_low[1], self.goal_range_high[2]]))
        d.append(np.array([self.goal_range_high[0], self.goal_range_low[1], self.goal_range_low[2]]))
        d.append(np.array([self.goal_range_high[0], self.goal_range_high[1], self.goal_range_low[2]]))
        d.append(np.array([self.goal_range_high[0], self.goal_range_low[1], self.goal_range_high[2]]))

        subset = list(itertools.combinations(d, 2))

        for d1, d2 in subset:
            self.sim.create_debug_line(d1, d2)

        # for dot in d:
        #     for dot2 in d:
        #         self.sim.create_debug_line(dot, dot2)

        # self.sim.create_box(
        #     body_name="goal_space",
        #     ghost=True,
        #     half_extents=np.array([x, y, z]),
        #     mass=0.0,
        #     position=np.array([0.0, 0.0, self.goal_range_high[2]/2]),
        #     rgba_color=np.array([0.0, 0.0, 0.5, 0.2])
        # )

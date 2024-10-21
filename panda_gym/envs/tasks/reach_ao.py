import itertools
import json
import logging
import random
import sys
import time
from copy import copy
from math import exp
from pathlib import Path
from typing import Any, Dict

import gymnasium
import numpy as np
import pybullet as p
from pyb_utils.collision import NamedCollisionObject, CollisionDetector

from panda_gym.envs.core import Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.utils import distance, unit_vector
from classes.train_config import TrainConfig

sys.modules["gym"] = gymnasium

ROOT_DIR = Path(__file__).parent.parent.parent
SCENARIO_DIR = f"{ROOT_DIR}\\assets\\scenarios"


class ReachAO(Task):
    def __init__(
            self,
            sim,
            robot,
            get_ee_position,
            scenario='wangexp_3',
            config=TrainConfig(),
            ee_error_threshold=0.05,
            speed_threshold=0.5,

    ) -> None:
        super().__init__(sim)

        self.config = config

        if self.config.task_observations is None:
            task_observations = {'obstacles': 'vectors+closest_per_link', 'prior': None}

        self.sim_id = self.sim.physics_client._client

        self.robot: Panda = robot
        self.obstacles = {}

        self.past_obstacle_observations = []
        if "vectors+past" in self.config.task_observations["obstacles"]:
            self.obstacle_obs = "vectors"
            for _ in range(10):
                self.get_obs()

        self.obstacle_obs = self.config.task_observations["obstacles"]

        self.scenario = scenario
        self.randomize_robot_pose = config.randomize_robot_pose
        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(
            self.sample_inside_torus)  # lambda: self.set_robot_random_joint_position_ik_sphere(0.45, 0.5)

        # self.robot_params = self.create_robot_debug_params()
        self.cube_size_large = np.array([0.05, 0.05, 0.05])
        self.cube_size_medium = np.array([0.03, 0.03, 0.03])
        self.cube_size_small = np.array([0.02, 0.02, 0.02])
        self.cube_size_mini = np.array([0.01, 0.01, 0.01])

        self.ee_error_threshold = np.float32(ee_error_threshold)
        self.ee_speed_threshold = np.float32(speed_threshold)
        self.get_ee_position = get_ee_position
        self.goal_range = 0.3
        self.obstacle_count = 0
        self.goal_reached = False

        self._sample_goal = self._sample_obstacle = self.sample_from_goal_range
        # set default goal range
        self.x_offset = 0.6
        self.goal_range_low = np.array([-self.goal_range / 2.5 + self.x_offset, -self.goal_range / 1.5, 0])
        self.goal_range_high = np.array([self.goal_range / 2.5 + self.x_offset, self.goal_range / 1.5, self.goal_range])

        self.truncate_episode_on_collision = config.truncate_on_collision
        if not self.truncate_episode_on_collision:
            self.is_truncated = lambda: ()

        # # init pose generator env
        # self.reach_checker_env = gym.make("PandaReachChecker", control_type = "js", render=False)

        # # load pose checker model
        # path = pathlib.Path(__file__).parent.resolve()
        # self.pose_generator_model = TD3.load(fr"{path}/pose_generator_model.zip",
        #                                      env=self.reach_checker_env)

        self.bodies = {"robot": self.robot.id}

        exclude_links = ["panda_grasptarget", "panda_leftfinger", "panda_rightfinger", "panda_hand"]
        self.collision_links = [i for i in self.robot.link_names if i not in exclude_links]
        self.collision_objects = []

        # set default obstacle mode
        self.randomize_obstacle_position = False  # Randomize obstacle placement
        self.randomize_obstacle_velocity = False
        self.random_size_cuboids = False
        self.random_num_obs = False
        self.sample_size_obs = [0, 0]
        self.allow_overlapping_obstacles = False

        # extra observations
        self.distances_closest_obstacles = np.zeros(len(self.collision_links))
        self.is_collided = False
        self.action_magnitude = 0
        self.action_diff = 0
        self.manipulability = 0

        # default visual settings (on render)
        self.cameraTargetPosition = (0.06289377063512802, 0.05091303586959839, 0.27599984407424927)
        self.cameraDistance = 1.359999656677246
        self.cameraPitch = -33.20000076293945
        self.cameraYaw = 51.800025939941406

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
                    # get all links of obstacle
                    num_joints = self.sim.physics_client.getNumJoints(obstacle_id)
                    if num_joints > 0:
                        # if joints, add each link as separate object
                        for joint_index in range(num_joints):
                            joint_info = self.sim.physics_client.getJointInfo(obstacle_id, joint_index)
                            self.collision_objects.append(
                                NamedCollisionObject(obstacle_name, link_name=joint_info[12].decode()))
                    else:
                        # if no joints, add as single object
                        self.collision_objects.append(NamedCollisionObject(obstacle_name))
                    self.bodies[obstacle_name] = obstacle_id

            # set velocity range
            self.velocity_range_low = np.array([-0.2, -0.2, -0.2])
            self.velocity_range_high = np.array([0.2, 0.2, 0.2])

        if config.show_goal_space:
            self.visualize_space_from_range(self.goal_range_low, self.goal_range_high)

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

        if self.config.render:
            # set camera
            p.resetDebugVisualizerCamera(cameraDistance=self.cameraDistance, cameraYaw=self.cameraYaw,
                                         cameraPitch=self.cameraPitch, cameraTargetPosition=self.cameraTargetPosition,
                                         physicsClientId=self.sim_id)

        # overwrite standard pybullet step
        def step_check_collision():
            """Step the simulation and check for collision at each step."""
            for _ in range(config.n_substeps):
                self.sim.physics_client.stepSimulation()
                self.is_collided = self.check_collided()
                if self.is_collided:
                    break

        def step_check_collision_once():
            """Step the simulation for n steps and check for collision once after (Higher performance, but can lead to
            collisions not being caught)."""
            for _ in range(config.n_substeps):
                self.sim.physics_client.stepSimulation()
            if not self.is_collided:
                self.is_collided = self.check_collided()

        sim.step = lambda: step_check_collision()

        if self.config.show_debug_labels:
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
            self.debug_jerk_label_name = "jerk"
            self.debug_jerk_base_text = "Jerk:"

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
        scenario = scenario.split(sep="-")
        name = scenario[0]

        scenarios = {
            "wang": self.create_scenario_wang,
            "wangexp": self.create_scenario_wang_experimental,
            "narrow_tunnel": self.create_scenario_narrow_tunnel,
            "tunnel": self.create_scenario_tunnel,
            "library": self.create_scenario_library,
            "library1": self.create_scenario_library,
            "library2": self.create_scenario_library,
            "workshop": self.create_scenario_workshop,
            "workshop2": self.create_scenario_workshop,
            "workshop3": self.create_scenario_workshop,
            "industrial": self.create_scenario_industrial,
            "kasys": self.create_scenario_kasys,
            "wall": self.create_stage_wall,
            "reachao1": self.create_scenario_reachao1,
            "reachao2": self.create_scenario_reachao2,
            "reachao3": self.create_scenario_reachao3,
            "reachao_rand": self.create_scenario_reachao_rand,
            "reachao_rand_start": self.create_scenario_reachao_rand_start,
            "reachao_rand_shape": self.create_scenario_reachao_rand_shape,
            "reach1": self.create_scenario_reach1,
            "reach2": self.create_scenario_reach2,
            "reach3": self.create_scenario_reach3,
            "showcase": self.create_showcase,
            "warehouse": self.create_scenario_warehouse,
            "countertop": self.create_scenario_countertop,
            "kitchen": self.create_scenario_kitchen,
            "raised_shelves": self.create_scenario_raised_shelves,
            "tabletop": self.create_scenario_tabletop,
            "tabletop2": self.create_scenario_tabletop2,
            "bookshelves": self.create_scenario_bookshelves,
        }

        if scenarios.get(name) is None:
            logging.warning("Scenario not found. Aborting")
            raise Exception(f"Scenario {scenario} not found!")
        else:
            return scenarios[name]

    def _create_scene(self):

        self.sim.create_plane(z_offset=-0.4)
        # self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=0.3)
        self.bodies["table"] = self.sim.create_table(length=2.0, width=1.3, height=0.4)
        #self.collision_objects.append(NamedCollisionObject("table"))

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

    def setup_benchmark_scenario(self, scenario_name):
        scenario_dir = f"{SCENARIO_DIR}\\{scenario_name}"

        with open(f"{scenario_dir}\\{scenario_name}.json") as t:
            urdfs = json.load(t)

        # append path
        for urdf in urdfs.values():
            fileName = urdf["fileName"]
            urdf["fileName"] = f"{scenario_dir}\\urdf\\{fileName}"

        indexes = self.sim.load_scenario(urdfs)
        for idx, body_name, urdf in zip(indexes, urdfs.keys(), urdfs.values()):
            name = f"{scenario_name}_obj_{body_name}"
            self.obstacles[f"{name}"] = idx

    def create_scenario_narrow_tunnel(self):

        def robot_pose_randomizer():
            range_low = np.array([0.0, -0.6, 0.2])
            range_high = np.array([0.5, -0.5, 0.7])
            target = self.np_random.uniform(range_low, range_high)
            joint_positions = self.robot.rtb_ik(target)
            self.robot.set_joint_angles(joint_positions)

            self.visualize_space_from_range(range_low, range_high, np.array([1.0, 0.0, 0.0, 0.1]))

        self.randomize_robot_pose = True
        self.robot_pose_randomizer = robot_pose_randomizer

        self.robot.neutral_joint_values = np.array([-1.0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])

        self.robot.set_joint_neutral()
        self.goal_range_low = np.array([0.55, 0.2, 0.2])
        self.goal_range_high = np.array([0.75, 0.4, 0.75])

        self.setup_benchmark_scenario("narrow_tunnel")

        # visual settings
        self.cameraTargetPosition = (0.6126745939254761, -0.002699121832847595, 0.3287460505962372)
        self.cameraDistance = 1.119999885559082
        self.cameraPitch = -42.800045013427734
        self.cameraYaw = 221.39991760253906

    def create_scenario_tunnel(self):

        # self.robot.neutral_joint_values = np.array(
        #     [-2.34477029,  1.69617261,  1.81619755, - 1.98816377, - 1.58805049,  1.2963265,
        #      0.41092735]
        #     )

        self.robot.neutral_joint_values = np.array([-1.0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])

        self.robot.set_joint_neutral()
        self.goal_range_low = np.array([0.55, 0.2, 0.2])
        self.goal_range_high = np.array([0.75, 0.4, 0.75])

        self.setup_benchmark_scenario("tunnel")

    def create_scenario_workshop(self):

        # default workshop scenario
        self.goal_range_low = np.array([-0.7, -0.7, 0.4])
        self.goal_range_high = np.array([0.1, -0.4, 0.7])

        self.robot.neutral_joint_values = np.array(
            [0.00887326, - 0.05377409, - 0.03621967, - 1.9094068, 0.08791409, 2.00265486,
             0.76681184])
        self.robot.set_joint_neutral()

        if self.scenario == "workshop2":
            def robot_pose_randomizer():
                pose_sampling_range_low = np.array([-0.5, -0.6, 0.6])
                pose_sampling_range_high = np.array([0.2, -0.3, 0.7])
                target = self.np_random.uniform(pose_sampling_range_low, pose_sampling_range_high)
                joint_positions = self.robot.rtb_ik(target)
                self.robot.set_joint_angles(joint_positions)

                self.visualize_space_from_range(pose_sampling_range_low, pose_sampling_range_high, np.array([1.0, 0.0, 0.0, 0.1]))



            self.randomize_robot_pose = True
            self.robot_pose_randomizer = robot_pose_randomizer

            self.goal_range_low = np.array([0.5, -0.15, 0.4])
            self.goal_range_high = np.array([0.6, 0.15, 0.5])


        self.goal = self.config.fixed_target

        self.setup_benchmark_scenario("workshop")



        # visual settings
        self.cameraTargetPosition = (-0.05694245547056198, -0.08760415762662888, 0.3400000035762787)
        self.cameraDistance = 1.299999713897705
        self.cameraPitch = -42.8000373840332
        self.cameraYaw = 224.19996643066406

    def create_scenario_industrial(self):

        def robot_pose_randomizer():
            range_low = np.array([-0.5, -0.8, 0.4])
            range_high = np.array([0.2, -0.4, 0.7])
            target = self.np_random.uniform(range_low, range_high)
            joint_positions = self.robot.inverse_kinematics(link=self.robot.ee_link, position=target)[:7]
            self.robot.set_joint_angles(joint_positions)

        self.randomize_robot_pose = True
        self.robot_pose_randomizer = robot_pose_randomizer

        self.robot.set_joint_neutral()

        self.goal_range_low = np.array([0.5, -0.1, 0.55])
        self.goal_range_high = np.array([0.6, 0.1, 0.75])

        self.goal = self.config.fixed_target
        self.setup_benchmark_scenario("industrial")

    def create_scenario_kasys(self):

        self.robot.set_joint_neutral()
        self.goal_range_low = np.array([1.4, -0.15, 0.45])
        self.goal_range_high = np.array([1.7, 0.12, 0.6])
        self.goal = self.config.fixed_target

        self.setup_benchmark_scenario("kasys")

    def create_scenario_library(self):

        self.robot.neutral_joint_values = [0.0, 0.12001979, 0.0, -1.64029458, 0.02081271, 3.1,
                                           0.77979846]  # above table

        self.robot.set_joint_neutral()

        self.setup_benchmark_scenario("library")

        # Create custom goal space
        if self.scenario == "library1":
            self.robot.neutral_joint_values = [-2.961, -0.031, -0.212, -1.603, 0.008, 3.087, 0.775]
            self.goal_range_low = np.array([0.5, -0.3, 0])
            self.goal_range_high = np.array([0.85, 0.3, 0.3])
        elif self.scenario == "library2":
            self.goal_range_low = np.array([-0.7, -0.4, 0.4])
            self.goal_range_high = np.array([-0.55, 0.4, 0.85])
        else:
            self.goal_range_low = np.array([0.2, -0.3, 0])
            self.goal_range_high = np.array([0.7, 0.3, 0.6])

        # visual settings
        self.cameraTargetPosition = (0.09789270907640457, 0.09457920491695404, 0.3600000739097595)
        self.cameraDistance = 1.5399994850158691
        self.cameraPitch = -36.4000129699707
        self.cameraYaw = 49.800018310546875

    def create_scenario_bookshelves(self):
        self.robot.set_joint_neutral()

        self.setup_benchmark_scenario("bookshelves")

        self.goal_range_low = np.array([0.6, -0.35, 0.2])
        self.goal_range_high = np.array([0.7, 0.35, 0.8])

    def create_stage_wall(self):
        """parkour."""

        self.goal_range_low = np.array([0.45, -0.6, 0.1])
        self.goal_range_high = np.array([0.7, -0.1, 0.3])

        self.robot.neutral_joint_values = [0.94551719, 0.65262327, 0.12742699, -1.74347465, -0.16996126, 1.97424632,
                                           0.88058222]

        self.create_obstacle_cuboid(
            position=np.array([0.0, 0.0, 0.1]),
            size=np.array([0.2, 0.05, 0.3]))

    def create_scenario_warehouse(self):
        self.robot.set_joint_neutral()

        self.setup_benchmark_scenario("tabletop2")

        self.goal_range_low = np.array([0.5, -0.3, 0])
        self.goal_range_high = np.array([0.85, 0.3, 0.3])

    def create_scenario_countertop(self):
        self.robot.set_joint_neutral()

        self.setup_benchmark_scenario("countertop")

        self.goal_range_low = np.array([0.5, -0.3, 0])
        self.goal_range_high = np.array([0.85, 0.3, 0.3])

    def create_scenario_kitchen(self):
        self.robot.set_joint_neutral()

        self.setup_benchmark_scenario("kitchen")

        self.goal_range_low = np.array([0.5, -0.3, 0])
        self.goal_range_high = np.array([0.85, 0.3, 0.3])

    def create_scenario_raised_shelves(self):
        self.robot.set_joint_neutral()

        self.setup_benchmark_scenario("raised_shelves")

        self.goal_range_low = np.array([0.5, -0.3, 0])
        self.goal_range_high = np.array([0.85, 0.3, 0.3])

    def create_scenario_tabletop(self):
        self.robot.set_joint_neutral()

        self.setup_benchmark_scenario("tabletop")

        self.goal_range_low = np.array([0.5, -0.3, 0])
        self.goal_range_high = np.array([0.85, 0.3, 0.3])

    def create_scenario_tabletop2(self):
        self.robot.set_joint_neutral()

        self.setup_benchmark_scenario("tabletop2")

        self.goal_range_low = np.array([0.5, -0.3, 0])
        self.goal_range_high = np.array([0.85, 0.3, 0.3])

    def create_scenario_reach1(self):
        self.goal_range_low = np.array([-0.4 / 2 + 0.6, -0.4 / 2, 0])
        self.goal_range_high = np.array([0.4 / 2 + 0.6, 0.4 / 2, 0.4])
        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(
            lambda: self.sample_inside_torus(front_half_only=True))

    def create_scenario_reach2(self):
        goal_radius_minor = 0.5
        goal_radius_major = 0.85
        self._sample_goal = lambda: self.sample_within_hollow_sphere(goal_radius_minor, goal_radius_major,
                                                                     upper_half_only=True,
                                                                     three_quarter_front_half_only=True)
        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(
            lambda: self.sample_inside_torus(front_half_only=True))

    def create_scenario_reach3(self):
        goal_radius_minor = 0.5
        goal_radius_major = 0.85
        self._sample_goal = lambda: self.sample_within_hollow_sphere(goal_radius_minor, goal_radius_major,
                                                                     upper_half_only=True,
                                                                     three_quarter_front_half_only=True)
        self.robot_pose_randomizer = lambda: self.set_robot_random_pose(self.sample_inside_torus)

    def create_scenario_reachao1(self):
        self.create_scenario_reach1()
        self.randomize_obstacle_position = True

        self.create_obstacle_sphere(radius=0.04)

    def create_scenario_reachao2(self):
        self.randomize_obstacle_position = True
        self.random_num_obs = False
        goal_radius_minor = 0.5
        goal_radius_major = 0.8

        # PandaReach with 1 obstacle
        def sample_reachao2_goal():
            return self.sample_within_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half_only=True,
                                                    front_half_only=True)

        self._sample_goal = sample_reachao2_goal
        self._sample_obstacle = self.sample_obstacle_wang

        for i in range(2):
            self.create_obstacle_sphere(radius=0.05)

        self.robot_pose_randomizer = lambda: self.set_robot_random_joint_position_ik_goal_space(sample_reachao2_goal)

    def sample_reachao3_goal(self, goal_radius_minor, goal_radius_major):
        return self.sample_within_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half_only=True)

    def sample_reachao3_obstacle(self):
        sample = self.sample_within_hollow_sphere(0.1, 0.5)
        return sample + self.goal

    def create_scenario_reachao3(self):
        self.randomize_obstacle_position = True
        self.random_num_obs = False
        goal_radius_minor = 0.5
        goal_radius_major = 0.8

        self._sample_goal = lambda: self.sample_within_hollow_sphere(goal_radius_minor, goal_radius_major,
                                                                     upper_half_only=True)
        self._sample_obstacle = lambda: self.sample_obstacle_wang()
        for i in range(3):
            self.create_obstacle_sphere(radius=0.05)

        self.robot_pose_randomizer = lambda: self.set_robot_random_joint_position_ik_goal_space(self._sample_goal)

    def create_scenario_reachao_rand(self):
        """Random number of obstacles up to 6, random position, overlapping allowed; Rest as in reachao3"""
        self.create_scenario_reachao3()

        self.randomize_obstacle_position = True
        self.random_num_obs = True
        self.allow_overlapping_obstacles = True

        self._sample_obstacle = lambda: self.sample_obstacle_experimental()
        for i in range(3):
            self.create_obstacle_cuboid(size=self.cube_size_large)

        self.sample_size_obs = [4, 6]

    def create_scenario_reachao_rand_start(self):
        self.create_scenario_reachao_rand()
        self.randomize_robot_pose = True
        self.robot_pose_randomizer = lambda: self.set_robot_random_joint_position_ik_sphere(0.45, 0.7)

    def create_scenario_reachao_rand_shape(self):
        self.create_scenario_reachao_rand()
        self.random_size_cuboids = True

    def sample_obstacle_default(self):

        if self.np_random.random() > 0.3:
            # move to goal
            sample = self.sample_within_hollow_sphere(0.1, 0.4)
            return sample + self.goal
        else:
            sample = self.sample_within_hollow_sphere(0.1, 0.4)
            return self.robot.get_ee_position() + sample

    def sample_obstacle_wang(self, front_half_only=False, upper_half_only=False):
        rand = self.np_random.random()
        if rand > 0.3:
            # sample near goal
            sample = self.sample_within_hollow_sphere(0.1, 0.5, upper_half_only, front_half_only)
            return sample + self.goal
        elif rand > 0.1:
            # sample near ee
            sample = self.sample_within_hollow_sphere(0.1, 0.4, upper_half_only, front_half_only)
            return self.robot.get_ee_position() + sample
        else:
            # sample near base
            sample = self.sample_within_hollow_sphere(0.3, 0.6, True)
            return sample + self.robot.get_link_position(0)

    def sample_obstacle_experimental(self, front_half_only=False, upper_half_only=False):
        rand = self.np_random.random()
        if rand > 0.5:
            # sample near goal
            sample = self.sample_within_hollow_sphere(0.1, 0.25, upper_half_only, front_half_only)
            return sample + self.goal
        elif rand > 0.3:
            # sample near ee
            sample = self.sample_within_hollow_sphere(0.1, 0.4, upper_half_only, front_half_only)
            return self.robot.get_ee_position() + sample
        else:
            # sample near base
            sample = self.sample_within_hollow_sphere(0.3, 0.6, True)
            return sample + self.robot.get_link_position(0)

    def create_scenario_wang(self):
        goal_radius_minor = 0.4
        goal_radius_major = 0.95

        def sample_wang_obstacle():

            if self.np_random.random() > 0.3:
                # move to goal
                sample = self.sample_within_hollow_sphere(0.2, 0.6)
                return sample + self.goal
            else:
                sample = self.sample_within_hollow_sphere(0.2, 0.4)
                return self.robot.get_ee_position() + sample
            # else:
            #     # sample near base
            #     sample = self.sample_sphere(0.3,0.5, True)
            #     return sample + self.robot.get_link_position(0)

        num_spheres = int(self.scenario.split(sep="-")[1])

        def sample_wang_goal():
            return self.sample_within_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half_only=True)

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
            position=np.array([0, 0, 0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
            physics_client=self.sim.physics_client,
            ghost=True
        )

        # major sphere
        self.sim.create_sphere(
            body_name=f"b",
            radius=goal_radius_major,
            mass=0.0,
            position=np.array([0, 0, 0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.2]),
            physics_client=self.sim.physics_client,
            ghost=True
        )

    def create_scenario_wang_experimental(self):
        """Scenario for trying out different variations of wang"""
        goal_radius_minor = 0.5
        goal_radius_major = 0.8

        num_obstacles = int(self.scenario.split(sep="-")[1])

        def sample_wang_goal():
            return self.sample_within_hollow_sphere(goal_radius_minor, goal_radius_major, upper_half_only=True)

        self.robot_pose_randomizer = lambda: self.set_random_robot_base()

        self._sample_obstacle = lambda: self.sample_obstacle_wang()
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
            self.create_obstacle_sphere(self.sample_within_hollow_sphere(goal_radius_minor, goal_radius_major))

        self.sim.create_sphere(
            body_name=f"{obstacle_name}_{len(self.obstacles)}",
            radius=goal_radius_minor,
            mass=0.0,
            position=np.array([0, 0, 0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
            physics_client=self.sim.physics_client,
            ghost=True
        )

        # major sphere
        self.sim.create_sphere(
            body_name=f"{obstacle_name}_{len(self.obstacles)}",
            radius=goal_radius_major,
            mass=0.0,
            position=np.array([0, 0, 0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.2]),
            physics_client=self.sim.physics_client,
            ghost=True
        )

    def sample_from_robot_workspace(self):
        return self.sample_random_joint_position_within_workspace()

    def sample_random_joint_position_within_workspace(self):
        random_joint_conf = self.np_random.uniform(low=np.array(self.robot.joint_lim_min),
                                                   high=np.array(self.robot.joint_lim_max))
        with self.sim.no_rendering():
            self.robot.set_joint_angles(random_joint_conf)
            goal = np.array(self.get_ee_position())
            self.robot.set_joint_neutral()

            return goal

    def set_robot_random_joint_position(self):
        joint_positions = self.np_random.uniform(low=np.array(self.robot.joint_lim_min),
                                                 high=np.array(self.robot.joint_lim_max))
        self.robot.set_joint_angles(joint_positions)

    def set_robot_random_joint_position_ik(self):
        ee_target = self.sample_random_joint_position_within_workspace()  # self.sample_sphere(0.3, 0.6, upper_half=True)
        joint_positions = self.robot.inverse_kinematics(link=16, position=ee_target)[:7]

        self.robot.set_joint_angles(joint_positions)

    def set_robot_random_joint_position_ik_sphere(self, radius_minor, radius_major):
        collided = True
        while collided:
            ee_target = self.sample_within_hollow_sphere(radius_minor, radius_major, upper_half_only=True)
            joint_positions = self.robot.rtb_ik(ee_target)
            self.robot.set_joint_angles(joint_positions)
            collided = self.check_collided(0.0)

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

    def create_obstacle_sphere(self, name=None, position=np.array([0.1, 0, 0.1]), radius=0.02,
                               velocity=np.array([0, 0, 0]),
                               alpha=1.0):
        obstacle_name = "sphere"

        ids = []
        for physics_client in (self.sim.physics_client,):
            ids.append(self.sim.create_sphere(
                body_name=f"{obstacle_name}_{len(self.obstacles)}",
                radius=radius,
                mass=0.0,
                position=position,
                rgba_color=np.array([1, 0, 0, alpha]),
                physics_client=physics_client
            ))

        if name is None:
            name = f"{obstacle_name}_{len(self.obstacles)}"

        self.obstacles[name] = ids[0]
        self.bodies[name] = ids[0]

    def create_obstacle_cuboid(self, name=None, position=np.array([0.1, 0, 0.1]),
                               size=np.array([0.01, 0.01, 0.01])):
        obstacle_name = "cuboid"

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

        if name is None:
            name = f"{obstacle_name}_{len(self.obstacles)}"

        self.obstacles[name] = ids[0]
        self.bodies[name] = ids[0]

    def create_robot_debug_params(self):
        """Create debug params to set the robot joint positions from the GUI."""
        params = {}
        for i in range(p.getNumJoints(self.robot.id, physicsClientId=self.sim_id)):
            joint_name = p.getJointInfo(self.robot.id, i)[1].decode("ascii")
            params[joint_name] = p.addUserDebugParameter(
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

    def check_collided(self, safety_distance=0.0) -> bool:
        self.distances_closest_obstacles = self.collision_detector.get_distances_per_link(max_distance=999.0)
        distances_table = self.collision_detector.get_link_object_distance(object_name="table", ignore_link=["panda_link0", "panda_link1"])
        return (min(self.distances_closest_obstacles) <= safety_distance
                or min(distances_table) <= safety_distance)  # safety distance

    def get_obs(self) -> np.ndarray:

        obstacle_obs = np.ones(27)
        if self.obstacles:
            # q = self.robot.get_joint_angles(self.robot.joint_indices[:7])
            obs_per_link, info = self.collision_detector.compute_distances_per_link(max_distance=999.0)
            match self.obstacle_obs:
                case "closest_per_link":
                    # returns closest obstacle per link
                    obstacle_obs = np.array([min(i) for i in obs_per_link.values()])
                case "closest":
                    # returns closest obstacle of all links
                    obstacle_obs = np.array(min(obs_per_link.values()))
                case "vectors":
                    # returns unit vectors pointing to the closest obstacles for each link
                    obstacle_obs = self.get_vector_obs(obs_per_link, info)
                    self.past_obstacle_observations.append(obstacle_obs)
                case "vectors+past":
                    # concatenates current and past unit vectors pointing to the closest obstacles
                    self.past_obstacle_observations.append(self.get_vector_obs(obs_per_link, info))
                    # concatenate latest observations
                    obstacle_obs = np.concatenate(self.past_obstacle_observations[-3:])
                case "vectors+closest_per_link":
                    # Concatenates unit vectors pointing to the closest obstacles and closest obstacle per link
                    closest_dist_vectors_per_link = self.get_vector_obs(obs_per_link, info)
                    closest_dist_per_link = np.array([min(i) for i in obs_per_link.values()])
                    obstacle_obs = np.concatenate((closest_dist_per_link, closest_dist_vectors_per_link))

        prior_action = np.empty(0)
        if self.config.task_observations["prior"] == "rrmc_neo":
            prior_action = self.robot.compute_action_neo(
                self.goal,
                self.obstacles,
                self.collision_detector,
                self.robot.inverse_kinematics(link=11, position=self.goal)[:7],
                'default')

        observations = np.concatenate((obstacle_obs, prior_action))

        return observations

    def get_vector_obs(self, obs_per_link, info):
        closest_points = list(info["closest_points"].values())
        obs_distances = list(obs_per_link.values())
        closest_per_obstacle = [sorted(zip(i, y)) for i, y in zip(obs_distances, closest_points)]
        closest_pairs = np.array([x[0][1] for x in closest_per_obstacle])
        # get array of unit vectors pointing to the closest obstacles
        closest_distances = np.array([unit_vector(i[0], i[1]) for i in closest_pairs]).flatten()

        if self.config.debug_collision:
            # visualize shortest distances

            for pair in closest_pairs:
                x = pair[0]
                y = pair[1]
                p.addUserDebugLine(x, y, lifeTime=0.1, lineWidth=1, physicsClientId=0)

        return closest_distances

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal_reached = False
        self.is_collided = False

        if self.random_size_cuboids:
            # remove all cuboids
            removed_cuboids = []
            obstacles = self.obstacles.copy()
            for obstacle in obstacles.keys():
                if "cuboid" in obstacle:
                    removed_cuboids.append(self.sim.remove_body(obstacle))
                    self.obstacles.pop(obstacle)

            if removed_cuboids:
                self.add_random_size_cuboids(removed_cuboids)

        if self.config.fixed_target is None:
            self.set_coll_free_goal(["table", "robot"])
        # sample new (collision free) obstacles
        if self.randomize_obstacle_position:
            self.set_coll_free_obs(0.03)
        else:
            coll_obj = [i for i in self.obstacles.keys()]
            coll_obj.extend(["table", "robot"])
            self.set_coll_free_goal(coll_obj, margin=0.03)

        if self.randomize_robot_pose:
            self.set_coll_free_robot()

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

        # empty past obstacle observations
        self.past_obstacle_observations = []
        # fill past obstacle observations with observation
        if self.obstacle_obs == "vectors+past":
            for _ in range(2):
                self.get_obs()

    def set_coll_free_robot(self):
        # make sure that robot is not colliding
        margin = 0.05
        for obstacle in self.obstacles.keys():
            if obstacle == "table":
                continue
            # get collision free obstacle position
            collided = self.check_collision("robot", obstacle, margin=margin)

            # sample new obstacle position until it doesn't obstruct robot
            i = 0
            while collided:
                try:
                    if i > 9999:
                        raise StopIteration("Couldn't find collision free robot pose!")
                    else:
                        i += 1
                except StopIteration as e:
                    # set neutral pose
                    self.robot.set_joint_neutral()
                    print(e)
                    break

                # generate new robot pose
                self.robot.reset()
                collided = self.check_collision("robot", obstacle, margin=margin)

    def set_random_num_obs(self):
        keys = list(self.obstacles.keys())

        if "table" in keys:
            keys.remove("table")

        rand_num_obs = self.np_random.integers(self.sample_size_obs[0], self.sample_size_obs[1])
        obs_to_move = abs(rand_num_obs - len(keys))
        # shuffle obstacle order
        keys = list(self.obstacles.keys())
        self.np_random.shuffle(keys)

        for obstacle in keys:
            if obs_to_move <= 0:
                break
            # move obstacle far away from work space
            self.sim.set_base_pose(obstacle, np.array([99.9, 99.9, -99.9]), np.array([0.0, 0.0, 0.0, 1.0]))
            # self.sim.set_base_pose_dummy(self.dummy_obstacle_id[obstacle_id], np.array([99.9, 99.9, -99.9]),
            #                        np.array([0.0, 0.0, 0.0, 1.0]),
            #                        physics_client=self.sim.dummy_collision_client)
            obs_to_move -= 1

    def add_random_size_cuboids(self, cuboids: list):
        high = 0.2
        for cuboid in cuboids:
            fractions = self.np_random.dirichlet(alpha=[1, 1, 1], size=1)[0]
            shape = [high * fraction for fraction in fractions]
            self.create_obstacle_cuboid(name=cuboid, size=np.array(shape))

    def set_random_obs_velocity(self):
        for obstacle in self.obstacles.keys():
            obstacle_id = self.obstacles[obstacle]
            velocity = self.np_random.uniform(self.velocity_range_low, self.velocity_range_high)
            self.sim.set_base_velocity(obstacle, velocity=velocity)
            # self.sim.set_base_velocity_dummy(self.dummy_obstacle_id[obstacle_id],
            #                                  velocity=velocity,
            #                                  physics_client=self.sim.dummy_collision_client
            #                                  )

    def set_coll_free_goal(self, coll_obj, margin=0.1):
        collision = [True]
        # get collision free goal within goal space
        i = 0
        while any(collision):
            self.goal = self._sample_goal()
            try:
                if i > 9999:
                    raise StopIteration("Couldn't find collision free goal!")
                else:
                    i += 1
            except StopIteration as e:
                # set goal at ee position
                self.goal = np.array(self.get_ee_position()) # in case of failure, simply set goal to ee position
                print(e)
                break

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

    def set_coll_free_obs(self, margin=0.1):
        # get randomized obstacle position within goal space
        for obstacle in self.obstacles.keys():
            if obstacle == "table":
                continue
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
                collision = [self.check_collision("robot", obstacle, margin=margin + self.config.safety_distance),
                             self.check_collision("table", obstacle, margin=margin),
                             self.check_collision("dummy_sphere", obstacle, margin=margin)]
                if not self.allow_overlapping_obstacles:
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

    def sample_within_hollow_sphere(self, radius_minor, radius_major, upper_half_only=False, front_half_only=False,
                                    three_quarter_front_half_only=False) -> np.ndarray:

        phi = self.np_random.uniform(0, 2 * np.pi)
        if upper_half_only:
            theta = self.np_random.uniform(0, 0.5 * np.pi)
        else:
            theta = self.np_random.uniform(0, np.pi)

        if front_half_only:
            phi = self.np_random.uniform(-0.5 * np.pi, 0.5 * np.pi)

        if three_quarter_front_half_only:
            phi = self.np_random.uniform(-0.75 * np.pi, 0.75 * np.pi)

        r = np.cbrt(self.np_random.uniform(radius_minor ** 3, radius_major ** 3))

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        sample = np.array([x, y, z])

        return sample

    def sample_inside_torus(self, R=0.5, r=0.05, front_half_only=False):
        """
        R : float
            Distance from the center of the tube to the center of the torus.
        r : float
            Radius of the tube (i.e., cross-sectional radius of the torus).
        """

        # Generate random angles theta and phi
        theta = 2.0 * np.pi * self.np_random.random()
        if front_half_only:
            theta = self.np_random.uniform(-0.5 * np.pi, 0.5 * np.pi)

        phi = 2.0 * np.pi * self.np_random.random()

        # Generate a random radius
        rad = r * np.sqrt(self.np_random.random())

        # Convert to Cartesian coordinates
        x = (R + rad * np.cos(phi)) * np.cos(theta)
        y = (R + rad * np.cos(phi)) * np.sin(theta)
        z = rad * np.sin(phi)

        return np.array([x, y, z + 0.5])

    def set_random_robot_base(self):
        self.robot.neutral_joint_values[0] = self.np_random.uniform(self.robot.joint_lim_min[0],
                                                                    self.robot.joint_lim_max[0])
        self.robot.set_joint_neutral()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        # if self.config.reward_type == "sparse":
        #     return np.array(d < self.distance_threshold, dtype=bool)
        # else:
        # if self.config.goal_condition == "reach":
        #     return np.array(d < self.distance_threshold, dtype=bool)
        # else:
        #
        #     return np.array(np.logical_and((d < self.distance_threshold), (ee_speed < self.ee_speed_threshold)), dtype=bool)
        if self.config.goal_condition == "halt":
            if not self.goal_reached:
                ee_speed = np.linalg.norm(self.robot.get_ee_velocity())
                self.goal_reached = np.array(
                    np.logical_and((d < self.ee_error_threshold), (ee_speed < self.ee_speed_threshold)), dtype=bool)

        else:
            return np.array(d < self.ee_error_threshold, dtype=bool)
        return np.array(self.goal_reached, dtype=bool)

    def is_truncated(self) -> np.ndarray:
        return np.array(self.is_collided, dtype=bool)

    def update_labels(self, manipulability, distance, obstacle_distances, action_difference, effort, jerk, reward):
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
                                           f"{self.debug_effort_base_text} {np.round(effort, 5)}")
                self.sim.create_debug_text(self.debug_jerk_label_name,
                                           f"{self.debug_jerk_base_text} {np.round(jerk, 5)}")
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

        return np.linalg.norm(self.robot.current_joint_acceleration)

    def get_norm_jerk(self):
        return np.linalg.norm(self.robot.current_joint_jerk)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        ee_error = distance(achieved_goal, desired_goal)
        ee_speed = np.linalg.norm(self.robot.get_ee_velocity())
        manipulability = self.robot.get_manipulability()
        obs_distance = self.distances_closest_obstacles
        action_diff = self.get_norm_action_difference()
        effort = self.get_norm_effort()
        jerk = self.get_norm_jerk()

        if self.config.reward_type == "sparse":
            if self.config.goal_condition == "reach":
                ee_error += self.is_collided  # if collided and reached goal, don't give reward
                reward = -1 + np.array((ee_error < self.ee_error_threshold), dtype=np.float32)
            else:
                reward = np.array(1 - np.logical_and((ee_error < self.ee_error_threshold),
                                                     (ee_speed < self.ee_speed_threshold)), dtype=np.float32)

        elif self.config.reward_type == "wang":
            weight_distance = 10e-3
            weight_obs = 0.1
            tolerance_distance = 10e-4
            # IPPO reward
            distance_reward = weight_distance * np.square(ee_error) + np.log(
                np.square(ee_error) + tolerance_distance)
            obstacle_reward = weight_obs * np.sum([np.max([0, 1 - d / 0.05]) for d in obs_distance])
            reward = -np.array(distance_reward + obstacle_reward, dtype=np.float32)
        elif self.config.reward_type == "kumar_her":
            if self.config.goal_condition == "reach":
                reward = -np.array((ee_error > self.ee_error_threshold) * jerk, dtype=np.float32)
            else:
                reward = np.array(np.logical_and((ee_error < self.ee_error_threshold),
                                                 (ee_speed < self.ee_speed_threshold)), dtype=np.float32)
                reward -= jerk
        elif self.config.reward_type == "kumar_optim":
            weight_effort = 0.005
            reward = -np.array((ee_error > self.ee_error_threshold), dtype=np.float32)
            reward -= effort

        elif self.config.reward_type == "kumar":
            weight_distance = 20
            weight_effort = 0.005
            weight_obs = 0.1

            distance_reward = np.exp(-weight_distance * np.square(ee_error))
            effort_reward = - weight_effort * effort
            obstacle_reward = - weight_obs * np.sum([np.max([0, 1 - d / 0.05]) for d in obs_distance])

            # finish_reward =  1000 * np.array(np.logical_and((ee_error < self.distance_threshold), (ee_speed < self.ee_speed_threshold)), dtype=np.float32)
            # speed_reward = np.array(ee_error < self.distance_threshold, dtype=np.float32) * np.exp(-weight_distance * np.square(ee_speed))
            # print(speed_reward)

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

        if self.sim.render_env and self.config.show_debug_labels:
            self.update_labels(manipulability, ee_error, obs_distance, action_diff, effort, jerk, reward)

        if self.truncate_episode_on_collision and self.config.reward_type in ["sparse", "kumar_her", "kumar_optim"]:
            reward += self.is_collided * self.config.collision_reward

        self.action_diff = action_diff
        self.action_magnitude = effort
        self.manipulability = manipulability

        return reward

    def visualize_space_from_range(self, range_low, range_high, color=np.array([0.0, 1.0, 0.0, 0.2])):
        x = (range_high[0] - range_low[0])
        y = (range_high[1] - range_low[1])
        z = (range_high[2] - range_low[2])

        self.sim.create_box(
            body_name=f"goal_space",
            half_extents=np.array([x / 2, y / 2, z / 2]),
            ghost=True,
            mass=0.0,
            position=np.array(
                [range_high[0] - x / 2, range_high[1] - y / 2, range_high[2] - z / 2]),
            rgba_color=color
        )

        self.visualize_outline_from_range(range_low, range_high, color)

    def visualize_outline_from_range(self, range_low, range_high, color):
        d = [range_low, range_high,
             np.array([range_low[0], range_high[1], range_high[2]]),
             np.array([range_low[0], range_low[1], range_high[2]]),
             np.array([range_low[0], range_high[1], range_low[2]]),
             np.array([range_high[0], range_low[1], range_high[2]]),
             np.array([range_high[0], range_low[1], range_low[2]]),
             np.array([range_high[0], range_high[1], range_low[2]]),
             np.array([range_high[0], range_low[1], range_high[2]])]

        for dot in d:
            for dot2 in d:
                intersection_count = sum([dot[0] == dot2[0], dot[1] == dot2[1], dot[2] == dot2[2]])
                if intersection_count >= 2:
                    self.sim.create_debug_line(dot, dot2, id=self.sim_id, color=color[:3])

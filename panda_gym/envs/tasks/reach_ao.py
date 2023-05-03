import logging
import pathlib
import random
from math import exp
from typing import Any, Dict

import numpy as np
import pybullet
from pathlib import Path

import gymnasium as gym

from panda_gym.envs.core import Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.utils import distance
from pyb_utils.collision import NamedCollisionObject, CollisionDetector
import pybullet as p
import sys
import gymnasium
import itertools

sys.modules["gym"] = gymnasium

from stable_baselines3 import TD3

ROOT_DIR = Path(__file__).parent.parent.parent
SCENARIO_DIR = f"{ROOT_DIR}\\assets\\scenarios"
NARROW_TUNNEL_DIR = f"{SCENARIO_DIR}\\narrow_tunnel"
LIBRARY_DIR = f"{SCENARIO_DIR}\\library"


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
            show_debug_labels=True,
            fixed_target=None,
            # adjustment factors for dense rewards
            factor_punish_distance=10.0,  # def: 10.0
            factor_punish_collision=1.0,  # def: 1.0
            factor_punish_action_magnitude=0.000,  # def: 0.005
            factor_punish_action_difference_magnitude=0.0,  # def: ?
            factor_punish_obstacle_proximity=0.0  # def: ?

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

        # dense reward configuration
        self.factor_punish_distance = factor_punish_distance
        self.factor_punish_collision = factor_punish_collision
        self.factor_punish_action_magnitude = factor_punish_action_magnitude
        self.factor_punish_action_difference_magnitude = factor_punish_action_difference_magnitude
        self.factor_punish_obstacle_proximity = factor_punish_obstacle_proximity

        # if target is fixed, it won't be randomly sampled on each episode
        self.fixed_target = fixed_target

        self.scenario = scenario

        # self.robot_params = self.create_robot_debug_params()
        self.cube_size_medium = np.array([0.03, 0.03, 0.03])
        self.cube_size_small = np.array([0.02, 0.02, 0.02])
        self.cube_size_mini = np.array([0.01, 0.01, 0.01])

        self.reward_type = reward_type
        self.distance_threshold = goal_distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range = 0.3
        self.obstacle_count = 0

        self._sample_goal = self.sample_from_goal_range
        self._sample_obstacle = self.sample_from_goal_range
        # set default goal range
        self.x_offset = 0.6
        self.goal_range_low = np.array([-self.goal_range / 2.5 +self.x_offset, -self.goal_range / 1.5, 0])
        self.goal_range_high = np.array([self.goal_range / 2.5 +self.x_offset, self.goal_range / 1.5, self.goal_range])

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
        self.sample_size_obs = [0,0]

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
        for link_name in self.collision_links:
            link = NamedCollisionObject("robot", link_name=link_name)
            for obstacle in self.collision_objects:
                self.named_collision_pairs_rob_obs.append((link, obstacle))
        self.collision_detector = CollisionDetector(col_id=self.sim_id, bodies=self.bodies,
                                                    named_collision_pairs=self.named_collision_pairs_rob_obs)

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
            self.debug_action_magnitude_label_name = "action_magnitude"
            self.debug_action_magnitude_base_text = "Action Magnitude:"

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
            case scenario if "cube" in scenario.split(sep="_"): found_scenario = self.create_scenario_cube
            case scenario if "wang" in scenario.split(sep="_"): found_scenario = self.create_scenario_wang
            case "narrow_tunnel": found_scenario = self.create_scenario_narrow_tunnel
            case "library": found_scenario = self.create_scenario_library
            case "library1": found_scenario = self.create_scenario_library
            case "library2": found_scenario = self.create_scenario_library
            case "ga_training": found_scenario = self.scenario_ga_training

        if not found_scenario:
            logging.warning("Scenario not found. Defaulting to narrow_tunnel.")
            found_scenario = self.create_scenario_library

        return found_scenario

    def _create_scene(self):

        self.sim.create_plane(z_offset=-0.4)
        #self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=0.3)
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
        # todo: Create custom goal space

        self.robot.neutral_joint_values = np.array(
            [-2.34477029,  1.69617261,  1.81619755, - 1.98816377, - 1.58805049,  1.2963265,
             0.41092735]
            )
        self.robot.set_joint_neutral()
        self.fixed_target = np.array([0.4, 0.15, 0.2])
        self.goal = self.fixed_target
        urdfs = {
            "tunnel": {
                "bodyName": "narrow_tunnel",
                "fileName": f"{NARROW_TUNNEL_DIR}\\narrow_tunnel.urdf",
                "basePosition": [0.8, -0.15, 0.0],
                "useFixedBase": True
            }
        }

        indexes = self.sim.load_scenario(urdfs)

        for idx, body in zip(indexes, [urdfs["tunnel"]]):
            name = body["bodyName"]
            self.obstacles[f"{name}_{len(self.obstacles)}"] = idx

    def create_scenario_library(self):


        self.robot.neutral_joint_values = [0.0, 0.12001979, 0.0, -1.64029458, 0.02081271, 3.1,
                                           0.77979846] # above table
        #self.fixed_target = np.array([0.7, 0.0, 0.2]) # below table
        #self.goal = self.fixed_target
        urdfs = {
            "shelf": {
                "bodyName": "shelf",
                "fileName": f"{LIBRARY_DIR}\\shelf.urdf",
                "basePosition": [-1.0, -0.5, 0.0], #v1: [-0.8,-0.6,0.0]
                "useFixedBase": True
            },
            "table": {
                "bodyName": "table",
                "fileName": f"{LIBRARY_DIR}\\table.urdf",
                "basePosition": [-0.7, -0.45, 0.1], #v1: [-0.7,-0.6,0.1]
                "useFixedBase": True
            },
        }
        self.sim.load_scenario(urdfs)


        indexes = self.sim.load_scenario(urdfs)

        for idx, body in zip(indexes, [urdfs["shelf"], urdfs["table"]]):
            name = body["bodyName"]
            self.obstacles[f"{name}_{len(self.obstacles)}"] = idx

        # Create custom goal space
        if self.scenario == "library1":
            self.goal_range_low = np.array([0.2, -0.3, 0])
            self.goal_range_high = np.array([0.7, 0.3, 0.6])
        elif self.scenario == "library2":
            # todo: expand goal range into shelf
            self.goal_range_low = np.array([-0.7, -0.4, 0.35])
            self.goal_range_high = np.array([-0.4, 0.4, 0.9])
        else:
            # todo: expand goal range into shelf
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

    def create_stage_wall_parkour_1(self):
        """wall parkour."""
        self.goal_range = 0.4

        self.create_obstacle_cuboid(
            np.array([0.0, -0.05, 0.1]),
            size=np.array([0.1, 0.001, 0.1]))

        # self.create_obstacle_cuboid(
        #     np.array([0.2, 0.0, 0.25]),
        #     size=np.array([0.1, 0.4, 0.001]))
        # self.create_obstacle_cuboid(
        #     np.array([0.2, 0.0, 0.25]),
        #     size=np.array([0.1, 0.4, 0.001]))

    def create_stage_box_3(self):
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
        self.sample_size_obs = [1,3]
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
        ee_target = self.sample_random_joint_position_within_workspace()#self.sample_sphere(0.3, 0.6, upper_half=True)
        joint_positions = self.robot.inverse_kinematics(link=11, position=ee_target)[:7]
        self.robot.set_joint_angles(joint_positions)


    def create_scenario_wang(self):
        def sample_wang_obstacle():

            # if np.random.rand() > 0.5:
                # sample near goal
                sample = self.sample_sphere(0.1,0.4)
                return sample + self.goal
            # else:
            #     # sample near base
            #     sample = self.sample_sphere(0.3,0.5, True)
            #     return sample + self.robot.get_link_position(0)
        num_spheres = int(self.scenario.split(sep="_")[1])

        def sample_wang_goal():
            return self.sample_sphere(0.4,0.95, upper_half=True)



        self._sample_obstacle = sample_wang_obstacle
        self._sample_goal = sample_wang_goal
        self.robot.reset = self.set_robot_random_joint_position

        self.randomize_obstacle_position = True
        self.random_num_obs = False

        for i in range(num_spheres):
            self.create_obstacle_sphere(radius=0.05)

    def sample_from_robot_workspace(self):
        return self.sample_random_joint_position_within_workspace()

    def scenario_ga_training(self):
        """1-3 randomly initialized Spheres with 8cm radius"""
        self._sample_obstacle = self._sample_goal = self.sample_from_robot_workspace

        self.randomize_obstacle_position = True
        self.random_num_obs = False
        self.sample_size_obs = [1,3]

        # Create 3 spherical obstacles
        for i in range(3):
            self.create_obstacle_sphere(radius=0.08)

    def create_stage_sphere_2(self):
        """2  spheres"""
        self.goal_range = 0.3
        self.fixed_target = np.array([-0.2, -0.4, 0.2])
        self.goal = self.fixed_target

        self.create_obstacle_sphere(
            radius=0.05,
            position=np.array([0.0, -0.2, 0.1])
        )

        self.create_obstacle_sphere(
            radius=0.05,
            position=np.array([0.0, -0.2, 0.2])
        )

    def create_stage_sphere_2_random(self):
        """2 randomly placed spheres"""
        self.goal_range = 0.4
        self.randomize_obstacle_position = True

        self.create_obstacle_sphere(
            radius=0.05
        )

        self.create_obstacle_sphere(
            radius=0.05
        )

    def create_stage_neo_test_1(self):
        self.goal_range = 0.4
        self.create_obstacle_sphere(radius=0.1, position=np.array([-0.1, -0.2, 0.25]))

    def create_stage_neo_test_2(self):
        """edge case, where NEO fails..."""
        self.goal_range = 0.4
        self.fixed_target = np.array([-1.0, 0.0, 0.3])
        self.goal = self.fixed_target
        self.create_obstacle_sphere(radius=0.0001, position=np.array([-0.1, -0.2, -0.25]))

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
        if self.obstacles:
            q = self.robot.get_joint_angles(self.robot.joint_indices[:7])
            obs_per_link = self.collision_detector.compute_distances_per_link(q, self.robot.joint_indices[:7],
                                                                              max_distance=10.0)
            # obs_per_link = {}
            # links = self.robot.get_rtb_links()
            # for link in links:
            #     link_obs = []
            #     for obstacle in self.dummy_obstacles.values():
            #
            #         for coll in link.collision:
            #             coll_distance = compute_distance(coll.co, obstacle, 1, 5.0)[0]
            #             if coll_distance is not None:
            #                 link_obs.append(coll_distance)
            #     obs_per_link[link] = link_obs

            if self.joint_obstacle_observation == "all":
                self.distances_links_to_closest_obstacle = np.array([min(i) for i in obs_per_link.values()])
            elif self.joint_obstacle_observation == "closest":
                self.distances_links_to_closest_obstacle = min(obs_per_link.values())

            self.is_collided = min(self.distances_links_to_closest_obstacle) <= 0.0

        return self.distances_links_to_closest_obstacle

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        # todo: set goal first, then obstacles
        # todo: make function to set obstacles around goal or manipulator
        with self.sim.no_rendering():
            if self.fixed_target is None:
                self.set_coll_free_goal()
            # sample new (collision free) obstacles
            if self.randomize_obstacle_position:
                self.set_coll_free_obs()

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
        for obstacle in self.obstacles.keys():
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

    def set_coll_free_goal(self):
        collision = [True]
        # get collision free goal within goal space
        i= 0
        while any(collision):
            self.goal = self._sample_goal()
            if i > 9999:
                raise StopIteration("Couldn't find collision free goal!")
            else: i += 1

            self.sim.set_base_pose("dummy_sphere", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
            #self.sim.physics_client.performCollisionDetection()
            collision = [self.check_collision("dummy_sphere", "robot", margin=0.05),
                         self.check_collision("dummy_sphere", "table", margin=0.05)]

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
                else: i += 1
                pos = self._sample_obstacle()
                self.sim.set_base_pose(obstacle, pos, np.array([0.0, 0.0, 0.0, 1.0]))
                collision = [self.check_collision("robot", obstacle, margin=0.05),
                             self.check_collision("table", obstacle, margin=0.05),
                             self.check_collision("dummy_sphere", obstacle, margin=0.05)]
                # check if out of boundaries
                self.sim.set_base_pose("dummy_sphere", np.array([0.0, 0.0, 0.0]),
                                       np.array([0.0, 0.0, 0.0, 1.0]))  # move dummy to origin
                collision.append(not self.check_collision("dummy_sphere", obstacle, margin=1.2))
        self.sim.set_base_pose("dummy_sphere", np.array([0.0, 0.0, -5.0]),
                               np.array([0.0, 0.0, 0.0, 1.0]))  # move dummy away

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

    def sample_sphere(self, radius_minor, radius_major, upper_half=False) -> np.ndarray:

        phi = np.random.uniform(0, 2 * np.pi)
        if upper_half:
            theta = np.random.uniform(0, 0.5*np.pi)
        else:
            theta = np.random.uniform(0, np.pi)

        r = np.cbrt(np.random.uniform(radius_minor ** 3, radius_major ** 3))

        sample = np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])

        return sample


    def sample_around_end_effector(self):
        # todo:
        pass



    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def is_truncated(self) -> np.ndarray:
        return np.array(self.is_collided, dtype=np.bool8)

    def update_labels(self, manipulability, distance, obstacle_distances, action_difference, action_magnitude, reward):
        with self.sim.no_rendering():
            #self.sim.remove_all_debug_text()
            # self.sim.remove_debug_text(self.debug_dist_label_name)
            # self.sim.remove_debug_text(self.debug_manip_label_name)

            try:
                self.sim.create_debug_text(self.debug_dist_label_name,
                                           f"{self.debug_dist_label_base_text} {round(distance, 3)}")
                self.sim.create_debug_text(self.debug_manip_label_name,
                                           f"{self.debug_manip_label_base_text} {round(manipulability, 5)}")
                self.sim.create_debug_text(self.debug_obs_label_name,
                                           f"{self.debug_obs_label_base_text} {round(min(obstacle_distances), 5)}")
                self.sim.create_debug_text(self.debug_action_difference_label_name,
                                           f"{self.debug_action_difference_base_text} {round(action_difference, 5)}")
                self.sim.create_debug_text(self.debug_action_magnitude_label_name,
                                           f"{self.debug_action_magnitude_base_text} {round(action_magnitude, 5)}")
                self.sim.create_debug_text(self.debug_reward_label_name,
                                           f"{self.debug_reward_base_text} {round(reward, 5)}")
                pass
            except BaseException:
                pass

    def get_reward_action_difference(self) -> float:
        """Calculate the magnitude of deviation between the recent and the previous action."""
        if self.robot.previous_action is None:
            # There has not been any recorded action yet
            return 0.0

        action_diff = self.robot.recent_action - self.robot.previous_action
        return np.square(np.linalg.norm(action_diff))

    def get_reward_small_actions(self) -> float:
        """Calculate the magnitude of the action, i.e. calculate the square of the norm of the action."""
        if self.robot.recent_action is None:
            # There has not been any recorded action yet
            return 0.0

        return np.square(np.linalg.norm(self.robot.recent_action))

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        manipulability = self.robot.get_manipulability()
        obs_d = self.distances_links_to_closest_obstacle
        action_diff = self.get_reward_action_difference()
        action_magnitude = self.get_reward_small_actions()

        if self.reward_type == "sparse":
            reward = -np.array((d > self.distance_threshold) + (min(obs_d) <= 0) * 100, dtype=np.float32)
        else:
            # calculate dense rewards

            reward = -np.array(action_diff * self.factor_punish_action_difference_magnitude +
                               action_magnitude * self.factor_punish_action_magnitude, dtype=np.float32)

            collision = min(obs_d) <= 0.0

            d *= self.factor_punish_distance
            if collision:
                reward += -np.array(d + 100 * self.factor_punish_collision, dtype=np.float32)
            else:
                reward += -np.array(d + exp(-min(obs_d)) * self.factor_punish_obstacle_proximity, dtype=np.float32)

        if self.sim.render_env and self.show_debug_labels:
            self.update_labels(manipulability, d, obs_d, action_diff, action_magnitude, reward)

        self.action_diff = action_diff
        self.action_magnitude = action_magnitude
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




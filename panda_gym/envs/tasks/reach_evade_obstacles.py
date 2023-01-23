import pathlib
from math import exp
from typing import Any, Dict

import numpy as np
import pybullet

from panda_gym.envs.core import Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.utils import distance
from pyb_utils.collision import NamedCollisionObject, CollisionDetector, compute_distance
import pybullet as p
import gymnasium as gym
from stable_baselines3 import TD3


class ReachEvadeObstacles(Task):
    def __init__(
            self,
            sim,
            robot,
            get_ee_position,
            reward_type="sparse",
            goal_distance_threshold=0.05,
            show_goal_space=False,
            joint_obstacle_observation="all",
            obstacle_layout=1,
            show_debug_labels=True,
            fixed_target=None,
            # adjustement factors for dense rewards
            factor_punish_distance=10.0,  # def: 10.0
            factor_punish_collision=1.0,  # def: 1.0
            factor_punish_action_magnitude=0.000,  # def: 0.005
            factor_punish_action_difference_magnitude=0.0,  # def: ?
            factor_punish_obstacle_proximity=0.0  # def: ?

    ) -> None:
        super().__init__(sim)
        self.sim_id = self.sim.physics_client._client
        # self.dummy_sim_id = self.sim.dummy_collision_client._client

        self.robot: Panda = robot
        self.obstacles = {}
        self.dummy_obstacles = {}
        self.dummy_obstacle_id = {}
        self.joint_obstacle_observation = joint_obstacle_observation

        # dense reward configuration
        self.factor_punish_distance = factor_punish_distance
        self.factor_punish_collision = factor_punish_collision
        self.factor_punish_action_magnitude = factor_punish_action_magnitude
        self.factor_punish_action_difference_magnitude = factor_punish_action_difference_magnitude
        self.factor_punish_obstacle_proximity = factor_punish_obstacle_proximity

        # if target is fixed, it won't be randomly sampled on each episode
        self.fixed_target = fixed_target

        create_obstacle_layout = {
            1: self.create_stage_1,
            2: self.create_stage_2,
            3: self.create_stage_3,
            "shelf_1": self.create_stage_shelf_1,
            "wall_parkour_1": self.create_stage_wall_parkour_1,
            "box_3": self.create_stage_box_3,

            # for curriculum learning
            "cube_1": self.create_stage_cube_1,
            "cube_1_random": self.create_stage_cube_1_random,
            "cube_2": self.create_stage_cube_2,
            "cube_2_random": self.create_stage_cube_2_random,
            "cube_3_random": self.create_stage_cube_3_random,
            "neo_test_1": self.create_stage_neo_test_1,
            "neo_test_2": self.create_stage_neo_test_2,
            "sphere_2": self.create_stage_sphere_2,
            "sphere_2_random": self.create_stage_sphere_2_random,
            # "cube_6": self.create_stage_cube_6,

        }
        # self.robot_params = self.create_robot_debug_params()
        self.cube_size_small = np.array([0.02, 0.02, 0.02])
        self.cube_size_mini = np.array([0.01, 0.01, 0.01])

        self.reward_type = reward_type
        self.distance_threshold = goal_distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range = 0.3
        self.obstacle_count = 0

        # # init pose generator env
        self.reach_checker_env = gym.make("PandaReachChecker", control_type = "js", render=True)

        # # load pose checker model
        path = pathlib.Path(__file__).parent.resolve()
        self.pose_generator_model = TD3.load(fr"{path}/pose_generator_model.zip",
                                             env=self.reach_checker_env)

        self.bodies = {"robot": self.robot.id}

        exclude_links = ["panda_grasptarget", "panda_leftfinger", "panda_rightfinger"]  # env has no grasptarget
        self.collision_links = [i for i in self.robot.link_names if i not in exclude_links]
        self.collision_objects = []
        self.randomize = False  # Randomize obstacle placement

        # extra observations
        self.distances_links_to_closest_obstacle = np.zeros(len(self.collision_links))
        self.is_collided = False

        # set scene
        with self.sim.no_rendering():
            self._create_scene()

            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

            if obstacle_layout:
                create_obstacle_layout[obstacle_layout]()

                # register collision objects for collision detector
                for obstacle_name, obstacle_id in self.obstacles.items():
                    self.collision_objects.append(NamedCollisionObject(obstacle_name))
                    self.bodies[obstacle_name] = obstacle_id

            # set goal range
            self.goal_range_low = np.array([-self.goal_range / 2.5, -self.goal_range / 1.5, 0])
            self.goal_range_high = np.array([self.goal_range / 2.5, self.goal_range / 1.5, self.goal_range])

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

        if show_debug_labels:
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

    def _create_scene(self):

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.8]),
        )

        self.bodies["dummy_target"] = self.sim.create_sphere(
            body_name="dummy_target",
            radius=0.05,  # dummy target is intentionally bigger to ensure safety distance to obstacle
            mass=0.0,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.0]),
        )

    def create_stage_1(self):
        """one obstacle in the corner of the goal space, small goal space"""
        self.goal_range = 0.3

        self.create_obstacle_cuboid(
            np.array([0, 0.05, 0.15]),
            size=np.array([0.02, 0.02, 0.02]))

    def create_stage_2(self):
        """Two obstacles surrounding end effector, small goal space"""
        self.goal_range = 0.3

        spacing = 3.3
        spacing_x = 5
        x_fix = -0.025
        y_fix = -0.15
        z_fix = 0.15
        for x in range(1):
            for y in range(2):
                for z in range(1):
                    self.create_obstacle_cuboid(
                        np.array([x / spacing_x + x_fix, y / spacing + y_fix, z / spacing + z_fix]),
                        size=np.array([0.02, 0.02, 0.02]))

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

    def create_stage_cube_1(self):
        """1 small cube near the ee. Harder to ignore."""
        self.goal_range = 0.4

        self.create_obstacle_cuboid(
            np.array([0.05, 0.1, 0.15]),
            size=self.cube_size_small)

    def create_stage_cube_2(self):
        """2 small cubes near the ee. Hard to ignore."""
        self.goal_range = 0.5

        self.create_obstacle_cuboid(
            np.array([0.05, 0.1, 0.15]),
            size=self.cube_size_small)

        self.create_obstacle_cuboid(
            np.array([-0.05, -0.1, 0.25]),
            size=self.cube_size_small)

    def create_stage_cube_1_random(self):
        """2 random small cubes. Annoying."""
        self.goal_range = 0.4
        self.randomize = True

        self.create_obstacle_cuboid(
            np.array([-0.05, -0.08, 0.5]),
            size=self.cube_size_small)

    def create_stage_cube_2_random(self):
        """2 random small cubes. Annoying."""
        self.goal_range = 0.5
        self.randomize = True

        self.create_obstacle_cuboid(
            np.array([0.05, 0.08, 0.2]),
            size=self.cube_size_small)

        self.create_obstacle_cuboid(
            np.array([-0.05, -0.08, 0.5]),
            size=self.cube_size_small)

    def create_stage_cube_3_random(self):
        """3 random small cubes. Infuriating."""
        self.goal_range = 0.5
        self.randomize = True
        self.create_obstacle_cuboid(
            np.array([0.05, 0.08, 0.2]),
            size=self.cube_size_small)

        self.create_obstacle_cuboid(
            np.array([-0.05, -0.08, 0.5]),
            size=self.cube_size_small)

        self.create_obstacle_cuboid(
            np.array([-0.05, -0.08, 0.5]),
            size=self.cube_size_small)

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
        self.randomize = True

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

    def create_obstacle_sphere(self, position=np.array([0.1, 0, 0.1]), radius=0.02, alpha=0.8):
        obstacle_name = "obstacle"
        # position[0] += 0.6

        ids = []
        for physics_client in (self.sim.physics_client, self.sim.dummy_collision_client):
            ids.append(self.sim.create_sphere(
                body_name=f"{obstacle_name}_{len(self.obstacles)}",
                radius=radius,
                mass=0.0,
                position=position,
                rgba_color=np.array([0.5, 0, 0, alpha]),
                physics_client=physics_client
            ))
        self.obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = ids[0]
        self.dummy_obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = ids[1]
        self.dummy_obstacle_id[ids[0]] = ids[1]

    def create_obstacle_cuboid(self, position=np.array([0.1, 0, 0.1]), size=np.array([0.01, 0.01, 0.01])):
        obstacle_name = "obstacle"
        # position[0] += 0.6
        ids = []
        for physics_client in (self.sim.physics_client, self.sim.dummy_collision_client):
            ids.append(self.sim.create_box(
                body_name=f"{obstacle_name}_{len(self.obstacles)}",
                half_extents=size,
                mass=0.0,
                position=position,
                rgba_color=np.array([0.5, 0.5, 0.5, 1]),
                physics_client=physics_client
            ))

        self.obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = ids[0]
        self.dummy_obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = ids[1]
        self.dummy_obstacle_id[ids[0]] = ids[1]

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

    def get_obs(self) -> np.ndarray:
        if self.obstacles:
            q = self.robot.get_joint_angles(self.robot.joint_indices[:7])
            # obs_per_link = self.collision_detector.compute_distances_per_link(q, self.robot.joint_indices[:7],
            #                                                                   max_distance=10.0)
            obs_per_link = {}
            links = self.robot.get_rtb_links()
            for link in links:
                link_obs = []
                for obstacle in self.dummy_obstacles.values():

                    for coll in link.collision:
                        distance = compute_distance(coll.co, obstacle, 1, 5.0)[0]
                        if distance is not None:
                            link_obs.append(distance)
                obs_per_link[link] = link_obs

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
        # sample new (collision free) obstacles
        if self.fixed_target is None:
            self.goal = self._sample_goal()
            if self.randomize:
                # randomize obstacle position within goal space
                for obstacle in self.obstacles.keys():
                    # get collision free obstacle position
                    collision = True
                    pos = None
                    obstacle_id = self.obstacles[obstacle]
                    while collision:
                        pos = self._sample_goal()
                        self.sim.set_base_pose(obstacle, pos, np.array([0.0, 0.0, 0.0, 1.0]))
                        collision = self.get_collision("robot", obstacle, margin=0.05)
                    else:
                        if pos is not None:
                            self.sim.set_base_pose_dummy(self.dummy_obstacle_id[obstacle_id], pos,
                                                         np.array([0.0, 0.0, 0.0, 1.0]),
                                                         physics_client=self.sim.dummy_collision_client)

        collision = True
        # get collision free goal
        while collision and self.obstacles:

            self.sim.set_base_pose("dummy_target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.physics_client.performCollisionDetection()
            for obstacle in self.obstacles:
                collision = self.get_collision("dummy_target", obstacle)

                if collision:
                    self.goal = self._sample_goal()
                    break

        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("dummy_target", np.array([0.0, 0.0, -5.0]),
                               np.array([0.0, 0.0, 0.0, 1.0]))  # move dummy away

        # use rl policy to generate joint angles (and save final pose)
        arrived = False
        self.reach_checker_env.task.set_fixed_target(self.goal)
        obs, _ = self.reach_checker_env.reset()

        for i in range(50):
            action, _ = self.pose_generator_model.predict(obs)
            obs, reward, done, truncated, info, = self.reach_checker_env.step(action)
            if done and info["is_success"]:
                arrived = True
                break
            elif done:
                break

        if arrived:
            optimal_angles = self.reach_checker_env.robot.get_joint_angles(self.robot.joint_indices[:7])
            optimal_angles[6] = 0.0
            self.robot.optimal_pose = optimal_angles
        else:
            self.robot.optimal_pose = None

        # reset robot actions
        self.robot.recent_action = None
        self.robot.previous_action = None

    def get_collision(self, obstacle_1, obstacle_2, margin=0.0):
        """Check if given bodies collide."""
        # margin in which overlapping counts as a collision
        self.sim.physics_client.performCollisionDetection()
        closest_points = p.getClosestPoints(bodyA=self.bodies[obstacle_1], bodyB=self.bodies[obstacle_2],
                                            distance=10.0,
                                            physicsClientId=self.sim_id)
        contact_distance = np.min([pt[8] for pt in closest_points])
        collision = margin >= contact_distance
        return collision

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal[0] += 0.6
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def is_truncated(self) -> np.ndarray:
        return np.array(self.is_collided, dtype=np.bool8)

    def update_labels(self, manipulability, distance, obstacle_distances, action_difference, action_magnitude, reward):
        with self.sim.no_rendering():
            self.sim.remove_all_debug_text()
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
        manip = self.robot.get_manipulability()
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

        if self.sim.render_env:
            self.update_labels(manip, d, obs_d, action_diff, action_magnitude, reward)

        return reward

    def show_goal_space(self):
        self.sim.create_box(
            body_name="goal_space",
            ghost=True,
            half_extents=np.array([(self.goal_range_high[0] - self.goal_range_low[0]) / 2, self.goal_range_high[1],
                                   self.goal_range_high[2]]),
            mass=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rgba_color=np.array([0.0, 0.0, 0.5, 0.2]),
        )

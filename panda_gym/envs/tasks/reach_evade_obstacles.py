from math import exp
from typing import Any, Dict

import numpy as np
import pybullet

from panda_gym.envs.core import Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.utils import distance
from pyb_utils.collision import NamedCollisionObject, CollisionDetector
import pybullet as p


class ReachEvadeObstacles(Task):
    def __init__(
            self,
            sim,
            robot,
            get_ee_position,
            reward_type="sparse",
            distance_threshold=0.05,
            goal_range=0.3,
            show_goal_space=False,
            joint_obstacle_observation="all",
            obstacle_layout=1,
            show_debug_labels=True
    ) -> None:
        super().__init__(sim)
        self.sim_id = self.sim.physics_client._client

        self.robot: Panda = robot
        self.obstacles = {}
        self.joint_obstacle_observation = joint_obstacle_observation
        create_obstacle_layout = {
            1: self.create_stage_1,
            2: self.create_stage_2,
            3: self.create_stage_3,
            "shelf_1": self.create_stage_shelf_1,
            "wall_parkour_1": self.create_stage_wall_parkour_1,
            "box_1": self.create_stage_box_3
        }
        # self.robot_params = self.create_robot_debug_params()

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range = 0.3
        self.obstacle_count = 0

        self.bodies = {"robot": self.robot.id}
        exclude_links = ["panda_link8", "panda_grasptarget"]
        self.collision_links = [i for i in self.robot.link_names if i not in exclude_links]
        self.collision_objects = []

        # extra observations
        self.obs_d = np.zeros(len(self.collision_links))
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
            self.debug_obs_label_base_text = "Closest Obstacle Distance"

            self.sim.create_debug_text(self.debug_manip_label_name, f"{self.debug_manip_label_base_text} 0")
            self.sim.create_debug_text(self.debug_dist_label_name, f"{self.debug_dist_label_base_text} 0")
            self.sim.create_debug_text(self.debug_obs_label_name, f"{self.debug_obs_label_base_text} 0")

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
            radius=0.02,
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

    def create_obstacle_sphere(self, position=np.array([0.1, 0, 0.1]), radius=0.02, alpha=0.8):
        obstacle_name = "obstacle"

        obstacle_id = self.sim.create_sphere(
            body_name=f"{obstacle_name}_{len(self.obstacles)}",
            radius=radius,
            mass=0.0,
            position=position,
            rgba_color=np.array([0.5, 0, 0, alpha]),
        )
        self.obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = obstacle_id

    def create_obstacle_cuboid(self, position=np.array([0.1, 0, 0.1]), size=np.array([0.01, 0.01, 0.01])):
        obstacle_name = "obstacle"

        obstacle_id = self.sim.create_box(
            body_name=f"{obstacle_name}_{len(self.obstacles)}",
            half_extents=size,
            mass=0.0,
            position=position,
            rgba_color=np.array([0.5, 0.5, 0.5, 1]),
        )
        self.obstacles[f"{obstacle_name}_{len(self.obstacles)}"] = obstacle_id

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

    def get_obs(self) -> np.ndarray:
        if self.obstacles:
            q = self.robot.get_joint_angles(self.robot.joint_indices[:7])
            obs_per_link = self.collision_detector.compute_distances_per_link(q, self.robot.joint_indices[:7],
                                                                              max_distance=10.0)

            if self.joint_obstacle_observation == "all":
                self.obs_d = np.array([min(i) for i in obs_per_link.values()])
            elif self.joint_obstacle_observation == "closest":
                self.obs_d = min(obs_per_link.values())

            self.is_collided = min(self.obs_d) <= 0

        return self.obs_d

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        collision = True
        margin = 0.0  # margin in which overlapping counts as a collision

        self.goal = self._sample_goal()
        # get collision free goal
        while collision and self.obstacles:

            self.sim.set_base_pose("dummy_target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.physics_client.performCollisionDetection()
            for obstacle in self.obstacles:
                closest_points = p.getClosestPoints(bodyA=self.bodies["dummy_target"], bodyB=self.bodies[obstacle],
                                                    distance=10.0,
                                                    physicsClientId=self.sim_id)
                contact_distance = np.min([pt[8] for pt in closest_points])
                collision = margin >= contact_distance
                if collision:
                    self.goal = self._sample_goal()
                    break
            if not collision:
                collision = False

        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("dummy_target", np.array([0.0, 0.0, -5.0]),
                               np.array([0.0, 0.0, 0.0, 1.0]))  # move dummy away

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def is_truncated(self) -> np.ndarray:
        return np.array(self.is_collided, dtype=np.bool8)

    def update_labels(self, manipulability, distance, obstacle_distances):
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
            except BaseException:
                pass

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        manip = self.robot.get_manipulability()
        obs_d = self.obs_d

        if self.sim.render_env:
            self.update_labels(manip, d, obs_d)

        if self.reward_type == "sparse":
            return -np.array((d > self.distance_threshold) + (min(obs_d) <= 0) * 100, dtype=np.float32)
        else:
            collision = min(obs_d) <= 0
            if collision:
                return -np.array(d + 100, dtype=np.float32)
            else:
                return -np.array(d + exp(-min(obs_d)), dtype=np.float32)

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

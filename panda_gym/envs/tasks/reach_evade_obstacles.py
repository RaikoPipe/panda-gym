from math import exp
from typing import Any, Dict

import numpy as np
import pybullet

from panda_gym.envs.core import Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.utils import distance
from pyb_utils.collision import NamedCollisionObject, CollisionDetector
import pybullet as p


# todo: add collision detection
class ReachEvadeObstacles(Task):
    def __init__(
            self,
            sim,
            robot,
            get_ee_position,
            reward_type="sparse",
            distance_threshold=0.05,
            goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.sim_id = self.sim.physics_client._client

        self.robot: Panda = robot
        self.obstacles = {}
        # self.robot_params = self.create_robot_debug_params()

        # extra observations
        self.obs_d = None
        self.is_collided = False

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.obstacle_count = 0

        self.bodies = {"robot": self.robot.id}
        exclude_links = ["panda_link8", "panda_grasptarget"]
        self.collision_links = [i for i in self.robot.link_names if i not in exclude_links]
        self.collision_objects = []
        self.named_collision_pairs = []

        self.debug_manip_label_name = "manip"
        self.debug_manip_label_base_text = "Manipulability Score:"
        self.debug_dist_label_name = "dist"
        self.debug_dist_label_base_text = "Distance:"
        self.debug_obs_label_name = "obs"
        self.debug_obs_label_base_text = "Closest Obstacle Distance"

        self.sim.create_debug_text(self.debug_manip_label_name, f"{self.debug_manip_label_base_text} 0")
        self.sim.create_debug_text(self.debug_dist_label_name, f"{self.debug_dist_label_base_text} 0")
        self.sim.create_debug_text(self.debug_obs_label_name, f"{self.debug_obs_label_base_text} 0")

        with self.sim.no_rendering():
            self._create_scene()
            self.create_obstacle_layout_1()

            for link_name in self.collision_links:
                link = NamedCollisionObject("robot", link_name=link_name)
                for obstacle in self.collision_objects:
                    self.named_collision_pairs.append((link, obstacle))



            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
            self.collision_detector = CollisionDetector(col_id=self.sim_id, bodies=self.bodies,
                                                        named_collision_pairs=self.named_collision_pairs)

    def _create_scene(self):

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def create_obstacle_layout_1(self):
        spacing = 4
        spacing_x = 2
        x_fix = -0.25
        y_fix = -0.25
        z_fix = 0.2
        for x in range(2):
            for y in range(3):
                for z in range(1):
                    self.create_obstacle(np.array([x/spacing_x+x_fix, y/spacing+y_fix, z/spacing+z_fix]))

        for obstacle_name, obstacle_id in self.obstacles.items():
            self.collision_objects.append(NamedCollisionObject(obstacle_name))
            self.bodies[obstacle_name] = obstacle_id


    def create_obstacle(self, position=np.array([0.1, 0, 0.1])):
        obstacle_name = "obstacle"

        obstacle_id = self.sim.create_sphere(
            body_name=f"{obstacle_name}_{len(self.obstacles)}",
            radius=0.02,
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
        q = self.robot.get_joint_angles(self.robot.joint_indices[:7])
        obs_per_link = self.collision_detector.compute_distances_per_link(q, self.robot.joint_indices[:7], max_distance=10.0)
        self.obs_d = np.array([min(i) for i in obs_per_link.values()])

        self.is_collided = min(self.obs_d) <= 0

        return self.obs_d

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

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
            return -np.array((d > self.distance_threshold) + (min(obs_d) <= 0)*100, dtype=np.float32)
        else:
            return -(d+exp(-min(d))).astype(np.float32)

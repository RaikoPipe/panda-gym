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
        # todo: put labels into physics client
        self.sim_id = self.sim.physics_client._client

        self.robot: Panda = robot
        # self.robot_params = self.create_robot_debug_params()

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.obstacle_count = 0

        self.bodies = {"robot": self.robot.id}
        self.collision_links = self.robot.link_names[:7]
        self.collision_objects = []
        self.named_collision_pairs = []

        self.debug_manip_label_name = "manip"
        self.debug_manip_label_base_text = "Manipulability Score:"
        self.debug_dist_label_name = "dist"
        self.debug_dist_label_base_text = "Distance:"

        self.sim.create_debug_text(self.debug_manip_label_name, f"{self.debug_manip_label_base_text} 0")
        self.sim.create_debug_text(self.debug_dist_label_name, f"{self.debug_dist_label_base_text} 0")

        with self.sim.no_rendering():
            collision_objects = self._create_scene()

            for link_name in self.collision_links:
                link = NamedCollisionObject("robot", link_name=link_name)
                for obstacle in collision_objects:
                    self.named_collision_pairs.append((link, obstacle))

            self.create_obstacle(np.array([0.1,0.1,0.1]))

            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
            self.collision_detector = CollisionDetector(col_id=self.sim_id, bodies=self.bodies,
                                                        named_collision_pairs=self.named_collision_pairs)

    def _create_scene(self) -> list:
        # todo: create obstacles, return named collision object
        collision_objects = []

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

        return collision_objects

    def create_obstacle(self, position=np.array([0.1, 0, 0.1])):
        obstacle_name = "obstacle"

        obstacle_id = self.sim.create_sphere(
            body_name=f"{obstacle_name}_{self.obstacle_count}",
            radius=0.02,
            mass=0.0,
            position=position,
            rgba_color=np.array([0.5, 0.5, 0.5, 1]),
        )

        self.collision_objects.append(NamedCollisionObject(obstacle_name))
        self.bodies[obstacle_name] = obstacle_id

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
        q = [self.robot.get_joint_angle(i) for i in self.robot.joint_indices[:7]]
        d = self.collision_detector.compute_distances(q, self.robot.joint_indices[:7], max_distance=999.0)

        return d  # no task-specific observation

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

    def update_labels(self, manipulability, distance):
        with self.sim.no_rendering():
            self.sim.remove_all_debug_text()
            #self.sim.remove_debug_text(self.debug_dist_label_name)
            #self.sim.remove_debug_text(self.debug_manip_label_name)

            self.sim.create_debug_text(self.debug_dist_label_name,
                                       f"{self.debug_dist_label_base_text} {round(distance, 3)}")
            self.sim.create_debug_text(self.debug_manip_label_name,
                                       f"{self.debug_manip_label_base_text} {round(manipulability, 5)}")

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        manip = self.robot.get_manipulability()
        self.update_labels(manip, d)

        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)

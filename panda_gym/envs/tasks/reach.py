from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Reach(Task):
    def __init__(
            self,
            sim,
            get_ee_position,
            reward_type="sparse",
            distance_threshold=0.05,
            goal_range=0.3,
            show_goal_space=False
    ) -> None:
        super().__init__(sim)
        self.fixed_target = None
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.collision_detector = None
        with self.sim.no_rendering():
            self._create_scene(show_goal_space)
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self, show_goal_space) -> None:
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
        if show_goal_space:
            self.sim.create_box(
                body_name="goal_space",
                ghost=True,
                half_extents=self.goal_range_high,
                mass=0.0,
                position=np.array([0.0, 0.0, 0.0]),
                rgba_color=np.array([0.0, 0.0, 0.5, 0.2]),
            )
        return

    def is_truncated(self) -> np.ndarray:
        return np.array(False)

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        if self.fixed_target is None:
            self.goal = self._sample_goal()
        else:
            self.fixed_target[0] -= 0.6
            self.goal = self.fixed_target

        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def set_fixed_target(self, target):
        self.fixed_target = target.copy()

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)

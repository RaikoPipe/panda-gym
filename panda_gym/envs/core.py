from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from panda_gym.pybullet import PyBullet
import pybullet


class PyBulletRobot(ABC):
    """Base class for robot env.

    Args:
        sim (PyBullet): Simulation instance.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    def __init__(
            self,
            sim: PyBullet,
            body_name: str,
            file_name: str,
            base_position: np.ndarray,
            action_space: spaces.Space,
            joint_indices: np.ndarray,
            joint_forces: np.ndarray,
    ) -> None:
        self.sim = sim
        self.sim_id = self.sim.physics_client._client
        # if self.sim.dummy_collision_client:
        #     self.sim_dummy_id = self.sim.dummy_collision_client._client

        self.body_name = body_name
        with self.sim.no_rendering():
            self.id = self._load_robot(file_name, base_position)
            self.setup()

        self.link_names = []
        n = pybullet.getNumJoints(self.id, self.sim_id)
        for i in range(n):
            info = pybullet.getJointInfo(self.id, i, self.sim_id)
            link_name = info[12].decode("utf-8")
            self.link_names.append(link_name)

        self.action_space = action_space
        self.joint_indices = joint_indices
        self.joint_forces = joint_forces

    def _load_robot(self, file_name: str, base_position: np.ndarray) -> int:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        idx = self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
        )

        return idx

    def setup(self) -> None:
        """Called after robot loading."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().

        Args:
            action (np.ndarray): The action.
        """

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Velocity as (vx, vy, vz)
        """
        return self.sim.get_link_velocity(self.body_name, link)

    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)

    # def get_joint_angles(self, joints: list) -> np.ndarray:
    #     """Returns the angle of a joint
    #
    #     Args:
    #         joints (list): A list of joint indices
    #
    #     Returns:
    #         List: Joint angles for the given joint indices
    #     """
    #
    #     return np.array([self.get_joint_angle(i) for i in joints])

    def get_joint_angles(self, joints: np.ndarray) -> np.ndarray:
        """Returns the angle of a joint

        Args:
            joints (list): A list of joint indices

        Returns:
            List: Joint angles for the given joint indices
        """

        return self.sim.get_joint_angles(self.body_name, joints)

    def get_joint_velocity(self, joint: int) -> float:
        """Returns the velocity of a joint as (wx, wy, wz)

        Args:
            joint (int): The joint index.

        Returns:
            np.ndarray: Joint velocity as (wx, wy, wz)
        """
        return self.sim.get_joint_velocity(self.body_name, joint)

    def get_joint_velocities(self, joints: np.ndarray) -> np.ndarray:
        """Returns the velocity of a joint as (wx, wy, wz)

        Args:
            joint (int): The joint index.

        Returns:
            np.ndarray: Joint velocity as (wx, wy, wz)
        """
        return self.sim.get_joint_velocities(self.body_name, joints)

    def control_joints(self, action: np.ndarray, control_mode: int) -> None:
        """Control the joints of the robot.

        Args:
            action (np.ndarray): The target angles. The length of the array must equal to the number of joints.
        """
        self.sim.control_joints(
            body=self.body_name,
            joints=self.joint_indices,
            action=action,
            forces=self.joint_forces,
            control_mode=control_mode
        )

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=angles)

    def inverse_kinematics(self, link: int, position: np.ndarray,
                           orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position,
                                                         orientation=orientation)
        return inverse_kinematics


class Task(ABC):
    """Base class for tasks.
    Args:
        sim (PyBullet): Simulation instance.
    """

    def __init__(self, sim: PyBullet) -> None:
        self.sim = sim
        self.goal = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal."""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Returns whether the achieved goal match the desired goal."""

    def is_failure(self) -> bool:
        """Whether the episode ended in a terminal failure state (e.g. a collision)."""
        return False

    def is_truncated(self) -> bool:
        """Whether the episode is cut off without being terminal (e.g. a collision under thesis semantics, which
        lets off-policy learners bootstrap through it). Time limits are handled by gymnasium's TimeLimit."""
        return False

    def step_info(self) -> Dict[str, Any]:
        """Extra per-step info, stored in the replay buffer and passed back to compute_reward (HER)."""
        return {}

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,
                       info: Dict[str, Any] = {}) -> np.ndarray:
        """Compute reward associated to the achieved and the desired goal."""


class RobotTaskEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, robot: PyBulletRobot, task: Task, terminate_on_success=False,
                 render_mode: Optional[str] = None, render_kwargs: Optional[dict] = None) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.render_mode = render_mode
        self.render_kwargs = render_kwargs or {}
        self.sim = robot.sim
        self.robot = robot
        self.task = task
        # required for init; seeded because some scenes sample the start pose by IK from the current joint state,
        # so the first seeded episode of a fresh env must not depend on an unseeded constructor reset
        observation, _ = self.reset(seed=0)
        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["achieved_goal"].shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                achieved_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
            )
        )
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self.terminate_on_success = terminate_on_success
        self._saved_goal = dict()  # For state saving and restoring

    def _get_obs(self) -> Dict[str, np.ndarray]:
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        robot_obs = self.robot.get_obs().astype(np.float32)  # robot state
        task_obs = self.task.get_obs().astype(np.float32)  # object position, velocity, etc...
        observation = np.concatenate([robot_obs, task_obs])

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal().astype(np.float32),
        }

    def reset(
            self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random, seed = seeding.np_random(seed)
        # with self.sim.no_rendering():
        self.robot.reset()
        self.task.reset()
        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def save_state(self) -> int:
        """Save the current state of the envrionment. Restore with `restore_state`.

        Returns:
            int: State unique identifier.
        """
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Resotre the state associated with the unique identifier.

        Args:
            state_id (int): State unique identifier.
        """
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state.

        Args:
            state_id (int): State unique identifier.
        """
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()

        # An episode is terminated if the agent has reached the target
        is_success = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        terminated = False
        if self.terminate_on_success:
            terminated = is_success

        terminated = terminated or self.task.is_failure()
        truncated = self.task.is_truncated()  # time limits are added by gymnasium's TimeLimit wrapper

        info = {"is_success": is_success, **self.task.step_info()}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        """Real-time pacing for "human", an RGB frame for "rgb_array" (camera via render_kwargs)."""
        if self.render_mode is None:
            return None
        kwargs = {"target_position": np.zeros(3), **self.render_kwargs}
        return self.sim.render(self.render_mode, **kwargs)

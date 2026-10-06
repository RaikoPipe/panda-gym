import os
from typing import Optional

import numpy as np

from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet





class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
            self,
            sim: PyBullet,
            block_gripper: bool = False,
            base_position: Optional[np.ndarray] = None,
            control_type: str = "js",
            obs_type: tuple = ("ee",),
            limiter: str = "sim",
            action_limiter: str = "clip",
            n_substeps=20
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        self.obs_type = obs_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)

        path_to_urdf = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "robots", "panda", "panda.urdf")

        super().__init__(
            sim,
            body_name="panda",
            file_name=path_to_urdf,
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])
        self.ee_link = 11 # 17
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

        self.max_change_position = 0.05

        # limits
        self.joint_lim_min = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
        self.joint_lim_max = np.array(
            [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])  # from specifications in radians

        self.joint_velocity_limits = np.array([150, 150, 150, 150, 180, 180, 180])  # degrees per second
        self.joint_acceleration_limits = np.array([150, 150, 150, 150, 180, 180, 180])  # degrees per second
        self.joint_max_jerk = np.array([150, 150, 150, 150, 180, 180, 180])

        self.action_limiter = action_limiter

        # remember actions
        self.previous_action = None
        self.recent_action = None

        self.current_joint_velocity = np.zeros(7)
        self.previous_joint_velocity = np.zeros(7)
        self.current_joint_acceleration = np.zeros(7)  # pybullet doesn't allow for getting acceleration
        self.previous_joint_acceleration = np.zeros(7)
        self.current_joint_jerk = np.zeros(7)


    def set_action(self, action: np.ndarray, action_limiter=None) -> None:

        action = action.copy()  # ensure action don't change

        velocity = self.get_ee_velocity()

        if action_limiter is None:
            action_limiter = self.action_limiter

        if action_limiter == "scale":
            if any([x < -1 or x > 1 for x in action]):
                # if any action is smaller than -1 or bigger than -1, scale down
                max_value = max(abs(action))
                action = np.array([x / max_value for x in action])
        elif action_limiter == "clip":
            action = np.clip(action, self.action_space.low, self.action_space.high)

        # save action
        self.previous_action = self.recent_action
        self.recent_action = action

        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        if self.control_type == "jsd":
            # velocity control
            action = np.concatenate((action, np.array([0, 0])))
            self.control_joints(action=action, control_mode=self.sim.physics_client.VELOCITY_CONTROL)
        elif self.control_type == "pcc":
            # position change control (teleporting)
            target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
            self.set_joint_angles(target_angles)
        else:
            target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
            self.control_joints(action=target_angles, control_mode=self.sim.physics_client.POSITION_CONTROL)

        # save current state
        self.previous_joint_velocity = self.current_joint_velocity
        self.previous_joint_acceleration = self.current_joint_acceleration
        self.current_joint_velocity = np.array([self.get_joint_velocity(joint=i) for i in range(7)])
        self.current_joint_acceleration = self.previous_joint_velocity - self.current_joint_velocity
        self.current_joint_jerk = abs(self.previous_joint_acceleration - self.current_joint_acceleration)


    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * self.max_change_position  # default: * 0.05  # limit maximum change in position

        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        observation = []

        if "ee" in self.obs_type:
            # end-effector position and velocity
            position = np.array(self.get_ee_position())
            velocity = np.array(self.get_ee_velocity())
            observation.extend([position, velocity])

        if "js" in self.obs_type:
            # joint angles and joint velocities
            # position = np.array([self.get_joint_angle(joint=i) for i in range(7)])
            # velocity = np.array([self.get_joint_velocity(joint=i) for i in range(7)])
            position = np.array([self.get_joint_angles(joints=np.array([i for i in range(7)]))]).flatten()
            velocity = np.array([self.get_joint_velocities(joints=np.array([i for i in range(7)]))]).flatten()
            observation.extend([position, velocity])

        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation.append([fingers_width])

        observation = np.concatenate(observation)

        return observation

    def reset(self) -> None:
        self.set_joint_neutral()
        #     self.optimal_pose = None

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def get_manipulability(self) -> float:
        """Yoshikawa manipulability index of the translational Jacobian."""
        q = list(self.get_joint_angles(self.joint_indices))
        jac_t, _ = self.sim.physics_client.calculateJacobian(
            self.id, self.ee_link, [0.0, 0.0, 0.0], q, [0.0] * len(q), [0.0] * len(q)
        )
        jac_t = np.array(jac_t)[:, :7]
        return float(np.sqrt(max(np.linalg.det(jac_t @ jac_t.T), 0.0)))

    def ik(self, target: np.ndarray) -> np.ndarray:
        """Arm joint angles that bring the end-effector to `target` (gripper pointing down)."""
        return self.inverse_kinematics(self.ee_link, target, np.array([1.0, 0.0, 0.0, 0.0]))[:7]

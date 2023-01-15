from typing import Optional

import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet

import roboticstoolbox as rtb

from ruckig import InputParameter, Ruckig, Trajectory, Result
import pathlib
import ruckig


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
            obs_type: str = "ee",
            limiter: str = "sim"
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        self.obs_type = obs_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        # todo: load urdf file from roboticstoolbox panda COPY VISUAL MESHES INTO URDF FOLDER
        # path_to_rtb_urdf = "C:\\Users\\eclip\\Documents\\GitHub\\panda-gym\\panda_gym\\URDF\\robots\\panda.urdf"
        super().__init__(
            sim,
            body_name="panda",
            file_name= "franka_panda_custom/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

        # get panda in roboticstoolbox
        self.rtb_panda = rtb.models.Panda()

        # limits
        self.joint_position_limits_min = np.array([-166, -101, -166, -176, -166, -1, -166])
        self.joint_position_limits_max = np.array([166, 101, 166, -4, 166, 215, 166])

        self.joint_velocity_limits = np.array([150, 150, 150, 150, 180, 180, 180])  # degrees per second
        self.joint_acceleration_limits = np.array([150, 150, 150, 150, 180, 180, 180])  # degrees per second
        self.joint_max_jerk = np.array([150, 150, 150, 150, 180, 180, 180])

        # ruckig
        self.use_ruckig_limiter = True if limiter == "ruckig" else False

        # remember actions
        self.previous_action = None
        self.recent_action = None

        # current state (ruckig)
        self.current_joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.previous_joint_velocities = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.current_joint_acceleration = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def set_action(self, action: np.ndarray) -> None:

        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # save action
        self.previous_action = self.recent_action
        self.recent_action = action

        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            if self.use_ruckig_limiter:
                target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles_ruckig(arm_joint_ctrl)
            else:
                target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))

        self.control_joints(target_angles=target_angles)

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
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position

        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles_ruckig(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position

        inp = InputParameter(7)  # DOFs

        current_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        current_joint_velocities = np.array([self.get_joint_velocity(joint=i) for i in range(7)])
        current_acceleration = current_joint_velocities - self.previous_joint_velocities

        inp.current_position = current_joint_angles
        inp.current_velocity = current_joint_velocities
        inp.current_acceleration = current_acceleration

        inp.target_position = self.current_joint_angles + arm_joint_ctrl

        inp.max_position = self.joint_position_limits_max
        inp.max_velocity = self.joint_velocity_limits
        inp.max_acceleration = self.joint_acceleration_limits
        inp.max_jerk = self.joint_max_jerk

        otg = Ruckig(7)
        trajectory = Trajectory(7)
        result = otg.calculate(inp, trajectory)
        if result == Result.ErrorInvalidInput:
            raise Exception('Invalid input!')

        new_time = self.sim.timestep

        target_joint_angles, target_joint_velocities, target_joint_accelerations = trajectory.at_time(new_time)

        self.previous_joint_velocities = current_joint_velocities

        # target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_joint_angles

    def get_obs(self) -> np.ndarray:
        if self.obs_type == "ee":
            # end-effector position and velocity
            position = np.array(self.get_ee_position())
            velocity = np.array(self.get_ee_velocity())

        else:
            # joint angles and joint velocities
            position = np.array([self.get_joint_angle(joint=i) for i in range(7)])
            velocity = np.array([self.get_joint_velocity(joint=i) for i in range(7)])

        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((position, velocity, [fingers_width]))
        else:
            observation = np.concatenate((position, velocity))

        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

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
        """Returns the manipulability as the yoshikawa index"""
        q = [self.get_joint_angle(i) for i in self.joint_indices[:7]]
        return self.rtb_panda.manipulability(q, axes="trans")

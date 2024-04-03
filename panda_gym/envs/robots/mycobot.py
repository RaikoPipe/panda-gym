from typing import Optional

import numpy as np

from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet
import pybullet as p

import roboticstoolbox as rtb
from spatialmath import SE3

import qpsolvers as qp

class MyCobot(PyBulletRobot):
    """MyCobot robot in PyBullet.

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
            obs_type:tuple=("ee",),
            limiter: str = "sim",
            action_limiter: str = "clip",
            use_robotics_toolbox=True,
            n_substeps = 20
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        self.obs_type = obs_type
        n_action = 3 if self.control_type == "ee" else 6  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)

        # path_to_urdf = "C:\\Users\\eclip\\Documents\\GitHub\\panda-gym\\panda_gym\\URDF\\robots\\franka_panda_custom\\panda.urdf"

        super().__init__(
            sim,
            body_name="mycobot",
            file_name="mycobot/mycobot.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5]),
            joint_forces=np.array([0,0,0,0,0,0]),
        )

        self.neutral_joint_values = np.array([0, 0, 0, 0, 0, 0])
        self.ee_link = 6

        self.action_limiter = action_limiter
        self.max_change_position = 0.05

        # remember actions
        self.previous_action = None
        self.recent_action = None

        self.current_joint_velocity = np.zeros(6)
        self.previous_joint_velocity = np.zeros(6)
        self.current_joint_acceleration = np.zeros(6) # pybullet doesn't allow for getting acceleration
        self.previous_joint_acceleration = np.zeros(6)
        self.current_joint_jerk = np.zeros(6)

        self.rtb = use_robotics_toolbox
        self.panda_rtb = rtb.models.Panda()

        if self.rtb:
            # init roboticstoolbox panda
            # self.swift_env = Swift()
            # self.swift_env.launch()

            self.optimal_pose = None
            # move = spatialmath.SE3(-0.6, 0, 0)
            # self.panda_rtb.base = move
            self.link_collision_location_info = {}
            #
            # self.swift_env.add(self.panda_rtb)
            # # initialise pybullet collision
            # start = self.panda_rtb.link_dict["panda_link1"],
            # end = self.panda_rtb.link_dict["panda_hand"],
            # end, start, _ = self.panda_rtb._get_limit_links(start=start[0], end=end[0])
            # links, n, _ = self.panda_rtb.get_path(start=start, end=end)
            # self.panda_rtb.q = self.neutral_joint_values[:7]
            # self.init_swift_robot()
            # self.update_dummy_robot_link_positions()

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
        elif self.control_type == "jsd":
            # velocity control
            action = np.concatenate((action, np.array([0, 0])))
            self.control_joints(action=action, control_mode=self.sim.physics_client.VELOCITY_CONTROL)
        else:
            arm_joint_ctrl = action[:6]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0

        # save current state
        self.previous_joint_velocity = self.current_joint_velocity
        self.previous_joint_acceleration = self.current_joint_acceleration
        self.current_joint_velocity = np.array([self.get_joint_velocity(joint=i) for i in range(6)])
        self.current_joint_acceleration = self.previous_joint_velocity - self.current_joint_velocity
        self.current_joint_jerk = abs(self.previous_joint_acceleration - self.current_joint_acceleration)



        # if self.rtb:
        #     self.update_dummy_robot_link_positions()

    def update_dummy_robot_link(self, link_id, rtb_link):
        ls = p.getLinkState(bodyUniqueId=self.id, linkIndex=link_id)
        link_world_pos = ls[0]
        link_world_orn = ls[1]

        local_frame_pos = []
        local_frame_orn = []
        csd = p.getCollisionShapeData(objectUniqueId=self.id, linkIndex=link_id)
        for c in csd:
            local_frame_pos.append(c[5])
            local_frame_orn.append(c[6])

        locations = []
        for idx, (pos, orn) in enumerate(zip(local_frame_pos, local_frame_orn)):
            position, orientation = p.multiplyTransforms(positionA=link_world_pos, orientationA=link_world_orn,
                                                         positionB=pos, orientationB=orn)
            col_id = rtb_link.collision.data[idx].co
            # final_pos = np.array(link_world_pos)+np.array(pos)
            # final_orn = p.getQuaternionFromEuler(np.array(link_world_orn) + np.array(orn))
            p.resetBasePositionAndOrientation(bodyUniqueId=col_id, posObj=position, ornObj=orientation,
                                              physicsClientId=1)
            locations.append((position, orientation))

        return locations

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
        target_arm_angles = target_arm_angles[:6]
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * self.max_change_position # default: * 0.05  # limit maximum change in position

        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(6)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        observation = []

        if "ee" in self.obs_type:
            # end-effector position and velocity
            velocity = np.array(self.get_ee_velocity())
            observation.extend([velocity])

        if "js" in self.obs_type:
            # joint angles and joint velocities
            position = np.array([self.get_joint_angles(joints=np.array([i for i in range(6)]))]).flatten()
            velocity = np.array([self.get_joint_velocities(joints=np.array([i for i in range(6)]))]).flatten()
            observation.extend([position, velocity])

        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation.append([fingers_width])

        observation = np.concatenate(observation)

        return observation

    def reset(self) -> None:
        self.set_joint_neutral()
        # if self.rtb:
        #     self.update_dummy_robot_link_positions()
        #     self.optimal_pose = None

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
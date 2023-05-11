from typing import Optional

import numpy as np
import pybullet
import spatialmath
from swift import Swift
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet
import pybullet as p

import roboticstoolbox as rtb
from spatialmath import SE3

from ruckig import InputParameter, Ruckig, Trajectory, Result
import pathlib
import ruckig
import qpsolvers as qp
from pathlib import Path

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
            obs_type:tuple=("ee",),
            limiter: str = "sim",
            action_limiter: str = "clip",
            use_robotics_toolbox=True
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        self.obs_type = obs_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)

        # path_to_urdf = "C:\\Users\\eclip\\Documents\\GitHub\\panda-gym\\panda_gym\\URDF\\robots\\franka_panda_custom\\panda.urdf"

        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda_custom/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

        # limits
        self.joint_lim_min = np.array([-2.7437,-1.7837,-2.9007,-3.0421,-2.8065,0.5445,-3.0159])
        self.joint_lim_max = np.array([2.7437, 1.7837,2.9007,-0.1518,2.8065,4.5169, 3.0159]) # from specifications in radians

        self.joint_velocity_limits = np.array([150, 150, 150, 150, 180, 180, 180])  # degrees per second
        self.joint_acceleration_limits = np.array([150, 150, 150, 150, 180, 180, 180])  # degrees per second
        self.joint_max_jerk = np.array([150, 150, 150, 150, 180, 180, 180])

        self.action_limiter = action_limiter

        # remember actions
        self.previous_action = None
        self.recent_action = None

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
            #self.init_swift_robot()
            #self.update_dummy_robot_link_positions()

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
            action = np.concatenate((action, np.array([0,0])))
            self.control_joints(action=action, control_mode=self.sim.physics_client.VELOCITY_CONTROL)
        elif self.control_type == "pcc":
            # position change control (teleporting)
            target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
            self.set_joint_angles(target_angles)
        else:
            target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
            self.control_joints(action=target_angles, control_mode= self.sim.physics_client.POSITION_CONTROL)

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

    def update_dummy_robot_link_positions(self):
        for idx, link in enumerate(self.get_rtb_links()):
            locations = self.update_dummy_robot_link(idx, link)
            self.link_collision_location_info[link.name] = locations

    def get_rtb_links(self):
        end, start, _ = self.panda_rtb._get_limit_links(start=self.panda_rtb.link_dict["panda_link1"],
                                                        end=self.panda_rtb.link_dict["panda_hand"], )
        links, n, _ = self.panda_rtb.get_path(start=start, end=end)
        return links

    def init_swift_robot(self):
        links = self.get_rtb_links()
        for link in links:
            col = link.collision
            for shape in col.data:
                shape.init_pybullet(self.sim_dummy_id)

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

        inp.max_position = self.joint_lim_max
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
        observation = []

        if "ee" in self.obs_type:
            # end-effector position and velocity
            position = np.array(self.get_ee_position())
            velocity = np.array(self.get_ee_velocity())
            observation.extend([position, velocity])

        if "js" in self.obs_type:
            # joint angles and joint velocities
            position = np.array([self.get_joint_angle(joint=i) for i in range(7)])
            velocity = np.array([self.get_joint_velocity(joint=i) for i in range(7)])
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
        return self.panda_rtb.manipulability(q, axes="trans")

    def compute_action_neo(self, target, collision_objects, collision_detector):
        self.panda_rtb.q = self.get_joint_angles(self.joint_indices[:7])
        self.panda_rtb.qd = [self.get_joint_velocity(i) for i in self.joint_indices[:7]]
        # self.swift_env.step(render=True)
        # self.collision_detector.set_collision_geometries()

        n = self.panda_rtb.n

        # Transform the goal into an SE3 pose
        if self.optimal_pose is not None:
            Tep = self.panda_rtb.fkine(self.optimal_pose)
        else:
            # TODO: Figure out to how to get more natural poses
            if target[1] < 0:
                Tep = SE3().Rx(135, unit="deg")
            else:
                Tep = SE3().Rx(-135, unit="deg")
            # Tep = self.panda_rtb.fkine(self.panda_rtb.q)
        # Tep = SE3.OA([0, 1, 0], [0, 0, -1])
        Tep.A[:3, 3] = target

        # sol = self.panda_rtb.ik_lm_chan(Tep)
        # Tep = self.panda_rtb.fkine(sol.q)

        # The se3 pose of the Panda's end-effector
        Te = self.panda_rtb.fkine(self.panda_rtb.q)

        # Transform from the end-effector to desired pose
        eTep = Te.inv() * Tep

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        v, arrived = rtb.p_servo(Te, Tep, 2.5, 0.05)

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(self.panda_rtb.n + 6)

        # Joint velocity component of Q
        Q[:n, :n] *= Y

        # Slack component of Q
        Q[n:, n:] = (1 / e) * np.eye(6)

        # The equality contraints
        Aeq = np.c_[self.panda_rtb.jacobe(self.panda_rtb.q), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((n + 6, n + 6))
        bin = np.zeros(n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.05

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[:n, :n], bin[:n] = self.panda_rtb.joint_velocity_damper(ps, pi, n)

        # For each collision in the scene
        for collision in collision_objects.keys():
            # Form the velocity damper inequality constraint for each collision
            # object on the robot to the collision in the scene
            c_Ain, c_bin = self.panda_rtb.link_collision_damper_pybullet(
                collision,
                collision_detector,
                self.panda_rtb.q[:n],
                0.4,  # influence distance in which the damper becomes active
                0.1,  # minimum distance in which the link is allowed to approach the object shape
                0.9,
                start=self.panda_rtb.link_dict["panda_link1"],
                end=self.panda_rtb.link_dict["panda_hand"]
            )

            # If there are any parts of the robot within the influence distance
            # to the collision in the scene
            if c_Ain is not None and c_bin is not None:
                c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

                # Stack the inequality constraints
                Ain = np.r_[Ain, c_Ain]
                bin = np.r_[bin, c_bin]

        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-self.panda_rtb.jacobm(self.panda_rtb.q).reshape((n,)), np.zeros(6)]

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[self.panda_rtb.qdlim[:n], 10 * np.ones(6)]
        ub = np.r_[self.panda_rtb.qdlim[:n], 10 * np.ones(6)]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="gurobi")

        # self.panda_rtb.qd[:] = qd[:n]

        # Return the joint velocities
        if qd is None:
            return np.zeros(n)
        return qd[:n]

    def compute_action_neo_pybullet(self, target, collision_objects, collision_detector):
        self.panda_rtb.q = self.get_joint_angles(self.joint_indices[:7])
        # self.swift_env.step(render=True)
        # self.collision_detector.set_collision_geometries()

        n = self.panda_rtb.n

        # Transform the goal into an SE3 pose
        if self.optimal_pose is not None:
            Tep = self.panda_rtb.fkine(self.optimal_pose)
            Tep.A[:3, 3] = target
        else:
            Tep = self.panda_rtb.fkine(self.panda_rtb.q)
            Tep.A[:3, 3] = target

        # sol = self.panda_rtb.ik_lm_chan(Tep)
        # Tep = self.panda_rtb.fkine(sol.q)

        # The se3 pose of the Panda's end-effector
        Te = self.panda_rtb.fkine(self.panda_rtb.q)

        # Transform from the end-effector to desired pose
        eTep = Te.inv() * Tep

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        v, arrived = rtb.p_servo(Te, Tep, 1.0, 0.05)

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(self.panda_rtb.n + 6)

        # Joint velocity component of Q
        Q[:n, :n] *= Y

        # Slack component of Q
        Q[n:, n:] = (1 / e) * np.eye(6)

        # The equality constraints
        Aeq = np.c_[self.panda_rtb.jacobe(self.panda_rtb.q), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((n + 6, n + 6))
        bin = np.zeros(n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.05

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[:n, :n], bin[:n] = self.panda_rtb.joint_velocity_damper(ps, pi, n)

        # For each collision in the scene
        for collision in collision_objects.keys():
            # Form the velocity damper inequality constraint for each collision
            # object on the robot to the collision in the scene
            c_Ain, c_bin = self.panda_rtb.link_collision_damper_pybullet(
                collision,
                collision_detector,
                self.panda_rtb.q[:n],
                0.3,  # influence distance in which the damper becomes active
                0.01,  # minimum distance in which the link is allowed to approach the object shape
                1.0,
                start=self.panda_rtb.link_dict["panda_link1"],
                end=self.panda_rtb.link_dict["panda_hand"],
            )

            # If there are any parts of the robot within the influence distance
            # to the collision in the scene
            if c_Ain is not None and c_bin is not None:
                c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

                # Stack the inequality constraints
                Ain = np.r_[Ain, c_Ain]
                bin = np.r_[bin, c_bin]

        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-self.panda_rtb.jacobm(self.panda_rtb.q).reshape((n,)), np.zeros(6)]

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[self.panda_rtb.qdlim[:n], 10 * np.ones(6)]
        ub = np.r_[self.panda_rtb.qdlim[:n], 10 * np.ones(6)]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="gurobi")

        # self.panda_rtb.qd[:] = qd[:n]

        # Return the joint velocities
        if qd is None:
            return np.zeros(n)
        return qd[:n]

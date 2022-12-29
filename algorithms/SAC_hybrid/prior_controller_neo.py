#!/usr/bin/env python

import spatialmath
from spatialmath import SE3, SO3
from spatialgeometry import Cuboid, Sphere
import pdb
import roboticstoolbox as rtb
import numpy as np
import qpsolvers as qp

from panda_gym.envs.robots.panda import Panda
from swift import Swift


class RRMC:

    def __init__(self, env, collisions):
        self.env = env
        self.panda: Panda = env.robot
        self.collision_detector = env.task.collision_detector
        self.collision_objects = [x for x in env.task.bodies if x not in ["robot", "dummy_target"]]
        # todo: idea: if we can get the robot from the env, why not also the stage and obstacles?
        # todo. compare this to the panda-gym robot
        self.panda_rtb = rtb.models.Panda()
        move = spatialmath.SE3(-0.6,0,0)
        self.panda_rtb.base = move
        self.panda_rtb.q = self.panda.get_joint_angles(self.panda.joint_indices[:7])

        # Tep = panda.fkine(panda.q)
        # Tep.A[:3, 3] = target.T[:3, -1]

        self.n = 7 # number of joints
        self.collisions = collisions

        s0 = Cuboid(np.array([0.02, 0.02, 0.02]), pose=spatialmath.SE3(0, 0.05, 0.15))

        self.target_object = Sphere(0.05, pose=spatialmath.SE3(0.0, 0.0, 0.0), color="green")

        self.collisions = []



    def p_servo(self, wTe, wTep, gain=2):
        '''
        Position-based servoing.

        Returns the end-effector velocity which will cause the robot to approach
        the desired pose.

        :param wTe: The current pose of the end-effecor in the base frame.
        :type wTe: SE3
        :param wTep: The desired pose of the end-effecor in the base frame.
        :type wTep: SE3
        :param gain: The gain for the controller
        :type gain: float
        :param threshold: The threshold or tolerance of the final error between
            the robot's pose and desired pose
        :type threshold: float

        :returns v: The velocity of the end-effecotr which will casue the robot
            to approach wTep
        :rtype v: ndarray(6)
        :returns arrived: True if the robot is within the threshold of the final
            pose
        :rtype arrived: bool

        '''

        if not isinstance(wTe, SE3):
            wTe = SE3(wTe)

        if not isinstance(wTep, SE3):
            wTep = SE3(wTep)

        # Pose difference
        eTep = wTe.inv() * wTep

        # Translational velocity error
        ev = eTep.t

        # Angular velocity error
        ew = eTep.rpy() * np.pi / 180

        # Form error vector
        e = np.r_[ev, ew]

        # Desired end-effector velocity
        v = gain * e

        return v

    def fkine(self):
        # Tip pose in world coordinate frame
        pose = self.panda_rtb.fkine(q=self.panda.get_joint_angles(self.panda.joint_indices[:7]))
        return pose

    def target_pose(self):
        # Target pose in world coordinate frame
        pose = SE3(self.env.task.goal)
        robot_pose = self.panda_rtb.fkine(self.panda_rtb.qr)  # self.fkine
        pose.A[:3, :3] = robot_pose.A[:3, :3]
        return pose

    def compute_action(self, target, gain=1):

        # get goal
        Tep = self.panda_rtb.fkine(self.panda.get_joint_angles(self.panda.joint_indices[:7]))
        Tep.A[:3, 3] = target

        try:
            self.panda_rtb.q = self.panda.get_joint_angles(self.panda.joint_indices[:7])
            ee_pos = self.panda.get_ee_position()
            panda_rtb_pose = self.panda_rtb.fkine(self.panda_rtb.q)
            panda_rtb_pose.A[:3, -1] = ee_pos

            v = self.p_servo(panda_rtb_pose, self.target_pose(), gain=gain)
            v[3:] *= 10
            action = np.linalg.pinv(self.panda_rtb.jacobe(self.panda_rtb.q)) @ v

        except np.linalg.LinAlgError:
            action = np.zeros(self.env.action_space.shape[0])
            print('Fail')

        return action

    def compute_action_neo(self, target):
        self.panda_rtb.q = self.panda.get_joint_angles(self.panda.joint_indices[:7])

        # Transform the goal into an SE3 pose
        Tep = self.fkine()
        Tep.A[:3, 3] = target

        # The se3 pose of the Panda's end-effector
        Te = self.fkine()

        # Transform from the end-effector to desired pose
        eTep = Te.inv() * Tep

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        # todo: nothing, this function is generic
        v, arrived = rtb.p_servo(Te, Tep, 1.0, 0.01)

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(self.n + 6)

        # Joint velocity component of Q
        Q[:self.n, :self.n] *= Y

        # Slack component of Q
        Q[self.n:, self.n:] = (1 / e) * np.eye(6)

        # The equality contraints
        Aeq = np.c_[self.panda_rtb.jacobe(self.panda_rtb.q), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((self.n + 6, self.n + 6))
        bin = np.zeros(self.n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.05

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[:self.n, :self.n], bin[:self.n] = self.panda_rtb.joint_velocity_damper(ps, pi, self.n)

        # For each collision in the scene
        for collision in self.collision_objects:
            # Form the velocity damper inequality constraint for each collision
            # object on the robot to the collision in the scene
            c_Ain, c_bin = self.panda_rtb.link_collision_damper_pybullet(
                collision,
                self.collision_detector,
                self.panda,
                self.panda.get_joint_angles(self.panda.joint_indices[:7]),
                0.3, # influence distance in which the damper becomes active
                0.05, # minimum distance in which the link is allowed to approach the object shape
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
        c = np.r_[-self.panda_rtb.jacobm(self.panda_rtb.q).reshape((self.n,)), np.zeros(6)]

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[self.panda_rtb.qdlim[:self.n], 10 * np.ones(6)]
        ub = np.r_[self.panda_rtb.qdlim[:self.n], 10 * np.ones(6)]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="gurobi")

        # self.panda_rtb.qd[:self.n] = qd[:self.n]

        # Return the joint velocities
        return qd[:self.n]






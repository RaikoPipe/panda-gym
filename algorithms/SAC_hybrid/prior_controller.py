#!/usr/bin/env python

import numpy as np
from spatialmath import SE3, SO3
import pdb
import roboticstoolbox as rb
import numpy as np

from panda_gym.envs.robots.panda import Panda


class RRMC():

    def __init__(self, env):
        self.env = env
        self.panda: Panda = env.robot
        self.panda_rtb = rb.models.Panda()

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

    def compute_action(self, gain=1):

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

"""NEO reactive controller (Haviland & Corke, RA-L 2021) as the thesis reference, ported to pybullet + OSQP.

    neo = NEO()                      # thesis gains
    neo.bind(env.unwrapped)          # after the env is built; the target pose is derived per goal
    qd, _ = neo(obs)                 # joint velocities for control_type="jsd" (velocity control)

Port of 9e12871^:panda_gym/envs/robots/panda.py:compute_action_neo with the collision damper of the thesis'
roboticstoolbox fork (RaikoPipe/robotics-toolbox-python e07d313, Robot.link_collision_damper_pybullet).
Kept from the thesis: the gains below, the QP (joint-velocity and slack weights, slack weight 1/e with the thesis'
error e = sum|t| + sum|rpy| * pi/180), the joint-limit velocity damper, the manipulability term, one closest
distance per link-obstacle pair for panda_link1..panda_hand (plus panda_ee, which carries the finger part of the
thesis hand's collision geometry), the obstacles of the scene without the table, and the target orientation from an
IK solution for the goal. The control rate is the env's (n_substeps=20: 25 Hz).
Replaced: roboticstoolbox kinematics by pybullet Jacobians, Gurobi by OSQP (qpsolvers).
Corrected: the thesis damper multiplied a world-frame unit normal with roboticstoolbox' end-effector-frame Jacobian
of a tool offset equal to the link's world pose. Here the normal multiplies the world-frame translational Jacobian
of the link's closest point, which is NEO's constraint. The rotational error is the rotation vector of the
end-effector-frame pose error (roboticstoolbox' p_servo uses roll-pitch-yaw; equal to first order).
"""
import numpy as np
import pybullet as p
import qpsolvers
from scipy.spatial.transform import Rotation

THESIS_GAINS = dict(
    velocity_gain=0.5,  # gain of the end-effector velocity towards the target pose
    threshold_error=0.05,  # unused by the controller (roboticstoolbox' p_servo "arrived" flag)
    gain_control_minimization=0.01,  # weight of the joint velocities in the QP
    min_angle_joint=0.05,  # rad: closest approach to a joint limit
    min_angle_joint_damp_active=0.9,  # rad: influence angle of the joint-limit damper
    min_dist_obstacle=0.05,  # m: closest approach to an obstacle
    min_dist_obstacle_damp_active=0.3,  # m: influence distance of the collision damper
    damp_gain=1.0,
)
# Franka Panda limits as in roboticstoolbox' Panda model
QLIM = np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                 [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]])
QDLIM = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
DAMPED_LINKS = ("panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6",
                "panda_link7", "panda_link8", "panda_hand", "panda_ee")
N = 7


class NEO:
    def __init__(self, gains=None):
        self.g = {**THESIS_GAINS, **(gains or {})}
        self.env = None

    def bind(self, env):
        """env: the unwrapped PandaReachAO env (control_type "jsd"). Call after construction, before the episode."""
        self.env, self.task, self.robot = env, env.task, env.robot
        self.cid, self.body = self.task.sim_id, self.robot.id
        n_joints = p.getNumJoints(self.body, physicsClientId=self.cid)
        self.movable = [j for j in range(n_joints) if p.getJointInfo(self.body, j, physicsClientId=self.cid)[2] != 4]
        self.links = [self.robot.link_names.index(name) for name in DAMPED_LINKS]
        self._goal = None
        self._ee_local = None  # the end-effector point (the ee link's centre of mass) in that link's frame

    # --- kinematics -----------------------------------------------------------------------------------------------
    def _q(self):
        return np.array([s[0] for s in p.getJointStates(self.body, self.movable, physicsClientId=self.cid)])

    def _local(self, link, point_world):
        """A world point in the link's URDF frame. calculateJacobian's localPosition is relative to that frame, not
        to the centre of mass as its documentation suggests (checked against finite differences, test/neo_test.py)."""
        frame_pos, frame_orn = p.getLinkState(self.body, link, computeForwardKinematics=True,
                                              physicsClientId=self.cid)[4:6]
        return Rotation.from_quat(frame_orn).inv().apply(np.asarray(point_world) - frame_pos)

    def _jacobian(self, link, local, q):
        """World-frame translational and rotational Jacobians (3 x 7) of a point fixed in the link frame, at the
        configuration q (the simulation state is not touched)."""
        zeros = [0.0] * len(self.movable)
        jt, jr = p.calculateJacobian(self.body, link, list(local), list(q), zeros, zeros, physicsClientId=self.cid)
        return np.array(jt)[:, :N], np.array(jr)[:, :N]

    def _ee_pose(self):
        com, orn = p.getLinkState(self.body, self.robot.ee_link, computeForwardKinematics=True,
                                  physicsClientId=self.cid)[:2]
        return np.array(com), Rotation.from_quat(orn)

    def _target_orientation(self, goal):
        """Orientation of the end effector in an IK solution for the goal (thesis section 6.2)."""
        q_ik = self.robot.inverse_kinematics(self.robot.ee_link, goal)[:N]
        states = p.getJointStates(self.body, self.movable, physicsClientId=self.cid)
        for j, qj in zip(self.movable[:N], q_ik):
            p.resetJointState(self.body, j, qj, physicsClientId=self.cid)
        _, orn = self._ee_pose()
        for j, st in zip(self.movable, states):  # restore position and velocity
            p.resetJointState(self.body, j, st[0], st[1], physicsClientId=self.cid)
        return orn

    def _manipulability(self, q):
        jt, jr = self._jacobian(self.robot.ee_link, self._ee_local, q)
        j = np.r_[jt, jr]
        return np.sqrt(max(np.linalg.det(j @ j.T), 0.0))

    def _manipulability_gradient(self, q, eps=1e-3):
        """d(manipulability)/dq by central differences of analytic Jacobians (roboticstoolbox' jacobm)."""
        return np.array([(self._manipulability(q + eps * e) - self._manipulability(q - eps * e)) / (2 * eps)
                         for e in np.eye(len(q))])[:N]

    # --- controller -----------------------------------------------------------------------------------------------
    def __call__(self, obs=None, deterministic=True):
        g = self.g
        goal = np.array(self.task.goal, dtype=float)
        if self._goal is None or not np.array_equal(goal, self._goal):
            self._goal, self._target_orn = goal, self._target_orientation(goal)
        q_all = self._q()
        q = q_all[:N]
        pos, orn = self._ee_pose()
        if self._ee_local is None:  # the end-effector point (the ee link's centre of mass) is fixed in that link
            self._ee_local = self._local(self.robot.ee_link, pos)

        # closest point per (obstacle, link) within the influence distance, attached to the link for its Jacobian
        ds, di, xi = g["min_dist_obstacle"], g["min_dist_obstacle_damp_active"], g["damp_gain"]
        contacts = []
        for obstacle in self.task.obstacles.values():
            for link in self.links:
                pts = p.getClosestPoints(self.body, obstacle, di, linkIndexA=link, physicsClientId=self.cid)
                if pts:
                    c = min(pts, key=lambda pt: pt[8])
                    d, on_link, on_obstacle = c[8], np.array(c[5]), np.array(c[6])
                    normal = (on_obstacle - on_link) / d if d > 1e-9 else -np.array(c[7])  # link -> obstacle
                    contacts.append((link, self._local(link, on_link), normal, d))
        jps = [self._jacobian(link, local, q_all)[0] for link, local, _, _ in contacts]
        jt_ee, jr_ee = self._jacobian(self.robot.ee_link, self._ee_local, q_all)
        jee = np.r_[jt_ee, jr_ee]
        grad_m = self._manipulability_gradient(q_all)

        # end-effector velocity towards the target pose, in the end-effector frame (p_servo, gain 0.5)
        t_err = orn.inv().apply(goal - pos)
        r_err = (orn.inv() * self._target_orn).as_rotvec()
        e = np.sum(np.abs(t_err)) + np.sum(np.abs(r_err)) * np.pi / 180  # thesis error (rpy in rad times pi/180)
        v = g["velocity_gain"] * np.r_[t_err, r_err]
        rot_t = orn.inv().as_matrix()
        je = np.r_[rot_t @ jee[:3], rot_t @ jee[3:]]  # end-effector-frame Jacobian (roboticstoolbox' jacobe)

        Q = np.eye(N + 6)
        Q[:N, :N] *= g["gain_control_minimization"]
        Q[N:, N:] = (1 / max(e, 1e-6)) * np.eye(6)
        Aeq = np.c_[je, np.eye(6)]
        beq = v

        # joint-limit velocity damper (roboticstoolbox' joint_velocity_damper)
        ps, pi = g["min_angle_joint"], g["min_angle_joint_damp_active"]
        Ain, bin_ = np.zeros((N + 6, N + 6)), np.zeros(N + 6)
        for i in range(N):
            if q[i] - QLIM[0, i] <= pi:
                bin_[i] = -((QLIM[0, i] - q[i]) + ps) / (pi - ps)
                Ain[i, i] = -1
            if QLIM[1, i] - q[i] <= pi:
                bin_[i] = ((QLIM[1, i] - q[i]) - ps) / (pi - ps)
                Ain[i, i] = 1

        # collision damper: n^T J_p qd <= xi (d - ds) / (di - ds), one row per (link, obstacle)
        rows = [np.r_[normal @ jp, np.zeros(6)] for (_, _, normal, _), jp in zip(contacts, jps)]
        rhs = [xi * (d - ds) / (di - ds) for _, _, _, d in contacts]
        if rows:
            Ain, bin_ = np.r_[Ain, np.array(rows)], np.r_[bin_, rhs]

        c = np.r_[-grad_m, np.zeros(6)]
        lb, ub = -np.r_[QDLIM, 10 * np.ones(6)], np.r_[QDLIM, 10 * np.ones(6)]
        qd = qpsolvers.solve_qp(Q, c, Ain, bin_, Aeq, beq, lb=lb, ub=ub, solver="osqp", eps_abs=1e-6, eps_rel=1e-6,
                                max_iter=20000, polish=True, verbose=False)
        action = np.zeros(N) if qd is None else qd[:N]
        return action.astype(np.float32), None

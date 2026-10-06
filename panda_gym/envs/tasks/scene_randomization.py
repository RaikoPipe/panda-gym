"""Procedural obstacle scenes for ReachAO ("reachao_dr", "reachao_dr_plr").

Domain randomization follows the neural-motion-planning datasets (MpiNets, Fishman et al. CoRL 2022; Neural MP,
Dalal et al. 2024): besides blobs, scenes contain thin plates, shelf boards and two-wall tunnels, placed so that they
block the straight path to the goal or enclose the goal, and episodes start from varied robot configurations.

Optional automatic curriculum ("_adr"): a per-environment difficulty d in [0, 1] scales the number of objects, the
share of path-blocking/enclosing placements and of varied starts. It grows when >= 70 % of the last 20 episodes
succeed and shrinks when >= 30 % collide (two-sided rule, Kulkarni & Alexis ICRA 2024; boundary expansion as in
OpenAI's ADR); 20 % of episodes stay easy (d = 0) against forgetting.

Optional level replay ("_plr") follows Sampling For Learnability (Rutherford et al., NeurIPS 2024): half of the
episodes replay stored scenes ranked by p(1-p), i.e. scenes the agent solves only sometimes.
"""
from collections import deque

import numpy as np

PARK = np.array([0.0, 0.0, -10.0])  # unused pool bodies wait below the floor
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
OBSTACLE_COLOR = np.array([0.95, 0.55, 0.15, 1.0])


def yaw_quat(yaw, pitch=0.0):
    """Quaternion (x, y, z, w) for a rotation about z by yaw after a rotation about y by pitch."""
    cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([-sy * sp, cy * sp, sy * cp, cy * cp])


class SceneRandomizer:
    def __init__(self, task, level_replay=False, curriculum=False, max_objects=4, regenerate_every=200,
                 buffer_size=200):
        self.task, self.sim, self.rng = task, task.sim, None
        self.level_replay = level_replay
        self.max_objects = max_objects
        self.regenerate_every = regenerate_every
        self.buffer_size = buffer_size
        self.pool = {}  # name -> kind ("sphere" | "box" | "plate")
        self.episodes = 0
        self.levels = []  # [level, attempts, successes]
        self.current = None
        self.last_success = False
        self.curriculum = curriculum
        self.difficulty = 0.0 if curriculum else 1.0
        self.outcomes = deque(maxlen=20)  # (success, collided) of recent episodes

    # --- obstacle pool (bodies are reused; shapes are re-drawn every `regenerate_every` episodes) ---------------
    def build_pool(self):
        rng = self.task.np_random
        for name in self.pool:
            self.sim.remove_body(name)
            self.task.obstacles.pop(name)
            self.task.bodies.pop(name)
        self.pool = {}
        self.levels = []  # stored levels refer to the old shapes
        for i in range(3):
            self._add(f"dr_sphere_{i}", "sphere", self.sim.create_sphere(
                body_name=f"dr_sphere_{i}", radius=rng.uniform(0.03, 0.08), mass=0.0, position=PARK,
                rgba_color=np.array([0.9, 0.15, 0.15, 1.0])))
        for i in range(3):
            self._add(f"dr_box_{i}", "box", self.sim.create_box(
                body_name=f"dr_box_{i}", half_extents=rng.uniform(0.015, 0.15, 3), mass=0.0, position=PARK,
                rgba_color=OBSTACLE_COLOR))
        for i in range(6):  # thin plates: local x is the thin axis
            half = np.array([rng.uniform(0.005, 0.01), rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)])
            self._add(f"dr_plate_{i}", "plate", self.sim.create_box(
                body_name=f"dr_plate_{i}", half_extents=half, mass=0.0, position=PARK, rgba_color=OBSTACLE_COLOR))

    def _add(self, name, kind, uid):
        self.pool[name] = kind
        self.task.obstacles[name] = uid
        self.task.bodies[name] = uid

    # --- episode reset --------------------------------------------------------------------------------------
    def reset(self, collided=False):
        rng = self.task.np_random
        if self.current is not None:
            if self.level_replay:
                self._record(self.current, self.last_success)
            self._update_difficulty(self.last_success, collided)
        if self.episodes % self.regenerate_every == 0:
            self.build_pool()
        self.episodes += 1

        replay = self.level_replay and len(self.levels) >= 20 and rng.random() < 0.5
        d = 0.0 if self.curriculum and rng.random() < 0.2 else self.difficulty
        level = self._pick_level() if replay else self._sample_level(d)
        self._apply(level)
        self.current = level

    def _apply(self, level):
        robot_q, goal, placements = level
        if robot_q is not None:
            self.task.robot.set_joint_angles(robot_q)
        self.task.goal = goal
        self.sim.set_base_pose("target", goal, IDENTITY)
        for name in self.pool:
            self.sim.set_base_pose(name, PARK, IDENTITY)
        for name, pos, quat in placements:
            self.sim.set_base_pose(name, pos, quat)
        self.task.rebuild_collision_detector([name for name, _, _ in placements])

    def _update_difficulty(self, success, collided):
        if not self.curriculum:
            return
        self.outcomes.append((success, collided))
        if len(self.outcomes) == self.outcomes.maxlen:
            succ, coll = np.mean(self.outcomes, axis=0)
            if succ >= 0.7 or coll >= 0.3:
                self.difficulty = float(np.clip(self.difficulty + (0.05 if succ >= 0.7 else -0.05), 0.0, 1.0))
                self.outcomes.clear()

    def _sample_level(self, d=1.0):
        task, rng = self.task, self.task.np_random
        robot_q = None
        if rng.random() < 0.5 * d:  # varied start configuration (collision-free with the table)
            for _ in range(10):
                target = task.sample_within_hollow_sphere(0.35, 0.7, upper_half_only=True, three_quarter_front_half_only=True)
                robot_q = task.robot.ik(target)
                task.robot.set_joint_angles(robot_q)
                clearance = task.collision_detector.get_link_object_distance(
                    "table", max_distance=0.1, ignore_link=["panda_link0", "panda_link1"])
                if min(clearance) > 0.03:  # the base links always touch the table
                    break
                task.robot.set_joint_neutral()
                robot_q = None
        task.set_coll_free_goal(["table", "robot"])
        goal, ee = task.goal.copy(), task.robot.get_ee_position()

        placements = []
        free = list(self.pool)
        rng.shuffle(free)
        for _ in range(rng.integers(1, 1 + round(1 + (self.max_objects - 1) * d))):
            mode = rng.choice(["block", "enclose", "wang"], p=[0.4 * d, 0.3 * d, 1 - 0.7 * d])
            group = self._propose(mode, free, ee, goal)
            if group and self._place_ok(group):
                placements += group
                for name, _, _ in group:
                    free.remove(name)
            else:
                for name, _, _ in group:
                    self.sim.set_base_pose(name, PARK, IDENTITY)
        if not placements:  # observations need at least one obstacle
            name = next(n for n in free if self.pool[n] == "sphere")
            placements = [(name, task.sample_obstacle_wang(), IDENTITY)]
            self.sim.set_base_pose(name, placements[0][1], IDENTITY)
        self.sim.set_base_pose("dummy_sphere", PARK, IDENTITY)
        return robot_q, goal, placements

    def _propose(self, mode, free, ee, goal):
        """Positions/orientations for one object (or a two-plate tunnel)."""
        rng = self.task.np_random
        take = lambda kinds: next((n for n in free if self.pool[n] in kinds), None)  # noqa: E731
        path = goal - ee
        heading = np.arctan2(path[1], path[0])
        if mode == "block":  # wall, box or ball across the straight path to the goal
            name = take(["plate", "box", "sphere"] if rng.random() < 0.6 else ["box", "sphere"])
            if name is None:
                return []
            pos = ee + rng.uniform(0.3, 0.8) * path + rng.normal(0, 0.05, 3)
            return [(name, pos, yaw_quat(heading + rng.normal(0, 0.3)))]
        if mode == "enclose":
            plates = [n for n in free if self.pool[n] == "plate"]
            if rng.random() < 0.5 and len(plates) >= 2:  # tunnel: two walls parallel to the approach direction
                approach = np.arctan2(goal[1], goal[0])
                side = np.array([-np.sin(approach), np.cos(approach), 0.0])
                gap = rng.uniform(0.08, 0.15)
                quat = yaw_quat(approach + np.pi / 2)
                return [(plates[0], goal + gap * side, quat), (plates[1], goal - gap * side, quat)]
            if plates:  # shelf board above or below the goal
                dz = rng.uniform(0.07, 0.15) * rng.choice([-1, 1])
                return [(plates[0], goal + np.array([0, 0, dz]), yaw_quat(rng.uniform(0, np.pi), np.pi / 2))]
            return []
        name = take(["sphere", "box"])
        return [] if name is None else [(name, self.task.sample_obstacle_wang(), yaw_quat(rng.uniform(0, np.pi)))]

    def _place_ok(self, group):
        """Objects must leave the start configuration and the goal free (dummy sphere marks the goal)."""
        task = self.task
        for name, pos, quat in group:
            self.sim.set_base_pose(name, pos, quat)
        task.sim.set_base_pose("dummy_sphere", task.goal, IDENTITY)
        return not any(task.check_collision("robot", n, margin=0.05) or task.check_collision("dummy_sphere", n, margin=0.02)
                       for n, _, _ in group)

    # --- level replay (Sampling For Learnability) -----------------------------------------------------------
    def _record(self, level, success):
        for entry in self.levels:
            if entry[0] is level:
                entry[1] += 1
                entry[2] += success
                return
        self.levels.append([level, 1, int(success)])
        if len(self.levels) > self.buffer_size:
            self.levels.remove(min(self.levels, key=self._learnability))

    @staticmethod
    def _learnability(entry):
        p = (entry[2] + 1) / (entry[1] + 2)  # Beta(1, 1) prior: rarely tried levels stay interesting
        return p * (1 - p)

    def _pick_level(self):
        scores = np.array([self._learnability(e) for e in self.levels])
        return self.levels[self.task.np_random.choice(len(self.levels), p=scores / scores.sum())][0]

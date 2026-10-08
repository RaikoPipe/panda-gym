"""Scene integrity check (T6): start collisions, goal reachability and clearances for every ReachAO scenario.

    python scripts/check_scenes.py [--scenes library2 wall] [--resets 20] [--workers 16]
    python scripts/check_scenes.py --search-wall      # smallest +x shift of the wall cuboid that passes

Per reset (fresh env, seed 50_000 + i, HEAD collision definition):
  start collision      the robot touches an obstacle or the table in its start configuration (link and body named)
  first-step collision a zero action collides
  reachable            position-only IK to the goal has a residual below 1 cm (not collision-aware)
  goal clearance       distance from the goal point to the nearest obstacle (table separately)
  link1-3 clearance    distance of panda_link1..3 to the nearest obstacle at the start
A scene passes when no reset collides at the start or first step and every goal is reachable.
Writes runs/scene_check.json.
"""
import os

# one BLAS/OpenMP thread per worker process (set before numpy is imported; spawned workers inherit it):
# with the default (one thread per core) a pool of workers exhausts the per-user thread limit
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import inspect
import json
import multiprocessing as mp
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as p

import panda_gym  # noqa: F401
from panda_gym.envs.tasks.reach_ao import ReachAO

SEED_BASE = 50_000
DUMMY_RADIUS = 0.05  # ReachAO's dummy_sphere


def all_scenarios():
    names = re.findall(r'"(\w+)": self\.create_', inspect.getsource(ReachAO._create_scenario))
    return [f"{n}-3" if n in ("wang", "wangexp") else n for n in dict.fromkeys(names)]


def closest(cid, a, b, link_a=None, max_distance=5.0):
    kw = {} if link_a is None else {"linkIndexA": link_a}
    pts = p.getClosestPoints(a, b, max_distance, physicsClientId=cid, **kw)
    return min((c[8] for c in pts), default=max_distance)


def check_reset(scene, seed, wall_offset=None):
    env = gym.make("PandaReachAO-v3", scenario=scene, max_episode_steps=10, render_mode=None)
    task = env.unwrapped.task
    cid, robot = task.sim_id, task.robot
    if wall_offset is not None:
        p.resetBasePositionAndOrientation(task.obstacles["cuboid_0"], np.array([0.0, 0.0, 0.1]) + wall_offset,
                                          [0, 0, 0, 1], physicsClientId=cid)
    env.reset(seed=seed)
    goal, names = np.array(task.goal, dtype=float), {u: n for n, u in task.bodies.items()}
    obstacles = list(task.obstacles.values())
    rec = {"seed": seed, "goal": goal.round(4).tolist()}

    p.performCollisionDetection(physicsClientId=cid)
    task.collision_link = None
    rec["start_collision"] = task.check_collided()
    rec["start_collision_with"] = task.collision_link
    rec["link123_clearance"] = min((closest(cid, robot.id, o, link) for o in obstacles for link in (0, 1, 2)),
                                   default=None)
    rec["ee_start"] = robot.get_ee_position().round(4).tolist()

    # goal clearance: move the dummy sphere (r = 5 cm) onto the goal
    task.sim.set_base_pose("dummy_sphere", goal, np.array([0.0, 0.0, 0.0, 1.0]))
    dummy = task.bodies["dummy_sphere"]
    rec["goal_obstacle_clearance"] = min((closest(cid, dummy, o) + DUMMY_RADIUS for o in obstacles), default=None)
    rec["goal_table_clearance"] = closest(cid, dummy, task.bodies["table"]) + DUMMY_RADIUS
    task.sim.set_base_pose("dummy_sphere", np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    if wall_offset is not None:  # does the wall still block the straight path from the start to the goal?
        hit = p.rayTest(rec["ee_start"], goal.tolist(), physicsClientId=cid)[0]
        rec["wall_blocks_straight_path"] = hit[0] == task.obstacles["cuboid_0"]

    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    rec["first_step_collision"] = bool(info["is_collided"])
    rec["first_step_collision_with"] = task.collision_link if info["is_collided"] else None

    # get_ee_position is the ee link's centre of mass (getLinkState[0]); IK targets the link frame, so shift the
    # IK target until the centre of mass lands on the goal
    target = goal.copy()
    for _ in range(5):
        q = p.calculateInverseKinematics(robot.id, robot.ee_link, target.tolist(), maxNumIterations=500,
                                         residualThreshold=1e-6, physicsClientId=cid)[:7]
        robot.set_joint_angles(np.concatenate([q, [0.0, 0.0]]))
        target += goal - robot.get_ee_position()
    rec["ik_residual"] = float(np.linalg.norm(robot.get_ee_position() - goal))
    rec["reachable"] = rec["ik_residual"] < 0.01
    p.performCollisionDetection(physicsClientId=cid)
    rec["ik_pose_collides"] = task.check_collided()
    env.close()
    return scene, rec


def _work(job):
    try:
        return check_reset(*job)
    except Exception as e:  # a broken scene is a finding, not a crash
        return job[0], {"seed": job[1], "error": repr(e)}


def summarize(recs):
    ok = [r for r in recs if "error" not in r]
    vals = lambda k: [r[k] for r in ok if r.get(k) is not None]
    s = {
        "resets": len(recs), "errors": len(recs) - len(ok),
        "start_collisions": sum(r["start_collision"] for r in ok),
        "first_step_collisions": sum(r["first_step_collision"] for r in ok),
        "unreachable_goals": sum(not r["reachable"] for r in ok),
        "ik_pose_collisions": sum(r["ik_pose_collides"] for r in ok),
        "min_goal_obstacle_clearance": min(vals("goal_obstacle_clearance"), default=None),
        "min_goal_table_clearance": min(vals("goal_table_clearance"), default=None),
        "min_link123_clearance": min(vals("link123_clearance"), default=None),
        "collision_partners": sorted({tuple(r[k]) for r in ok for k in ("start_collision_with",
                                     "first_step_collision_with") if r.get(k)}),
    }
    if any("wall_blocks_straight_path" in r for r in ok):
        s["wall_blocks_straight_path"] = sum(r["wall_blocks_straight_path"] for r in ok)
    s["pass"] = (s["errors"] == 0 and s["start_collisions"] == 0 and s["first_step_collisions"] == 0
                 and s["unreachable_goals"] == 0)
    return s


def run(scenes, resets, workers, wall_offset=None):
    jobs = [(scene, SEED_BASE + i, wall_offset) for scene in scenes for i in range(resets)]
    by_scene = {s: [] for s in scenes}
    with mp.get_context("spawn").Pool(workers) as pool:
        for scene, rec in pool.imap(_work, jobs):
            by_scene[scene].append(rec)
    return {s: {"summary": summarize(r), "resets": r} for s, r in by_scene.items()}


def fmt(x):
    return "  -  " if x is None else f"{x:+.3f}"


def main(args):
    if args.search_wall:
        found = None
        for dx in np.arange(0.0, 0.61, 0.02):
            res = run(["wall"], args.resets, args.workers, np.array([dx, 0.0, 0.0]))["wall"]["summary"]
            print(f"dx {dx:.2f}: start coll {res['start_collisions']:2d}  first-step {res['first_step_collisions']:2d}  "
                  f"link1-3 clearance {fmt(res['min_link123_clearance'])}  blocks straight path "
                  f"{res['wall_blocks_straight_path']}/{args.resets}", flush=True)
            if res["pass"] and res["min_link123_clearance"] >= 0.05:
                found = (dx, res)
                break
        print("smallest passing +x offset:", None if found is None else round(float(found[0]), 2))
        return
    results = run(args.scenes or all_scenarios(), args.resets, args.workers)
    print(f"{'scene':20s} pass start 1st-step unreach goal-obst goal-table link1-3  partners")
    for scene, r in results.items():
        s = r["summary"]
        print(f"{scene:20s} {'yes' if s['pass'] else 'NO ':4s} {s['start_collisions']:5d} {s['first_step_collisions']:8d} "
              f"{s['unreachable_goals']:7d} {fmt(s['min_goal_obstacle_clearance']):>9s} "
              f"{fmt(s['min_goal_table_clearance']):>10s} {fmt(s['min_link123_clearance']):>7s}  "
              f"{s['collision_partners'][:3]}{' errors=' + str(s['errors']) if s['errors'] else ''}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--resets", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--search-wall", action="store_true")
    ap.add_argument("--out", default="runs/scene_check.json")
    main(ap.parse_args())

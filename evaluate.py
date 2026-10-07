"""Zero-shot evaluation of ReachAO agents on shared episodes.

    python evaluate.py runs/ReachAO/final_reachao3.zip [--episodes 200] [--scenes library2 workshop]
    python evaluate.py --models 'runs/cmp/*/final*.zip' --collision-def thesis --horizon 100 --out runs/eval/x.jsonl

Episode i of scene s is reset with seed = base + 1000 * scene_index(s) + i (panda_gym/scenes.py), so every model,
collision definition and horizon sees the same goals, obstacles and start configurations; each record carries a
hash of them to check that. Writes one JSON line per episode to --out, prints success / collision / timeout rates
per scene and appends the summary line to runs/eval.jsonl.
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as p
import torch

import panda_gym  # noqa: F401
from panda_gym import scenes as S
from panda_gym.envs.tasks.reach_ao import COLLISION_DEFS, ReachAOConfig
from panda_gym.run_meta import RunMeta

EPISODES_PER_TASK = 25  # episodes per worker task


def env_config(env_id, collision_def):
    """The env's registered config (e.g. v4 observations) with the collision definition swapped in."""
    base = gym.spec(env_id).kwargs.get("config") or ReachAOConfig()
    return replace(base, **COLLISION_DEFS[collision_def])


def episode_hash(env):
    """Fingerprint of what an episode starts from: goal, obstacle poses, robot configuration."""
    task = env.unwrapped.task
    parts = [np.asarray(task.goal, dtype=float)]
    for name in sorted(task.obstacles):
        pos, orn = p.getBasePositionAndOrientation(task.obstacles[name], physicsClientId=task.sim_id)
        parts += [np.asarray(pos), np.asarray(orn)]
    parts.append(task.robot.get_joint_angles(np.arange(7)))
    return hashlib.sha1(np.round(np.concatenate(parts), 6).tobytes()).hexdigest()[:16]


def run_length(seq):
    out = []
    for x in seq:
        if out and out[-1][0] == x:
            out[-1][1] += 1
        else:
            out.append([x, 1])
    return out


def run_episode(policy, env, seed, deterministic):
    obs, _ = env.reset(seed=seed)
    task, robot = env.unwrapped.task, env.unwrapped.robot
    dt = env.unwrapped.sim.dt
    record = {"episode_seed": seed, "hash": episode_hash(env)}
    ee, q, latencies, members = [robot.get_ee_position()], [robot.get_joint_angles(np.arange(7))], [], []
    done, info = False, {}
    while not done:
        t0 = time.perf_counter()
        action, member = policy(obs, deterministic)
        latencies.append(time.perf_counter() - t0)
        members.append(member)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ee.append(robot.get_ee_position())
        q.append(robot.get_joint_angles(np.arange(7)))
    ee, q = np.array(ee), np.array(q)
    path = float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum())
    jerk = np.linalg.norm(np.diff(q, n=3, axis=0), axis=1) / dt ** 3 if len(q) >= 4 else np.zeros(1)
    collided = bool(info.get("is_collided"))
    record.update(
        outcome="collision" if collided else "success" if info.get("is_success") else "timeout",
        steps=len(latencies), path_length=path, mean_ee_speed=path / (len(latencies) * dt),
        mean_jerk=float(jerk.mean()), max_jerk=float(jerk.max()),
        latency_ms_mean=1e3 * float(np.mean(latencies)), latency_ms_p99=1e3 * float(np.percentile(latencies, 99)),
        collision_link=list(task.collision_link) if collided and task.collision_link else None,
    )
    if any(m is not None for m in members):
        record.update(members=run_length(members), switches=sum(a != b for a, b in zip(members, members[1:])))
    return record


# --- policies (loaded once per worker process) -----------------------------------------------------------------
_POLICIES = {}


class _Single:
    def __init__(self, model):
        self.model = model

    def __call__(self, obs, deterministic):
        return self.model.predict(obs, deterministic=deterministic)[0], None

    def reseed(self, seed):
        self.model.set_random_seed(seed)  # stochastic evaluation replays per episode seed


class _Ensemble:
    def __init__(self, models, strategy, space, reduce):
        from panda_gym.ensemble import PolicyEnsemble
        self.ens = PolicyEnsemble(models, strategy=strategy, space=space, reduce=reduce)

    def __call__(self, obs, deterministic):
        return self.ens.predict(obs, deterministic=deterministic)

    def reseed(self, seed):
        self.ens.rng = np.random.default_rng(seed)  # random / bayes_sample replay per episode seed


def get_policy(spec):
    """spec = (strategy or "single", model paths, (space, reduce)) -> callable(obs, deterministic) ->
    (action, acting member or None)."""
    if spec not in _POLICIES:
        from record import load
        strategy, paths, options = spec
        models = [load(path) for path in paths]
        _POLICIES[spec] = _Single(models[0]) if strategy == "single" else _Ensemble(models, strategy, *options)
    return _POLICIES[spec]


def _work(task):
    """A fresh env per episode: PyBullet's contact solver order depends on the env's history (the bodies
    teleported in earlier episodes), and panda_link1 always rubs on the table, so a reused env does not replay an
    episode seed exactly. Costs ~0.25 s per episode."""
    spec, env_id, scene, horizon, collision_def, seeds, deterministic = task
    torch.set_num_threads(1)
    policy = get_policy(spec)
    out = []
    for s in seeds:
        policy.reseed(s)
        env = gym.make(env_id, scenario=scene, max_episode_steps=horizon, render_mode=None,
                       config=env_config(env_id, collision_def))
        out.append(run_episode(policy, env, s, deterministic))
        env.close()
    return out


def evaluate(spec, scenes, args, pool):
    """Per-episode records for one policy on all scenes (shared episode seeds)."""
    tasks = []
    for scene in scenes:
        horizon = args.horizon or S.horizon(scene)
        seeds = [args.episode_seed_base + 1000 * S.scene_index(scene) + i for i in range(args.episodes)]
        for k in range(0, len(seeds), EPISODES_PER_TASK):
            tasks.append((spec, args.env, scene, horizon, args.collision_def, seeds[k:k + EPISODES_PER_TASK],
                          args.deterministic))
    records = []
    for task, recs in zip(tasks, pool.imap(_work, tasks)):
        for r in recs:
            records.append({"model": spec[1][0] if len(spec[1]) == 1 else list(spec[1]), "strategy": spec[0],
                            "scene": task[2], "horizon": task[3], "collision_def": task[4], "ensemble_options": spec[2],
                            "deterministic": task[6], **r})
    return records


def summarize(records, scenes):
    results = {}
    for scene in scenes:
        outcomes = [r["outcome"] for r in records if r["scene"] == scene]
        results[scene] = {k: outcomes.count(k) / len(outcomes) for k in ("success", "collision", "timeout")}
    bench = [s for s in results if s in S.BENCHMARK]
    mean = float(np.mean([results[s]["success"] for s in bench])) if bench else float("nan")
    return results, mean


def expand_models(patterns):
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise FileNotFoundError(pat)
        paths += hits
    return paths


def main(args):
    models = expand_models(args.model + (args.models or []))
    if not models:
        raise SystemExit("no model given")
    if args.select_best:
        S.check_selectable(args.scenes)
    if args.ensemble_strategy:
        specs = [(args.ensemble_strategy, tuple(models), (args.space, args.reduce))]
    else:
        specs = [("single", (m,), None) for m in models]
    out = Path(args.out) if args.out else None
    meta = RunMeta(out.with_suffix(".run_meta.json"), args, config=env_config(args.env, args.collision_def),
                   models=models) if out else None
    all_records, summaries = [], []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for spec in specs:
            t0 = time.time()
            records = evaluate(spec, args.scenes, args, pool)
            results, mean = summarize(records, args.scenes)
            name = spec[1][0] if spec[0] == "single" else f"ensemble[{spec[0]}, K={len(spec[1])}]"
            print(f"{name}  [{args.collision_def}, horizon {args.horizon or 'scene default'}, "
                  f"{args.episodes} episodes, {time.time() - t0:.0f} s]")
            for scene, r in results.items():
                print(f"  {scene:15s} success {r['success']:5.0%}  collision {r['collision']:5.0%}  "
                      f"timeout {r['timeout']:5.0%}", flush=True)
            print(f"  {'benchmark mean':15s} success {mean:5.0%}")
            summaries.append({"model": name, "collision_def": args.collision_def, "horizon": args.horizon,
                              "episodes": args.episodes, "benchmark_mean": mean, "results": results})
            all_records += records
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")
    Path("runs").mkdir(exist_ok=True)
    with open(Path("runs") / "eval.jsonl", "a") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")
    if args.select_best:
        best = max(summaries, key=lambda s: np.mean([r["success"] for r in s["results"].values()]))
        print(f"best on {args.scenes}: {best['model']}")
    if meta:
        meta.finish(benchmark_means={s["model"]: s["benchmark_mean"] for s in summaries})


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="*", default=[], help="model .zip (or glob); same as --models")
    ap.add_argument("--models", nargs="+", help="one or more model .zip files or globs")
    ap.add_argument("--env", default="PandaReachAO-v3")
    ap.add_argument("--scenes", nargs="*", default=list(S.IN_DISTRIBUTION + S.BENCHMARK))
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--horizon", type=int, help="control steps per episode (default: per scene, scenes.py)")
    ap.add_argument("--collision-def", choices=list(COLLISION_DEFS), default="strict")
    ap.add_argument("--episode-seed-base", type=int, default=10_000)
    ap.add_argument("--stochastic", dest="deterministic", action="store_false", help="sample actions")
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    ap.add_argument("--ensemble-strategy", choices=["mean", "precision_weighted", "bayes_sample", "min_variance",
                                                    "critic_argmax", "random"],
                    help="evaluate all --models as one ensemble (panda_gym/ensemble.py)")
    ap.add_argument("--space", choices=["pre_tanh", "action"], default="pre_tanh", help="ensemble variance space")
    ap.add_argument("--reduce", choices=["sum_log_std", "sum_std"], default="sum_log_std",
                    help="min_variance: how per-dimension sigmas are reduced")
    ap.add_argument("--select-best", action="store_true", help="report the best model (validation scenes only)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", help="per-episode JSONL, e.g. runs/eval/<name>.jsonl (+ <name>.run_meta.json)")
    return ap


if __name__ == "__main__":
    main(parser().parse_args())

"""Zero-shot evaluation of a ReachAO agent on the benchmark scenes.

    python evaluate.py runs/ReachAO/final_reachao3.zip [--episodes 50] [--scenes library2 workshop]

Prints success / collision / timeout rates per scene and appends a JSON line to runs/eval.jsonl.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import panda_gym  # noqa: F401
from record import load

SCENES = {"reachao3": 100, "wangexp-3": 100, "library1": 200, "library2": 150, "narrow_tunnel": 150, "tunnel": 150,
          "workshop": 150, "workshop2": 150, "industrial": 150, "bookshelves": 150}


def evaluate(model, scene, max_steps, episodes, n_envs=8, seed=10_000):
    env = make_vec_env("PandaReachAO-v3", n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"scenario": scene, "max_episode_steps": max_steps, "render_mode": None})
    counts = {"success": 0, "collision": 0, "timeout": 0}
    obs = env.reset()
    done_eps = 0
    while done_eps < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        for d, info in zip(dones, infos):
            if d and done_eps < episodes:
                done_eps += 1
                key = "success" if info["is_success"] else "collision" if info["is_collided"] else "timeout"
                counts[key] += 1
    env.close()
    return {k: v / episodes for k, v in counts.items()}


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    args = parser.parse_args()
    model = load(args.model)
    results = {}
    for scene in args.scenes:
        results[scene] = evaluate(model, scene, SCENES.get(scene, 150), args.episodes)
        r = results[scene]
        print(f"{scene:15s} success {r['success']:5.0%}  collision {r['collision']:5.0%}  timeout {r['timeout']:5.0%}", flush=True)
    bench = [s for s in results if s not in ("reachao3", "wangexp-3")]
    mean = float(np.mean([results[s]["success"] for s in bench])) if bench else float("nan")
    print(f"{'benchmark mean':15s} success {mean:5.0%}")
    with open(Path("runs") / "eval.jsonl", "a") as f:
        f.write(json.dumps({"model": args.model, "benchmark_mean": mean, "results": results}) + "\n")

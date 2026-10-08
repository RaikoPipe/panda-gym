"""Train goal-conditioned agents (CrossQ or TQC + HER) on panda-gym tasks.

    python train.py PandaPush-v3 --steps 1_000_000
    python train.py PandaReachAO-v3 --curriculum reachao1:50:0.9 reachao2:75:0.9 reachao3:100 --steps 3_000_000
    python train.py PandaReachAO-v3 --preset thesis_tqc --steps 900000

Runs land in runs/<name>/: TensorBoard logs, checkpoints (used for the training-stage videos), best and final model,
run_meta.json. Explicit flags override --preset, which overrides the defaults.
"""
import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import sb3_contrib.crossq.crossq
from sb3_contrib import TQC, CrossQ
from sb3_contrib.crossq.policies import CrossQPolicy
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import panda_gym  # noqa: F401  (registers the environments)
from panda_gym.envs.tasks.reach_ao import COLLISION_DEFS, ReachAOConfig
from panda_gym.run_meta import RunMeta

class _DictCatTorch:
    """torch, but cat() also joins dicts of tensors key-wise. CrossQ concatenates obs and next_obs for its joint
    BatchRenorm forward pass, which upstream only supports for flat observations."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def cat(tensors, dim=0):
        if isinstance(tensors[0], dict):
            return {k: torch.cat([t[k] for t in tensors], dim=dim) for k in tensors[0]}
        return torch.cat(tensors, dim=dim)


# ponytail: patches sb3-contrib's CrossQ module for dict obs (HER); drop once upstream supports MultiInputPolicy
sb3_contrib.crossq.crossq.th = _DictCatTorch()

HER = dict(replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"))

# Thesis TQC (1624534:run/train.py, hyperparameters_pybullet_defaults_tqc). train_freq counts vec-env steps, so
# 8 gradient steps every 8 steps of 8 envs is an update-to-data ratio of 0.125.
PRESETS = {
    "thesis_tqc": dict(
        algo="tqc", learning_rate=1e-3, lr_schedule="linear", gamma=0.98, tau=0.02, buffer_size=300_000,
        batch_size=256, train_freq=8, gradient_steps=8, use_sde=True, log_std_init=-3.0, net_arch="256,256",
        n_envs=8, eval_freq=5_000, n_eval_episodes=40, n_substeps=5,  # thesis base training ran at 10 ms steps
        curriculum=["reachao1:50:0.9:300000", "reachao2:75:0.9:300000", "reachao3:100::300000"],
    ),
}

# defaults when neither a flag nor the preset sets a value (None: algorithm-specific, see make_model)
DEFAULTS = dict(algo="crossq", learning_rate=1e-3, lr_schedule="const", gamma=0.95, tau=None, buffer_size=1_000_000,
                batch_size=None, train_freq=None, gradient_steps=None, use_sde=False, log_std_init=None,
                net_arch=None, n_envs=8, eval_freq=20_000, n_eval_episodes=40, curriculum=[], utd=None,
                n_substeps=None)


def resolve(args):
    """Explicit flags > preset > DEFAULTS; then train_freq/gradient_steps from --utd (default 1)."""
    preset = PRESETS.get(args.preset, {})
    cli_train_freq, cli_gradient_steps = args.train_freq, args.gradient_steps
    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, preset.get(key, default))
    if args.utd is not None or args.train_freq is None or args.gradient_steps is None:
        train_freq, gradient_steps = utd_schedule(args.utd if args.utd is not None else 1.0, args.n_envs)
        args.train_freq = cli_train_freq if cli_train_freq is not None else train_freq
        args.gradient_steps = cli_gradient_steps if cli_gradient_steps is not None else gradient_steps
    args.effective_utd = args.gradient_steps / (args.train_freq * args.n_envs)
    return args


def utd_schedule(utd, n_envs):
    """(train_freq in vec-env steps, gradient_steps) with gradient_steps / (train_freq * n_envs) == utd."""
    per_step = utd * n_envs
    if per_step >= 1:
        train_freq, gradient_steps = 1, round(per_step)
    else:
        train_freq, gradient_steps = round(1 / per_step), 1
    if not math.isclose(gradient_steps / (train_freq * n_envs), utd):
        raise ValueError(f"UTD {utd} is not reachable with {n_envs} envs (use --train-freq/--gradient-steps)")
    return train_freq, gradient_steps


def reach_ao_config(args):
    """ReachAOConfig from the collision/semantics flags (None for the classic tasks)."""
    base = gym.spec(args.env).kwargs.get("config")
    if "ReachAO" not in args.env:
        if (args.collision_def or args.collision_semantics or args.her_penalty_source or args.n_substeps
                or args.obstacle_obs):
            raise SystemExit("--collision-*, --her-penalty-source and --n-substeps only apply to PandaReachAO")
        return None
    overrides = dict(COLLISION_DEFS[args.collision_def]) if args.collision_def else {}
    for key in ("collision_semantics", "her_penalty_source", "n_substeps", "obstacle_obs"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    return replace(base or ReachAOConfig(), **overrides)


def make_env(env_id, n_envs, seed, scenario=None, max_episode_steps=None, vec_env_cls=SubprocVecEnv, config=None,
             **env_kwargs):
    kwargs = {"render_mode": None, **({"scenario": scenario} if scenario else {}), **env_kwargs}  # SB3 defaults to rgb_array
    if max_episode_steps:
        kwargs["max_episode_steps"] = max_episode_steps
    if config is not None:
        kwargs["config"] = config
    return make_vec_env(env_id, n_envs=n_envs, seed=seed, env_kwargs=kwargs, vec_env_cls=vec_env_cls)


def set_reward_env(model, env_id, scenario, config=None):
    """HER recomputes rewards on every gradient step; doing that in-process avoids a subprocess round trip each time.
    Valid because compute_reward only depends on (achieved_goal, desired_goal, info), not on the live simulation.
    With her_penalty_source="live" it does depend on it (thesis bug), so the training env computes the rewards."""
    if isinstance(model.replay_buffer.env, DummyVecEnv):
        model.replay_buffer.env.close()  # previous stage's reward env (the training env is closed by train())
    model.replay_buffer.env = None
    if config is not None and config.her_penalty_source == "live":
        model.replay_buffer.set_env(model.get_env())
    else:
        model.replay_buffer.set_env(make_env(env_id, 1, 0, scenario, vec_env_cls=DummyVecEnv, config=config))


def linear_schedule(initial):
    return lambda progress_remaining: progress_remaining * initial


def make_model(env, args, log_dir):
    lr = linear_schedule(args.learning_rate) if args.lr_schedule == "linear" else args.learning_rate
    common = dict(env=env, learning_starts=args.learning_starts, buffer_size=args.buffer_size, gamma=args.gamma,
                  train_freq=args.train_freq, gradient_steps=args.gradient_steps, seed=args.seed, learning_rate=lr,
                  tensorboard_log=str(log_dir), verbose=1, device="cuda" if torch.cuda.is_available() else "cpu",
                  **({"use_sde": True} if args.use_sde else {}), **HER)
    arch = [int(x) for x in args.net_arch.split(",")] if args.net_arch else None
    sde_kwargs = {"log_std_init": args.log_std_init} if args.log_std_init is not None else {}
    if args.algo == "crossq":
        # CrossQ (Bhatt et al., ICLR 2024): batch-renormalized critic, no target network -> sample efficient at UTD 1
        return CrossQ(CrossQPolicy, batch_size=args.batch_size or 1024, policy_delay=3,
                      policy_kwargs=dict(features_extractor_class=CombinedExtractor, **sde_kwargs,
                                         net_arch=dict(pi=arch or [256, 256], qf=arch or [1024, 1024])), **common)
    # baseline: TQC with the rl-baselines3-zoo panda-gym hyperparameters
    return TQC("MultiInputPolicy", batch_size=args.batch_size or 2048, tau=args.tau or 0.05,
               policy_kwargs=dict(net_arch=arch or [512, 512, 512], n_critics=2, **sde_kwargs), **common)


class StopOnSuccessRate(BaseCallback):
    """Used as EvalCallback(callback_after_eval=...): stop learning once the eval success rate is reached."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def _on_step(self) -> bool:
        return np.mean(self.parent._is_success_buffer) < self.threshold


class EvalLog(BaseCallback):
    """Used as EvalCallback(callback_after_eval=...): appends each evaluation (with wall-clock time since the start of
    the run) to runs/<name>/evals.jsonl, then runs the stop rule if there is one."""

    def __init__(self, path, stage, t_start, stop=None):
        super().__init__()
        self.path, self.stage, self.t_start, self.stop = path, stage, t_start, stop

    def _on_step(self) -> bool:
        ev = self.parent
        with open(self.path, "a") as f:
            f.write(json.dumps({"stage": self.stage, "timesteps": self.model.num_timesteps,
                                "wall_s": round(time.time() - self.t_start, 1),
                                "success_rate": float(np.mean(ev._is_success_buffer)) if ev._is_success_buffer else None,
                                "mean_reward": float(ev.last_mean_reward)}) + "\n")
        if self.stop is not None:
            self.stop.parent = ev
            return self.stop._on_step()
        return True


class Throughput(BaseCallback):
    """Logs env steps and gradient steps per second (wall clock) every `every` env steps."""

    def __init__(self, every=10_000):
        super().__init__()
        self.every = every

    def _on_training_start(self):
        self.last = (time.perf_counter(), self.model.num_timesteps, self.model._n_updates)

    def _on_step(self) -> bool:
        t0, steps0, updates0 = self.last
        if self.model.num_timesteps - steps0 >= self.every:
            now, dt = time.perf_counter(), time.perf_counter() - t0
            self.logger.record("time/env_steps_per_s", (self.model.num_timesteps - steps0) / dt)
            self.logger.record("time/grad_steps_per_s", (self.model._n_updates - updates0) / dt)
            self.last = (now, self.model.num_timesteps, self.model._n_updates)
        return True


def parse_stage(stage):
    """scenario:max_episode_steps[:success_threshold[:max_stage_steps]] (empty fields allowed, e.g. a::b)."""
    scenario, *rest = stage.split(":")
    rest += [""] * (3 - len(rest))
    as_num = lambda s, cls: cls(s) if s else None
    return scenario or None, as_num(rest[0], int), as_num(rest[1], float), as_num(rest[2], int)


def train(args):
    run_dir = Path("runs") / (args.name or f"{args.env.removesuffix('-v3')}_{args.algo}_s{args.seed}")
    config = reach_ao_config(args)
    meta = RunMeta(run_dir / "run_meta.json", args, config=config)
    # one stage for the classic tasks; scenario:max_episode_steps[:success_threshold[:max_stage_steps]] for ReachAO
    stages = [parse_stage(s) for s in args.curriculum] or [(None, None, None, None)]
    budget = args.steps
    model = None
    t_start = time.time()
    for i, (scenario, max_steps, threshold, stage_cap) in enumerate(stages):
        last = i == len(stages) - 1
        extra = {"grasp_start_prob": args.grasp_start_prob} if args.grasp_start_prob else {}
        env = make_env(args.env, args.n_envs, args.seed + i, scenario, max_steps, config=config, **extra)
        eval_env = make_env(args.env, 4, args.seed + 1000 + i, scenario, max_steps, config=config)  # normal starts
        if model is None and args.init:  # fine-tune a trained agent (fresh replay buffer, short random warm-up)
            model = {"crossq": CrossQ, "tqc": TQC}[args.algo].load(args.init, env=env, tensorboard_log=str(run_dir))
            model.learning_starts = model.num_timesteps + args.learning_starts
            model.save(run_dir / "checkpoints" / f"model_{scenario}_{model.num_timesteps}_steps")  # zero-shot policy
        elif model is None:
            model = make_model(env, args, run_dir)
            model.save(run_dir / "checkpoints" / "model_0_steps")  # untrained policy for the first video
        else:
            model.set_env(env, force_reset=True)
        set_reward_env(model, args.env, scenario, config)

        tag = f"_{scenario}" if scenario else ""
        callbacks = [
            CheckpointCallback(max(args.steps // args.n_checkpoints // args.n_envs, 1), str(run_dir / "checkpoints"),
                               name_prefix=f"model{tag}"),
            EvalCallback(eval_env, n_eval_episodes=args.n_eval_episodes, eval_freq=max(args.eval_freq // args.n_envs, 1),
                         best_model_save_path=str(run_dir / f"best{tag}"), verbose=1,
                         callback_after_eval=EvalLog(run_dir / "evals.jsonl", scenario, t_start,
                                                     StopOnSuccessRate(threshold) if threshold is not None and not last
                                                     else None)),
            Throughput(),
        ]
        steps_before = model.num_timesteps
        model.learn(total_timesteps=min(budget, stage_cap or budget), callback=callbacks, reset_num_timesteps=False,
                    tb_log_name="tb", progress_bar=False)
        budget -= model.num_timesteps - steps_before
        model.save(run_dir / f"final{tag}")
        env.close()
        eval_env.close()
        if budget <= 0:
            break
    meta.finish(num_timesteps=model.num_timesteps)
    return model


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env")
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--algo", choices=["crossq", "tqc"])
    ap.add_argument("--steps", type=int, default=1_000_000, help="total env steps over all curriculum stages")
    ap.add_argument("--curriculum", nargs="*",
                    help="scenario:max_episode_steps[:success_threshold[:max_stage_steps]]")
    ap.add_argument("--n-envs", type=int)
    ap.add_argument("--utd", type=float, help="gradient steps per env step (default 1; sets train freq/grad steps)")
    ap.add_argument("--train-freq", type=int, help="vec-env steps between updates (overrides --utd)")
    ap.add_argument("--gradient-steps", type=int, help="gradient steps per update (overrides --utd)")
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--lr-schedule", choices=["const", "linear"])
    ap.add_argument("--tau", type=float, help="TQC target update rate (default 0.05)")
    ap.add_argument("--buffer-size", type=int)
    ap.add_argument("--batch-size", type=int, help="default: CrossQ 1024, TQC 2048")
    ap.add_argument("--use-sde", action="store_true", default=None, help="generalized state-dependent exploration")
    ap.add_argument("--log-std-init", type=float)
    ap.add_argument("--net-arch", help="hidden layer sizes for actor and critic, e.g. 256,256")
    ap.add_argument("--collision-def", choices=list(COLLISION_DEFS), help="ReachAO collision definition")
    ap.add_argument("--collision-semantics", choices=["terminate", "truncate"])
    ap.add_argument("--her-penalty-source", choices=["info", "live"], help="live: thesis bug, G2 ablation only")
    ap.add_argument("--n-substeps", type=int, help="ReachAO physics substeps per control step (default 20)")
    ap.add_argument("--obstacle-obs", help="ReachAO obstacle observation, e.g. vectors+past (history of closest-point "
                    "vectors, for the dynamic scene); default: the env's")
    ap.add_argument("--learning-starts", type=int, default=10_000)
    ap.add_argument("--eval-freq", type=int)
    ap.add_argument("--n-eval-episodes", type=int)
    ap.add_argument("--n-checkpoints", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--init", help="model .zip to fine-tune instead of starting from scratch")
    ap.add_argument("--grasp-start-prob", type=float, default=0.0,
                    help="PickAndPlace/Stack: share of training episodes that start with the cube in the gripper")
    return ap


if __name__ == "__main__":
    torch.set_num_threads(1)  # the GPU does the math; more CPU threads only contend with the env processes
    train(resolve(parser().parse_args()))

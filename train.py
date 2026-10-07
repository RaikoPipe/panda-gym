"""Train goal-conditioned agents (CrossQ or TQC + HER) on panda-gym tasks.

    python train.py PandaPush-v3 --steps 1_000_000
    python train.py PandaReachAO-v3 --curriculum reachao1:50:0.9 reachao2:75:0.9 reachao3:100 --steps 3_000_000

Runs land in runs/<name>/: TensorBoard logs, checkpoints (used for the training-stage videos), best and final model.
"""
import argparse
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


def make_env(env_id, n_envs, seed, scenario=None, max_episode_steps=None, vec_env_cls=SubprocVecEnv, **env_kwargs):
    kwargs = {"render_mode": None, **({"scenario": scenario} if scenario else {}), **env_kwargs}  # SB3 defaults to rgb_array
    if max_episode_steps:
        kwargs["max_episode_steps"] = max_episode_steps
    return make_vec_env(env_id, n_envs=n_envs, seed=seed, env_kwargs=kwargs, vec_env_cls=vec_env_cls)


def set_reward_env(model, env_id, scenario):
    """HER recomputes rewards on every gradient step; doing that in-process avoids a subprocess round trip each time.
    Valid because compute_reward only depends on (achieved_goal, desired_goal, info), not on the live simulation."""
    if isinstance(model.replay_buffer.env, DummyVecEnv):
        model.replay_buffer.env.close()  # previous stage's reward env (the training env is closed by train())
    model.replay_buffer.env = None
    model.replay_buffer.set_env(make_env(env_id, 1, 0, scenario, vec_env_cls=DummyVecEnv))


def make_model(algo, env, args, log_dir):
    common = dict(env=env, learning_starts=args.learning_starts, buffer_size=1_000_000, gamma=args.gamma,
                  train_freq=1, gradient_steps=max(1, round(args.utd * env.num_envs)), seed=args.seed,
                  tensorboard_log=str(log_dir), verbose=1, device="cuda" if torch.cuda.is_available() else "cpu", **HER)
    if algo == "crossq":
        # CrossQ (Bhatt et al., ICLR 2024): batch-renormalized critic, no target network -> sample efficient at UTD 1
        return CrossQ(CrossQPolicy, learning_rate=1e-3, batch_size=args.batch_size, policy_delay=3,
                      policy_kwargs=dict(features_extractor_class=CombinedExtractor,
                                         net_arch=dict(pi=[256, 256], qf=[1024, 1024])), **common)
    # baseline: TQC with the rl-baselines3-zoo panda-gym hyperparameters
    return TQC("MultiInputPolicy", learning_rate=1e-3, batch_size=2048, tau=0.05,
               policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2), **common)


class StopOnSuccessRate(BaseCallback):
    """Used as EvalCallback(callback_after_eval=...): stop learning once the eval success rate is reached."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def _on_step(self) -> bool:
        return np.mean(self.parent._is_success_buffer) < self.threshold


def train(args):
    run_dir = Path("runs") / (args.name or f"{args.env.removesuffix('-v3')}_{args.algo}_s{args.seed}")
    # one stage for the classic tasks; scenario:max_episode_steps[:success_threshold] stages for ReachAO
    stages = [s.split(":") for s in args.curriculum] or [[None]]
    budget = args.steps
    model = None
    for i, (scenario, *rest) in enumerate(stages):
        last = i == len(stages) - 1
        max_steps = int(rest[0]) if rest else None
        threshold = float(rest[1]) if len(rest) > 1 else None
        extra = {"grasp_start_prob": args.grasp_start_prob} if args.grasp_start_prob else {}
        env = make_env(args.env, args.n_envs, args.seed + i, scenario, max_steps, **extra)  # eval env: normal starts
        eval_env = make_env(args.env, 4, args.seed + 1000 + i, scenario, max_steps)
        if model is None and args.init:  # fine-tune a trained agent (fresh replay buffer, short random warm-up)
            model = {"crossq": CrossQ, "tqc": TQC}[args.algo].load(args.init, env=env, tensorboard_log=str(run_dir))
            model.learning_starts = model.num_timesteps + args.learning_starts
            model.save(run_dir / "checkpoints" / f"model_{scenario}_{model.num_timesteps}_steps")  # zero-shot policy
        elif model is None:
            model = make_model(args.algo, env, args, run_dir)
            model.save(run_dir / "checkpoints" / "model_0_steps")  # untrained policy for the first video
        else:
            model.set_env(env, force_reset=True)
        set_reward_env(model, args.env, scenario)

        tag = f"_{scenario}" if scenario else ""
        callbacks = [
            CheckpointCallback(max(args.steps // args.n_checkpoints // args.n_envs, 1), str(run_dir / "checkpoints"),
                               name_prefix=f"model{tag}"),
            EvalCallback(eval_env, n_eval_episodes=args.n_eval_episodes, eval_freq=max(args.eval_freq // args.n_envs, 1),
                         best_model_save_path=str(run_dir / f"best{tag}"), verbose=1,
                         callback_after_eval=StopOnSuccessRate(threshold) if threshold is not None and not last else None),
        ]
        steps_before = model.num_timesteps
        model.learn(total_timesteps=budget, callback=callbacks, reset_num_timesteps=False,
                    tb_log_name="tb", progress_bar=False)
        budget -= model.num_timesteps - steps_before
        model.save(run_dir / f"final{tag}")
        env.close()
        eval_env.close()
        if budget <= 0:
            break
    return model


if __name__ == "__main__":
    torch.set_num_threads(1)  # the GPU does the math; more CPU threads only contend with the env processes
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env")
    parser.add_argument("--algo", choices=["crossq", "tqc"], default="crossq")
    parser.add_argument("--steps", type=int, default=1_000_000, help="total env steps over all curriculum stages")
    parser.add_argument("--curriculum", nargs="*", default=[], help="scenario:max_episode_steps[:success_threshold]")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--utd", type=float, default=1.0, help="gradient steps per env step")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=1024, help="CrossQ batch size (TQC uses 2048)")
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=40)
    parser.add_argument("--n-checkpoints", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name")
    parser.add_argument("--init", help="model .zip to fine-tune instead of starting from scratch")
    parser.add_argument("--grasp-start-prob", type=float, default=0.0,
                        help="PickAndPlace/Stack: share of training episodes that start with the cube in the gripper")
    train(parser.parse_args())

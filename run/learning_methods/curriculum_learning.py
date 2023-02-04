from typing import Optional, Union

import gymnasium
import gymnasium as gym
import numpy as np

import sys
import gymnasium

sys.modules["gym"] = gymnasium

from stable_baselines3 import SAC, TD3, PPO, DDPG, DQN
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
from run.learning_methods.imitation_learning import fill_replay_buffer

import wandb
from wandb.integration.sb3 import WandbCallback


def get_env(config, stage, deactivate_render=False):
    if config["n_envs"] > 1:

        env = make_vec_env(config["env_name"], n_envs=config["n_envs"],
                           env_kwargs={"render": False,
                                       "control_type": config["control_type"],
                                       "obs_type": config["obs_type"],
                                       "reward_type": config["reward_type"],
                                       "goal_distance_threshold": config["goal_distance_threshold"],
                                       "limiter": config["limiter"],
                                       "show_goal_space": False,
                                       "obstacle_layout": stage,
                                       "show_debug_labels": False
                                       },
                           vec_env_cls=SubprocVecEnv)
    else:
        # todo: check if obsolete
        # env = gym.make(config["env_name"],
        #                render=config["render"] if not deactivate_render else False,
        #                #n_envs=config["n_envs"],
        #                control_type=config["control_type"],
        #                obs_type=config["obs_type"],
        #                goal_distance_threshold=config["goal_distance_threshold"],
        #                reward_type=config["reward_type"],
        #                limiter=config["limiter"],
        #                show_goal_space=False,
        #                obstacle_layout=stage,
        #                show_debug_labels=False,)
        env = make_vec_env(config["env_name"], n_envs=config["n_envs"],
                           env_kwargs={"render": config["render"] if not deactivate_render else False,
                                       "control_type": config["control_type"],
                                       "obs_type": config["obs_type"],
                                       "reward_type": config["reward_type"],
                                       "goal_distance_threshold": config["goal_distance_threshold"],
                                       "limiter": config["limiter"],
                                       "show_goal_space": False,
                                       "obstacle_layout": stage,
                                       "show_debug_labels": False
                                       })

    return env


def get_model(algorithm, config):
    tags = get_tags(config)
    run = init_wandb(config, tags)

    env = get_env(config, config["stages"][0], deactivate_render=True)
    n_actions = env.action_space.shape[0]
    # env.close()

    if config.get("noise_std"):
        normal_action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                                sigma=config["noise_std"] * np.ones(n_actions))
        vectorized_action_noise = VectorizedActionNoise(n_envs=config["n_envs"], base_noise=normal_action_noise)
        action_noise = vectorized_action_noise if config["n_env"] > 1 else normal_action_noise
    else:
        action_noise = None

    if algorithm in ("TD3", "DDPG"):
        model = TD3(config["policy_type"], env=get_env(config, config["stages"][0], deactivate_render=True),
                    verbose=1, seed=config["seed"],
                    tensorboard_log=f"runs/{run.id}", device="cuda",
                    replay_buffer_class=config["replay_buffer"],
                    # hyperparameters
                    train_freq=1 if config["n_envs"] > 1 else (1, "episode"),
                    # config["n_envs"] if config["n_envs"] > 1 else (1, "episode"),
                    gradient_steps=config["gradient_steps"],
                    learning_starts=config["learning_starts"],
                    learning_rate=config["learning_rate"],
                    gamma=config["gamma"],
                    buffer_size=config["buffer_size"],
                    policy_kwargs=config["policy_kwargs"],
                    action_noise=action_noise
                    )
    elif algorithm == "SAC":
        model = SAC(config["policy_type"], env=get_env(config, config["stages"][0], deactivate_render=True),
                    verbose=1, seed=config["seed"],
                    tensorboard_log=f"runs/{run.id}", device="cuda",
                    replay_buffer_class=config["replay_buffer"],

                    # hyperparameters
                    learning_starts=config["learning_starts"],
                    learning_rate=config["learning_rate"],
                    gamma=config["gamma"],
                    tau=config["tau"],
                    buffer_size=config["buffer_size"],
                    gradient_steps=config["gradient_steps"],
                    train_freq=config["train_freq"],
                    use_sde=config["use_sde"],
                    policy_kwargs=config["policy_kwargs"]
                    )
    elif algorithm == "TQC":
        model = TQC(config["policy_type"], env=get_env(config, config["stages"][0], deactivate_render=True),
                    verbose=1, seed=config["seed"],
                    tensorboard_log=f"runs/{run.id}", device="cuda",
                    replay_buffer_class=config["replay_buffer"],

                    # hyperparameters
                    learning_starts=config["learning_starts"],
                    learning_rate=config["learning_rate"],
                    gamma=config["gamma"],
                    tau=config["tau"],
                    buffer_size=config["buffer_size"],
                    gradient_steps=config["gradient_steps"],
                    train_freq=config["train_freq"],
                    use_sde=config["use_sde"],
                    policy_kwargs=config["policy_kwargs"]
                    )

    return model, run


def get_tags(config):
    # set wandb tags
    tags = []
    tags.extend(str(x) for x in config["stages"])

    if len(config["stages"]) > 1:
        tags.append("curriculum_learning")
    if config["n_envs"] > 1:
        tags.append("multi_env")
    tags.append(config["algorithm"])
    tags.append(config["reward_type"])

    if config["prior_steps"] > 0:
        tags.append("prior")

    return tags


def init_wandb(config, tags):
    env_name = config["env_name"]
    project = f"{env_name}"
    run = wandb.init(
        project=f"{project}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        dir="run_data",
        tags=tags,
        group=config["stages"][-1]  # last stage is job goal
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    return run


def learn(config: dict, initial_model: Optional[OffPolicyAlgorithm] = None,
          starting_stage: Optional[str] = None, algorithm: str = "TD3"):
    if not initial_model:

        model, run = get_model(algorithm, config)

    else:
        model = initial_model
        stages: list = config["stages"]
        reward_thresholds = config["reward_thresholds"]
        if starting_stage:
            assert starting_stage in stages

            idx = stages.index(starting_stage)
            config["stages"] = stages[idx:]
            config["reward_thresholds"] = reward_thresholds[idx:]

    # model = TD3.load(r"run_data/wandb/run_panda_reach_evade_obstacle_stage_2_best_run/files/model.zip", env=env,
    #                  device="cuda", train_freq=n_envs, gradient_steps=2, replay_buffer=replay_buffer)

    assert len(config["stages"]) == len(config["reward_thresholds"])

    # model.env.close()
    # learn for each stage until reward threshold is reached
    for stage, reward_threshold in zip(config["stages"], config["reward_thresholds"]):
        model.set_env(get_env(config, stage))

        if config["learning_starts"]:
            model.learn(total_timesteps=config["learning_starts"])
            model.learning_starts = 0

        if config["prior_steps"]:
            model.replay_buffer = fill_replay_buffer(model, config["prior_steps"])

        eval_env = gym.make(config["env_name"], render=False, control_type=config["control_type"],
                            obs_type=config["obs_type"],
                            reward_type=config["reward_type"],
                            show_goal_space=False, obstacle_layout=stage,
                            show_debug_labels=False)

        stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)

        eval_callback = EvalCallback(eval_env, eval_freq=max(config["eval_freq"] // config["n_envs"], 1),
                                     callback_after_eval=stop_train_callback, verbose=1, n_eval_episodes=100,
                                     best_model_save_path=wandb.run.dir)

        model.learn(
            total_timesteps=config["max_timesteps"],
            callback=[WandbCallback(
                model_save_path=wandb.run.dir,
                model_save_freq=20_000
            ), eval_callback],
            progress_bar=True
        )

        eval_env.close()

    run.finish()
    return model

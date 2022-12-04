from typing import Optional, Union

import gymnasium
import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC, TD3, PPO, DDPG, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv

import wandb
from wandb.integration.sb3 import WandbCallback


def get_env(config, stage):
    if config["n_envs"] > 1:
        # rendering is not allowed in multiprocessing
        render = show_goal_space = show_debug_labels = False

        env = make_vec_env(config["env_name"], n_envs=config["n_envs"],
                           env_kwargs={"render": render, "control_type": config["control_type"],
                                       "obs_type": config["obs_type"],
                                       "reward_type": config["reward_type"], "limiter": config["limiter"],
                                       "show_goal_space": show_goal_space, "obstacle_layout": stage,
                                       "show_debug_labels": show_debug_labels}, vec_env_cls=SubprocVecEnv)
    else:
        show_goal_space = show_debug_labels = True if config["render"] else False

        env = gym.make(config["env_name"], render=config["render"], control_type=config["control_type"],
                       obs_type=config["obs_type"],
                       reward_type=config["reward_type"], limiter=config["limiter"],
                       show_goal_space=show_goal_space, obstacle_layout=stage,
                       show_debug_labels=show_debug_labels)

    return env

def get_model(algorithm, config, run):
    env = get_env(config, config["stages"][0])
    n_actions = env.action_space.shape[0]

    if config.get("noise_std"):
        normal_action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                                sigma=config["noise_std"] * np.ones(n_actions))
        vectorized_action_noise = VectorizedActionNoise(n_envs=config["n_envs"], base_noise=normal_action_noise)
        action_noise = vectorized_action_noise if config["n_env"] > 1 else normal_action_noise
    else:
        action_noise = None

    if algorithm in ("TD3", "DDPG"):
        model = TD3(config["policy_type"], env=get_env(config, config["stages"][0]),
                    verbose=1, seed=config["seed"],
                    tensorboard_log=f"runs/{run.id}", device="cuda",
                    replay_buffer_class=config["replay_buffer"],
                    # hyperparameters
                    train_freq=1 if config["n_envs"]> 1 else (1, "episode"), #config["n_envs"] if config["n_envs"] > 1 else (1, "episode"),
                    gradient_steps=config["gradient_steps"],
                    learning_starts=config["learning_starts"],
                    learning_rate=config["learning_rate"],
                    gamma=config["gamma"],
                    buffer_size=config["buffer_size"],
                    policy_kwargs=config["policy_kwargs"],
                    action_noise=action_noise

                    )
    elif algorithm == "SAC":
        model = SAC(config["policy_type"], env=get_env(config, config["stages"][0]),
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

    return model

def curriculum_learn(config: dict, initial_model: Optional[OffPolicyAlgorithm] = None,
                     starting_stage: Optional[str] = None, algorithm: str = "TD3"):
    env_name = config["env_name"]
    project = f"curriculum_learn_{env_name}"

    # set wandb tags
    tags = []
    tags.extend(str(x) for x in config["stages"])
    if initial_model:
        tags.append("pre-trained")
        tags.append("curriculum_learning")
    elif len(config["stages"]) > 1:
        tags.append("curriculum_learning")

    if config["n_envs"] > 1:
        tags.append("multi_env")

    tags.append(config["algorithm"])

    run = wandb.init(
        project=f"{project}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        dir="run_data",
        tags=tags
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    if not initial_model:
        model = get_model(algorithm, config, run)

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

    for stage, reward_threshold in zip(config["stages"], config["reward_thresholds"]):
        model.env = get_env(config, stage)

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
            total_timesteps=25_000,
            callback=[WandbCallback(
                model_save_path=wandb.run.dir,
                model_save_freq=20_000
            ), eval_callback],
        )

        eval_env.close()

    run.finish()
    return model



from typing import Optional

import gymnasium
import gymnasium as gym

from stable_baselines3 import SAC, TD3, PPO, DDPG, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback


# todo: make a new callback after calling evalCallback
#   After achieving a certain reward threshold stop training and start next stage
#   reward threshold should be sufficient since

def get_env(config, stage):
    if config["n_envs"] > 1:
        # rendering is not allowed in multiprocessing
        render, show_goal_space, show_debug_labels = False, False, False

        env = make_vec_env(config["env_name"], n_envs=config["n_envs"],
                           env_kwargs={"render": render, "control_type": config["control_type"],
                                       "reward_type": config["reward_type"],
                                       "show_goal_space": show_goal_space, "obstacle_layout": stage,
                                       "show_debug_labels": show_debug_labels}, vec_env_cls=SubprocVecEnv)
    else:
        show_goal_space, show_debug_labels = True if config["render"] else False

        env = gym.make(config["env_name"], render=config["render"], control_type=config["control_type"],
                       reward_type=config["reward_type"],
                       show_goal_space=show_goal_space, obstacle_layout=stage,
                       show_debug_labels=show_debug_labels)

    return env


def curriculum_learn(config: dict, eval_freq=5000, initial_model: Optional[OffPolicyAlgorithm] = None,
                     starting_stage: Optional[str] = None):
    env_name = config["env_name"]
    project = f"curriculum_learn_{env_name}"
    run = wandb.init(
        project=f"{project}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        dir="run_data",
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    if not initial_model:
        model = TD3(config["policy_type"], env=get_env(config, config["stages"][0]),
                    verbose=1, replay_buffer_class=config["replay_buffer"], seed=config["seed"],
                    tensorboard_log=f"runs/{run.id}", device="cuda", train_freq=config["n_envs"], gradient_steps=
                    config["gradient_steps"])
    else:
        model = initial_model
        stages:list = config["stages"]
        reward_thresholds = config["reward_thresholds"]

        assert starting_stage in stages

        idx = stages.index(starting_stage)
        config["stages"] = stages[idx:]
        config["reward_thresholds"] = reward_thresholds[idx:]



    # model = TD3.load(r"run_data/wandb/run_panda_reach_evade_obstacle_stage_2_best_run/files/model.zip", env=env,
    #                  device="cuda", train_freq=n_envs, gradient_steps=2, replay_buffer=replay_buffer)

    assert len(config["stages"]) == len(config["reward_thresholds"])

    for stage, reward_threshold in zip(config["stages"], config["reward_thresholds"]):
        model.env = get_env(config, stage)

        eval_env = gym.make(config["env_name"], render=True, control_type=config["control_type"],
                            reward_type=config["reward_type"],
                            show_goal_space=False, obstacle_layout=config["stages"][0],
                            show_debug_labels=True)

        stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)

        eval_callback = EvalCallback(eval_env, eval_freq=max(eval_freq // config["n_envs"], 1),
                                     callback_after_eval=stop_train_callback, verbose=1, n_eval_episodes=50)

        model.learn(
            total_timesteps=10_000_000,
            callback=[WandbCallback(
                model_save_path=wandb.run.dir,
                model_save_freq=20_000
            ), eval_callback],
        )

        eval_env.close()

    run.finish()

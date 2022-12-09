import os

import gymnasium as gym
import stable_baselines3
from stable_baselines3 import HerReplayBuffer, TD3, SAC, PPO
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback
import wandb
from torch.utils import tensorboard
import time

# noinspection PyUnresolvedReferences
import panda_gym


env_name = "PandaReach-v3"
total_timesteps = 50_000
render = False  # renders the pybullet env
goal_range = 0.3  # Size of the cuboid in which the goal is sampled
control_type = "js"  # "ee": end effector displacement; "js": joint angles
show_goal_space = False
replay_buffer = HerReplayBuffer
policy_type = "MultiInputPolicy"
show_debug_labels = True
learning_starts = 100

obstacle_layout = 2 # 0: No obstacles; 1: 1 small cube near ee; 2: 2 small cubes neighboring ee
joint_obstacle_observation = "closest"  # "all": closest distance to any obstacle of all joints is observed;
# "closest": only closest joint distance is observed

# env = gym.make(env_name, render=render, goal_range=goal_range, control_type=control_type,
#                show_goal_space=show_goal_space, obstacle_layout=obstacle_layout, show_debug_labels=show_debug_labels)



config = {
    "policy_type": policy_type,
    "total_timesteps": total_timesteps,
    "env_name": env_name,
    "replay_buffer_class": str(replay_buffer.__name__),
    "control_type": control_type,
    "max_ep_steps": 50
}

# register envs to gymnasium
panda_gym.register_envs(config["max_ep_steps"])

if __name__ == "__main__":
    env = make_vec_env(env_name, n_envs=1, seed=1, vec_env_cls=SubprocVecEnv, env_kwargs={"control_type":"js"})

    wandb.login(key=os.getenv("wandb_key"))
    # wandb.tensorboard.patch(root_logdir="run_data")
    run = wandb.init(
        project=f"{env_name}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        dir="run_data",
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    # todo: adjust learning starts argument
    # model = TD3(policy_type, env=env,
    #             verbose=1,
    #              tensorboard_log=f"runs/{run.id}", device="cuda")
    model = TD3.load(r"run_data/wandb/run-20221209_153453-1uf8wzs4/files/model.zip", env=env, device="cuda")
    tic = time.perf_counter()
    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            model_save_path=wandb.run.dir
        ),
    )
    toc = time.perf_counter()
    print(f"Learning Time: {toc-tic}")

    run.finish()

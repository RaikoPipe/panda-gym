import wandb
from stable_baselines3 import TD3

import gymnasium as gym
import numpy as np
from train_curr import config
from time import sleep

import panda_gym

from algorithms.SAC_hybrid.prior_controller_neo import RRMC


def evaluate(prior, model, num_steps=1000):
    """
    Evaluate a RL agent
    :param prior: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """

    episode_rewards = [0.0]
    obs, _ = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action = prior.compute_action_neo([0.07996564, -0.13340622, 0.02173809])
        rl_action, _ = model.predict(obs)

        obs, reward, done, truncated, info, = env.step(action)
        sleep(0.05)

        # Stats
        episode_rewards[-1] += reward
        if done or truncated:
            sleep(2)
            if info["is_success"]:
                print("Success!")
            elif info["is_truncated"]:
                print("Collision...")
            else:
                print("Timeout...")
            obs, _ = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = np.mean(episode_rewards[-100:])
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


panda_gym.register_envs(100)
# instantiate reachEvadeObstacle
# env = gym.make(config["env_name"], render=True, control_type=config["control_type"], reward_type=config["reward_type"],
#                show_goal_space=False, obstacle_layout=1,
#                show_debug_labels=True, limiter=config["limiter"])

# instantiate reach
# test the pyb_utils function
env = gym.make(config["env_name"], render=True, control_type=config["control_type"],
               obs_type=config["obs_type"], distance_threshold=config["distance_threshold"],
               reward_type=config["reward_type"], limiter=config["limiter"],
               show_goal_space=False, obstacle_layout="wall_parkour_1",
               show_debug_labels=False)

rrmc_neo = RRMC(env=env, collisions=[])
model = TD3.load(r"run_data/wandb/run_obs_layout_1_best_08_11/files/model.zip", env=env)

evaluate(rrmc_neo, model)

# todo:
#   solve equation with less constrains
#   double check input (target position)

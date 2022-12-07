import wandb
from stable_baselines3 import TD3

import gymnasium as gym
import numpy as np
from train_curr import config
from time import sleep

import panda_gym




def evaluate(model, num_steps=500):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs, _ = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        obs, reward, done, truncated, info, = env.step(action)
        sleep(0.05) # for human eval

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

panda_gym.register_envs(50)

env = gym.make(config["env_name"], render=True, control_type=config["control_type"], reward_type=config["reward_type"],
               show_goal_space=False, obstacle_layout="wall_parkour_1", obs_type=config["obs_type"],
               show_debug_labels=True, limiter=config["limiter"], distance_threshold=config["distance_threshold"])

model = TD3.load(r"run_data/wandb/run_panda_reach_evade_obstacle_wall_parkour_1_best_run/files/model.zip", env=env)

evaluate(model)
# run-20221128_154210-38h9a7q2 olive pond
#

# for i in range(100_000):
#     action,_=model.predict()
#     obs, reward, done, truncated, info, = env.step(action)
#     model.replay_buffer.add()

# let's say we did an episode for each env and do some training steps
# what would be the difference to doing it with subprocenv?



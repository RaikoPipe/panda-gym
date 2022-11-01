import wandb
from stable_baselines3 import TD3
import panda_gym
import gymnasium as gym
import numpy as np


def evaluate(model, num_steps=1000):
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

        obs, reward, done, info, _ = env.step(action)

        # Stats
        episode_rewards[-1] += reward
        if done:
            obs, _ = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = np.mean(episode_rewards[-100:])
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


env = gym.make("PandaReach-v3", render=True)
model = TD3.load(r"run_data/wandb/run-20221031_114132-134upbh8/files/model.zip", env=env)

evaluate(model)
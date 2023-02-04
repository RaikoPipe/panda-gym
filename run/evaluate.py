import wandb

import sys
import gymnasium
sys.modules["gym"] = gymnasium

from stable_baselines3 import TD3
from sb3_contrib import TQC

import numpy as np
from train_preo import config
from time import sleep

import gym

import panda_gym

from learning_methods.curriculum_learning import get_env



def evaluate(model, num_steps=500):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs, _ = env.reset()
    done_events = []
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        obs, reward, done, truncated, info, = env.step(action)
        #sleep(0.05) # for human eval

        # Stats
        episode_rewards[-1] += reward

        if done or truncated:
            #sleep(2)
            if info["is_success"]:
                print("Success!")
                done_events.append(1)
            elif info["is_truncated"]:
                print("Collision...")
                done_events.append(-1)
            else:
                print("Timeout...")
                done_events.append(0)
            obs, _ = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = np.mean(episode_rewards[-100:])

    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
    print(f"Success Rate: {done_events.count(1)/len(done_events)}")
    print(f"Collision Rate: {done_events.count(-1) / len(done_events)}")

    return mean_100ep_reward

panda_gym.register_envs(100)

#env = get_env(config, "cube_3_random")

env = gym.make(config["env_name"], render=True, control_type=config["control_type"],
               obs_type=config["obs_type"], goal_distance_threshold=config["goal_distance_threshold"],
               reward_type=config["reward_type"], limiter=config["limiter"],
               show_goal_space=False, obstacle_layout="cube_3_random",
               show_debug_labels=True)

model = TQC.load(r"run_data/wandb/run-20230204_153453-3k62p9ul/files/best_model.zip", env=env)

evaluate(model)
# run-20221128_154210-38h9a7q2 olive pond
#

# for i in range(100_000):
#     action,_=model.predict()
#     obs, reward, done, truncated, info, = env.step(action)
#     model.replay_buffer.add()

# let's say we did an episode for each env and do some learning_methods steps
# what would be the difference to doing it with subprocenv?



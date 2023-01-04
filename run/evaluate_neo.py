import wandb
from stable_baselines3 import TD3

import gymnasium as gym
import numpy as np
from train_preo import config
from time import sleep

import panda_gym

from algorithms.SAC_hybrid.prior_controller_neo import NEO


def evaluate(prior,  num_steps=10000):
    """
    Evaluate a RL agent
    :param prior: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """

    episode_rewards = [0.0]
    obs, _ = env.reset()
    done_events = []
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action = prior.compute_action(env.task.goal)# [0.07996564, -0.13340622, 0.02173809])
        #rl_action, _ = model.predict(obs)

        obs, reward, done, truncated, info, = env.step(action)
        #sleep(0.01)

        # Stats
        episode_rewards[-1] += reward
        if done or truncated:
            sleep(0.5)
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
            # unnecessary step (?) reset panda
            prior.panda_rtb.q = prior.panda_rtb.qr
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = np.mean(episode_rewards[-100:])
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
    print(f"Success Rate: {done_events.count(1)/len(done_events)}")
    print(f"Collision Rate: {done_events.count(-1) / len(done_events)}")

    return mean_100ep_reward


panda_gym.register_envs(200)
# instantiate reachEvadeObstacle
# env = gym.make(config["env_name"], render=True, control_type=config["control_type"], reward_type=config["reward_type"],
#                show_goal_space=False, obstacle_layout=1,
#                show_debug_labels=True, limiter=config["limiter"])

# instantiate reach
# test the pyb_utils function
env = gym.make(config["env_name"], render=True, control_type=config["control_type"],
               obs_type=config["obs_type"], goal_distance_threshold=config["goal_distance_threshold"],
               reward_type=config["reward_type"], limiter=config["limiter"],
               show_goal_space=False, obstacle_layout="neo_test_2",
               show_debug_labels=True)

rrmc_neo = NEO(env=env)
#model = TD3.load(r"run_data/wandb/run_obs_layout_1_best_08_11/files/model.zip", env=env)

evaluate(rrmc_neo) #, model)

import pybullet

import sys
import gymnasium
sys.modules["gym"] = gymnasium

import wandb
from stable_baselines3 import TD3


import gymnasium as gym
import numpy as np
from run.train import config
from time import sleep

import panda_gym

from algorithms.SAC_hybrid.prior_controller_neo import NEO


def evaluate(env, num_steps=10000):
    """
    Evaluate a RL agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """

    episode_rewards = [0.0]
    obs, _ = env.reset()
    done_events = []
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action = env.robot.compute_action_neo(env.task.goal, env.task.obstacles, env.task.collision_detector, "left")# [0.07996564, -0.13340622, 0.02173809])
        # action = env.robot.compute_action_neo_pybullet(env.task.goal, env.task.obstacles, env.task.collision_detector)
        # pybullet.removeAllUserDebugItems(physicsClientId=0)
        #rl_action, _ = model.predict(obs)
        env.robot.set_action(action)
        env.sim.step()
        obs, reward, done, truncated, info, = env.step(np.zeros(7))
        #sleep(0.01)

        # Stats
        episode_rewards[-1] += reward
        if done or truncated:

            if info["is_success"]:
                print("Success!")
                done_events.append(1)
            elif info["is_truncated"]:
                print("Collision...")
                #sleep(5)
                done_events.append(-1)
            else:
                print("Timeout...")
                done_events.append(0)
            obs, _ = env.reset()

            #sleep(0.01/20)


            episode_rewards.append(0.0)
        #pybullet.removeAllUserDebugItems(physicsClientId=1)
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


panda_gym.register_envs(800)
env = gym.make(config["env_name"], render=True, control_type="jsd",
               obs_type=config["obs_type"], goal_distance_threshold=config["goal_distance_threshold"],
               reward_type=config["reward_type"], limiter=config["limiter"],
               show_goal_space=False, scenario="narrow_tunnel",
               show_debug_labels=True, n_substeps=5, )


#rrmc_neo = NEO(env)

#model = TD3.load(r"run_data/wandb/run_obs_layout_1_best_08_11/files/model.zip", env=env)

evaluate(env) #, model)

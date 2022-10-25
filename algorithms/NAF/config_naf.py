import collections
import os
import time
from copy import copy

import torch
import numpy as np

from collections import deque

import wandb

from algorithms.NAF.agent import NAF_Agent
from env import PandaEnv
from utils import wandb_logging


# # def evaluate_policy(use_single_agent=True):
# #     global total_steps
# #
# #     rewards = []
# #     ep_ret = 0.0
# #     ep_len = 0
# #
# #     if env.render:
# #         env.mode_label.desc = "Evaluation"
# #
# #     for j in range(agent.num_eval_episodes):
# #         state, done = agent.test_env.reset(), False
# #         while not (done or (ep_len == agent.max_ep_steps)):
# #
# #             action = agent.act_without_noise(state)
# #
# #             state, reward, done, _ = agent.test_env.step(action)
# #
# #             ep_ret += reward
# #             ep_len += 1
# #             total_steps += 1
# #
# #             if env.render:
# #                 env.rewards_label.desc = "Rewards: " + str(ep_ret)
# #                 env.episode_steps_label.desc = "Episode: " + str(ep_len)
# #                 env.std_deviation_rewards.desc = "Standard Deviation Rewards: " + str(np.std(rewards))
# #                 env.current_manip_label.desc = "Current Manipulability: " + str(env.panda.manipulability(env.panda.q))
# #
# #         rewards.append(ep_ret)
# #         ep_len = 0
# #         ep_ret = 0
# #     avg_ret = np.mean(rewards)
# #     avg_len = ep_len / agent.num_eval_episodes
# #
# #     return {'rewards_eval': avg_ret,
# #             'len_eval': avg_len}
#
#
#
# def timer(start, end):
#     """ Helper to print training time """
#     hours, rem = divmod(end - start, 3600)
#     minutes, seconds = divmod(rem, 60)
#     print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
#
#
# def run(eval_every, num_steps):
#     """"NAF.
#
#     Params
#     ======
#
#     """
#
#     global total_steps
#     rewards = []  # list containing rewards from each episode
#     ep_rewards = deque(maxlen=100)
#     ep_len = 0
#     state = env.reset()
#     ep_ret = 0.0
#
#     ee_rewards = 0.0
#     obs_rewards = 0.0
#     manip_rewards = 0.0
#     e = 0.0
#
#     if env.render:
#         env.mode_label.desc = "Training"
#
#     for t in range(num_steps):
#         action = agent.get_action(state)
#
#         next_state, reward, done, code = agent.env.step(action)
#         agent.step(state, action, reward, next_state, done)
#
#         rewards.append(reward)
#
#         total_steps += 1
#         ep_len += 1
#         state = next_state
#         ep_ret += reward
#
#         ee_rewards += env.rew_end_eff_error
#         obs_rewards += env.rew_obs_dist
#         manip_rewards += env.rew_manip
#         e = env.e
#
#         if env.render:
#             # update env labels
#             env.rewards_label.desc = "Current Rewards: " + str(ep_ret)
#             env.episode_steps_label.desc = "Episode: " + str(ep_len)
#             env.std_deviation_rewards.desc = "Standard Deviation Rewards: " + str(np.std(rewards))
#             env.current_manip_label.desc = "Current Manipulability: " + str(env.panda.manipulability(env.panda.q))
#
#             env.rew_end_eff_error_label.desc= "EE Rewards: " +str(manip_rewards)
#             env.rew_obs_dist_label.desc = "Obstacle Distance Punish: " + str(obs_rewards)
#             env.rew_manip_label.desc = "Manipulability reward: " + str(manip_rewards)
#             env.e_label.desc = "End Effector Error: " + str(e)
#
#         if env.render and code == -2:
#             # pause to emphasize collision
#             time.sleep(1)
#
#
#
#         if (t+1) % agent.steps_per_epoch == 0:
#             metrics = evaluate_policy(agent)
#
#             wandb_logging.write_logs(metrics, total_steps)
#
#             if env.render:
#                 env.mode_label.desc = "Training"
#
#         if done:
#             ep_rewards.append(ep_ret)
#             print("\n")
#             print(f"Episode Rewards: {ep_ret}")
#             print(f"Episode Len: {ep_len}")
#             print(f"Average Episode Rewards: {np.mean(ep_rewards)}")
#             print(f"Termination Reason: {code_dict[code]}")
#
#             if env.render:
#                 # update average rewards
#                 env.avg_rewards_label.desc = "Average Rewards: " + str(np.mean(ep_rewards))
#
#             rewards = []
#             ep_len = 0
#             total_steps += 1
#             state = env.reset()
#             ep_ret = 0
#             ee_rewards = 0
#             obs_rewards = 0
#             manip_rewards = 0
#
#             wandb_logging.write_logs({'ep_rewards': reward}, total_steps)
#             wandb_logging.write_logs({'ep_length': ep_len}, total_steps)
#             wandb_logging.write_logs({'avg_rewards': np.mean(rewards)}, total_steps)




def get_naf_agent(env):

    TASK = "manipulation"

    PROJECT_LOG = "NAF"

    SEED = 0 # random seed

    NUM_STEPS = int(1e6) #: Number of training steps
    EVAL_EVERY = 5000 #: Evaluate current policy every X steps
    NUM_EVAL_EPISODES = 10 #: Number of evaluation runs
    BATCH_SIZE = 128 #: Batch size sampled from replay buffer
    LAYER_SIZE = 256 #: NN hidden layer size
    GAMMA = .99 # Discount Factor
    LEARNING_RATE = 1e-3
    LOSS_TYPE = "mse"

    # method specific parameters
    METHOD = "NAF"
    USE_PER = True # Use prioritized experience replay
    NSTEP = 1 #: nstep bootstraping ???
    D2RL = 1 #: Use Deep Dense NN Architecture ???
    TAU = 0.005 # soft update factor
    UPDATE_EVERY = 1 # Update frequency
    CLIP_GRAD = 1.0 # clip gradients

    # set up wandb
    variables = copy(locals())
    HYPERS = wandb_logging.get_ordered_hypers(variables)
    run_name = wandb_logging.start_wandb(PROJECT_LOG, TASK, SEED, METHOD, HYPERS)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    total_steps = 0

    agent = NAF_Agent(env_fn = lambda: env, method=METHOD,
                      d2rl=D2RL,
                      device=device,
                      tau=TAU,
                      gamma=GAMMA,
                      n_step=NSTEP,
                      update_every=UPDATE_EVERY,
                      n_updates=1,
                      batch_size=BATCH_SIZE,
                      per=USE_PER,
                      clip_grad=CLIP_GRAD,
                      learning_rate=LEARNING_RATE,
                      layer_size=LAYER_SIZE,
                      seed=SEED,
                      loss=LOSS_TYPE,
                      num_eval_episodes=NUM_EVAL_EPISODES
                      )

    return agent, run_name

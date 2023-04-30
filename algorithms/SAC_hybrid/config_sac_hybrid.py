
import collections
import itertools
import math
import random
import sys
import time
import traceback
from copy import deepcopy, copy

import numpy as np
import torch
from torch.distributions.normal import Normal
from wandb_utils import wandb_logging


import algorithms.SAC_hybrid.core as core
from algorithms.SAC_hybrid.agent import SAC_Agent
from algorithms.SAC_hybrid.prior_controller_neo import NEO



# # ---------------------------------------------------------------------------------------------------------------------------------------------#
# # --------------------------------------------------------- SAC_hybrid -------------------------------------------------------------------------------#
# # ---------------------------------------------------------------------------------------------------------------------------------------------#
#
#
#
# # ---------------------------------------------------------------------------------------------------------------------------------------------#
# # --------------------------------------------------------- Helpers ---------------------------------------------------------------------------#
# # ---------------------------------------------------------------------------------------------------------------------------------------------#
#
# def evaluate_prior():
#     rewards = []
#     ep_len = 0.0
#
#     for i in range(agents[0].num_test_episodes):
#         state, done = agents[0].test_env.reset(), False
#         while not (done or (ep_len == agents[0].max_ep_steps)):
#             act = prior.compute_action()
#             state, reward, done, _ = env.step(act)
#
#             rewards.append(reward)
#             ep_len += 1
#         print(i)
#         sys.stdout.flush()
#         ep_len = 0.0
#     avg_ret = np.mean(rewards)
#     avg_len = ep_len / agents[0].num_test_episodes
#
#     return {'rewards_eval': avg_ret,
#             'len_eval': avg_len}
#
# def fuse_ensembles_deterministic(ensemble_actions):
#     actions = torch.tensor([ensemble_actions[i][0] for i in range(NUM_AGENTS)])
#     mu = torch.mean(actions, dim=0)
#     var = torch.var(actions, dim=0)
#     sigma = np.sqrt(var)
#     return mu, sigma
#
#
# def fuse_ensembles_stochastic(ensemble_actions):
#     mu = (np.sum(np.array([ensemble_actions[i][0] for i in range(NUM_AGENTS)]), axis=0)) / NUM_AGENTS
#     var = (np.sum(
#         np.array([(ensemble_actions[i][1] ** 2 + ensemble_actions[i][0] ** 2) - mu ** 2 for i in range(NUM_AGENTS)]),
#         axis=0)) / NUM_AGENTS
#     sigma = np.sqrt(var)
#     return torch.from_numpy(mu), torch.from_numpy(sigma)
#
# def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma):
#     # The policy mu and sigma are from the stochastic SAC_hybrid output
#     # The sigma from prior is fixed
#     mu = (np.power(policy_sigma, 2) * prior_mu + np.power(prior_sigma, 2) * policy_mu) / (
#                 np.power(policy_sigma, 2) + np.power(prior_sigma, 2))
#     sigma = np.sqrt(
#         (np.power(prior_sigma, 2) * np.power(policy_sigma, 2)) / (np.power(policy_sigma, 2) + np.power(prior_sigma, 2)))
#     return mu, sigma
#
# def arg(tag, default):
#     HYPERS[tag] = type(default)((sys.argv[sys.argv.index(tag) + 1])) if tag in sys.argv else default
#     return HYPERS[tag]
#
#
# # ---------------------------------------------------------------------------------------------------------------------------------------------#
# # --------------------------------------------------------- Run -------------------------------------------------------------------------------#
# # ---------------------------------------------------------------------------------------------------------------------------------------------#
#
# def train(agents, env):
#     # Prepare for interaction with environment
#     global total_steps
#     state, ep_ret, ep_len = env.reset(), 0, 0
#     agent: SAC_Agent = random.choice(agents)
#     reward = 0
#     rewards = [] # rewards for every single step
#     ep_rewards = collections.deque(maxlen=100) # rewards for every episode
#     old_state = state
#
#     if env.render:
#         env.mode_label.desc = "Training"
#
#     # Main loop: collect experience in env and update/log each epoch
#     for t in range(NUM_STEPS):
#
#         mu_prior = prior.compute_action()
#
#         if t > agent.start_steps:
#             action, policy_action = agent.get_action(old_state=old_state, state=state, reward=reward)
#         else:
#             action = env.action_space.sample()
#             if METHOD == "residual":
#                 policy_action = action
#                 action = np.clip(mu_prior + policy_action, -1, 1)
#
#         new_state, reward, done, code = env.step(action)
#
#         mu_prior2 = prior.compute_action()
#         ep_ret += reward
#         rewards.append(reward)
#         ep_len += 1
#         total_steps += 1
#
#         if env.render:
#             # update env labels
#             env.rewards_label.desc = "Current Rewards: " + str(ep_ret)
#             env.episode_steps_label.desc = "Episode: " + str(ep_len)
#             env.std_deviation_rewards.desc = "Standard Deviation Rewards: " + str(np.std(rewards))
#             env.current_manip_label.desc = "Current Manipulability: " + str(env.panda.manipulability(env.panda.q))
#
#         if env.render and code == -2:
#             # pause to show collision
#             time.sleep(1)
#
#         # Ignore the "done" signal if it comes from hitting the time
#         # horizon (that is, when it's an artificial terminal signal
#         # that isn't based on the agent's state)
#         done = False if ep_len == agent.max_ep_steps else done
#
#         # Store experience to replay buffer
#         if METHOD == "residual":
#             agent.replay_buffer.store(state, policy_action, reward, new_state, done, mu_prior, mu_prior2)
#         else:
#             agent.replay_buffer.store(state, action, reward, new_state, done, mu_prior, mu_prior2)
#
#         old_state = state
#         state = new_state
#
#         # Update handling
#         if t >= agent.update_after:
#             for ag in agents:
#                 batch = agent.replay_buffer.sample_batch(agent.batch_size)
#                 metrics = ag.update(batch)
#             write_logs(metrics, total_steps)
#
#         # End of epoch handling
#         if (t + 1) % agent.steps_per_epoch == 0:
#             # Test the performance of the deterministic version of the agent.
#
#             metrics = evaluate_policy(agent)
#
#             if env.render:
#                 env.mode_label.desc = "Training"
#
#             write_logs(metrics, total_steps)
#             save_ensemble()
#
#         # End of trajectory handling
#         if done:
#             ep_rewards.append(ep_ret)
#             print("\n")
#             print(f"Episode Rewards: {ep_ret}")
#             print(f"Episode Len: {ep_len}")
#             print(f"Average Episode Rewards: {np.mean(ep_rewards)}")
#             print(f"Termination Reason: {const.code_dict[code]}")
#             write_logs({'ep_rewards': ep_ret}, total_steps)
#             write_logs({'ep_length': ep_len}, total_steps)
#             write_logs({'avg_rewards': np.mean(ep_rewards)}, total_steps)
#
#             if env.render:
#                 # update average rewards
#                 env.avg_rewards_label.desc = "Average Rewards: " + str(np.mean(ep_rewards))
#
#             state, ep_ret, ep_len, rewards = env.reset(), 0, 0, []
#             agent = random.choice(agents)


# ---------------------------------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------- Setup -----------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------#
NUM_AGENTS = 1  # Number of agents in ensemble
def get_sac_agent(env, max_ep_steps):

    TASK = "manipulation"

    SEED = 12

    PROJECT_LOG = "SAC_hybrid"

    EVAL_EVERY = 5000 #: Evaluate current policy every X steps
    NUM_EVAL_EPISODES = 10 #: Number of evaluation runs
    BATCH_SIZE = 128 #: Batch size sampled from replay buffer
    LAYER_SIZE = 256 #: NN hidden layer size
    GAMMA = .99 # Discount Factor
    LEARNING_RATE = 1e-3
    LOSS_TYPE = "mse"

    # env parameters
    OBSTACLE_MODE = 1
    REWARD_TYPE="hybrid_control"

    # method specific parameters
    METHOD = "BCF"  # Options: policy, BCF, residual, CORE-RL
    USE_KL = True
    NUM_AGENTS = 1  # Number of agents in ensemble
    NUM_AGENTS = NUM_AGENTS if METHOD == "BCF" else 1
    ALPHA =  0.5
    BETA = 0.1
    EPSILON = 2e-4
    TARGET_KL_DIV = 10e-3
    TARGET_ENTROPY = -7
    SIGMA_PRIOR = 0.4
    PRIOR_CONTROLLER = "APF"
    # CORE-RL parameters
    LAMBDA_MAX = 15.0
    FACTOR_C = 0.3

    # set up wandb
    variables = copy(locals())
    HYPERS = wandb_logging.get_ordered_hypers(variables)
    run_name = wandb_logging.start_wandb(PROJECT_LOG, TASK, SEED, METHOD, HYPERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    torch.set_num_threads(torch.get_num_threads())

    prior = NEO(env)
    sigma_prior = SIGMA_PRIOR

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Initialise an ensemble of agents
    agent = SAC_Agent(lambda: env, prior=prior,
                        actor_critic=core.MLPActorCritic,
                        gamma=0.99,
                        polyak=0.995,
                        lr=1e-3,
                        batch_size=100,
                        start_steps=10000,
                        num_eval_episodes=10,
                        alpha=ALPHA,
                        beta=BETA,
                        epsilon=EPSILON,
                        use_kl_loss=USE_KL,
                        target_entropy=TARGET_ENTROPY,
                        target_KL_div=TARGET_KL_DIV,
                        sigma_prior=sigma_prior,
                        device=device,
                        method=METHOD,
                      max_ep_steps=max_ep_steps)

    return agent, run_name

    # test
    # agent = agents[0]
    # agent.ac.pi = torch.load(r"saved_models/manipulation_SEED_12_residual_Mon_Sep_19_16_06_47_2022manipulation_SEED_12_residual_Mon_Sep_19_16_06_47_2022_0.pth")
    # agents = [agent]
    # env.render= True
    # ep_ret = 0.0
    # tic = time.perf_counter()
    # print(evaluate_policy())
    # toc = time.perf_counter()
    #
    # print(toc-tic)
    # #print(evaluate_prior())


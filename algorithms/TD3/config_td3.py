from copy import deepcopy, copy
import itertools
import numpy as np
import torch

from algorithms.TD3 import core
from algorithms.TD3.agent import TD3_Agent
from wandb_utils import wandb_logging


#
# def run(agent):
#     # Prepare for interaction with environment
#     total_steps = agent.steps_per_epoch * agent.epochs
#     start_time = time.time()
#     state, ep_ret, ep_len = env.reset(), 0, 0
#
#     # Main loop: collect experience in env and update/log each epoch
#     for t in range(total_steps):
#
#         # Until start_steps have elapsed, randomly sample actions
#         # from a uniform distribution for better exploration. Afterwards,
#         # use the learned policy (with some noise, via act_noise).
#         if t > agent.start_steps:
#             action = agent.get_policy_action(state, agent.act_noise)
#         else:
#             action = env.action_space.sample()
#
#         # Step the env
#         next_state, reward, done, _ = env.step(action)
#         ep_ret += reward
#         ep_len += 1
#
#         # Ignore the "done" signal if it comes from hitting the time
#         # horizon (that is, when it's an artificial terminal signal
#         # that isn't based on the agent's state)
#         done = False if ep_len == agent.env.max_ep_steps else done
#
#         # Store experience to replay buffer
#         agent.replay_buffer.store(state, action, reward, next_state, done)
#
#         # Super critical, easy to overlook step: make sure to update
#         # most recent observation!
#         state = next_state
#
#         # End of trajectory handling
#         if done or (ep_len == max_ep_steps):
#             logger.store(EpRet=ep_ret, EpLen=ep_len)
#             state, ep_ret, ep_len = env.reset(), 0, 0
#
#         # Update handling
#         if t >= agent.update_after and t % agent.update_every == 0:
#             for j in range(agent.update_every):
#                 batch = agent.replay_buffer.sample_batch(agent.batch_size)
#                 agent.update(data=batch, timer=j)
#
#         # End of epoch handling
#         if (t + 1) % agent.steps_per_epoch == 0:
#             epoch = (t + 1) // agent.steps_per_epoch
#
#             # Save model
#             if (epoch % save_freq == 0) or (epoch == agent.epochs):
#                 #logger.save_state({'env': env}, None)
#                 pass
#
#             # Test the performance of the deterministic version of the agent.
#             test_agent()
#
#             # Log info about epoch
#             # logger.log_tabular('Epoch', epoch)
#             # logger.log_tabular('EpRet', with_min_and_max=True)
#             # logger.log_tabular('TestEpRet', with_min_and_max=True)
#             # logger.log_tabular('EpLen', average_only=True)
#             # logger.log_tabular('TestEpLen', average_only=True)
#             # logger.log_tabular('TotalEnvInteracts', t)
#             # logger.log_tabular('Q1Vals', with_min_and_max=True)
#             # logger.log_tabular('Q2Vals', with_min_and_max=True)
#             # logger.log_tabular('LossPi', average_only=True)
#             # logger.log_tabular('LossQ', average_only=True)
#             # logger.log_tabular('Time', time.time() - start_time)
#             # logger.dump_tabular()


def get_td3_agent(env):
    TASK = "manipulation"

    SEED = 12

    PROJECT_LOG = "TD3"

    NUM_STEPS = int(1e6) #: Number of training steps
    EVAL_EVERY = 5000 #: Evaluate current policy every X steps
    NUM_EVAL_EPISODES = 10 #: Number of evaluation runs
    BATCH_SIZE = 128 #: Batch size sampled from replay buffer
    LAYER_SIZE = 256 #: NN hidden layer size
    GAMMA = .99 # Discount Factor
    LEARNING_RATE = 1e-3
    LOSS_TYPE = "mse"
    MAX_EP_STEPS=300

    # method specific parameters
    METHOD = "TD3"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # set up wandb
    variables = copy(locals())
    HYPERS = wandb_logging.get_ordered_hypers(variables)
    run_name = wandb_logging.start_wandb(PROJECT_LOG, TASK, SEED, METHOD, HYPERS)

    return TD3_Agent(lambda: env, actor_critic=core.MLPActorCritic,
        gamma=GAMMA, seed=SEED, epochs=100, max_ep_steps=MAX_EP_STEPS), run_name
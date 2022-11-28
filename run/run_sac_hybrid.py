"""
Author: Richard Reider, SpinningUp
Project: Panda Gym

"""
import time
import traceback
from collections import deque

import numpy as np

import torch
import os
import gymnasium as gym
import panda_gym

from algorithms.SAC_hybrid import config_sac_hybrid
from wandb_utils import wandb_logging
from train_curr import config

from utils.learning import get_env

code_dict = {1: "Reached Goal", -2: "Collision", -1: "Max Timesteps reached"}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = f"{ROOT_DIR}/saved_models"


def save_model():
    torch.save(agent.networks.pi, save_dir + ".pth")


def evaluate_policy():
    global total_steps

    rewards = []
    ep_ret = 0.0
    ep_len = 0



    # if env_eval.render:
    #     env_eval.mode_label.desc = "Evaluation"

    for j in range(agent.num_eval_episodes):
        state, done = agent.test_env.reset()[0]["observation"], False
        while not (done or (ep_len == agent.max_ep_steps)):

            action = agent.get_action_eval(state)

            state, reward, done, _, _ = agent.test_env.step(action)
            state = state["observation"]

            ep_ret += reward
            ep_len += 1

            # if env_eval.render:
            #     env_eval.rewards_label.desc = "Rewards: " + str(ep_ret)
            #     env_eval.episode_steps_label.desc = "Episode: " + str(ep_len)
            #     env_eval.std_deviation_rewards.desc = "Standard Deviation Rewards: " + str(np.std(rewards))
            #     env_eval.current_manip_label.desc = "Current Manipulability: " + str(
            #         env_eval.panda.manipulability(env_eval.panda.q))

        rewards.append(ep_ret)
        ep_len = 0
        ep_ret = 0
    avg_ret = np.mean(rewards)
    avg_len = ep_len / agent.num_eval_episodes

    return {'rewards_eval': avg_ret,
            'len_eval': avg_len}


def run():
    """Agent needs to implement the following methods:
    - update_agent
    - get_action
    - get_sample
    - add_to_replay_buffer"""
    # Prepare for interaction with environment
    total_steps = agent.steps_per_epoch * agent.epochs

    start_time = time.time()

    env_run = agent.env

    state, ep_ret, ep_len = env_run.reset()[0]["observation"], 0, 0

    rewards = []  # list containing reward for each step
    reward = 0
    ep_rewards = deque(maxlen=100)  # list containing reward for each episode

    # other metrics
    ee_rewards = 0.0
    obs_rewards = 0.0
    manip_rewards = 0.0
    e = 0.0

    # env_run.mode_label.desc = "Exploring"

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        old_state = state

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > agent.start_steps:
            kwargs = {"old_state": old_state, "reward": reward}
            action = agent.get_action(state, kwargs)
        else:
            action = agent.get_sample()

        # Step the env
        next_state, reward, done, truncated, info = env_run.step(action)
        next_state = next_state["observation"]
        ep_ret += reward
        ep_len += 1

        # ee_rewards += env_run.rew_end_eff_error
        # obs_rewards += env_run.rew_obs_dist
        # manip_rewards += env_run.rew_manip
        # e = env_run.e

        # Store experience to replay buffer
        agent.add_to_replay_buffer(state, action, reward, next_state, done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        state = next_state

        # if t == agent.start_steps + 1:
        #     env_run.mode_label.desc = "Training"
        # if env_run.render:
        #     # update env labels
        #     env_run.rewards_label.desc = "Current Rewards: " + str(ep_ret)
        #     env_run.episode_steps_label.desc = "Episode: " + str(ep_len)
        #     env_run.std_deviation_rewards.desc = "Standard Deviation Rewards: " + str(np.std(rewards))
        #     env_run.current_manip_label.desc = "Current Manipulability: " + str(
        #         env_run.panda.manipulability(env_run.panda.q))
        #
        #     env_run.rew_end_eff_error_label.desc = "EE Rewards: " + str(manip_rewards)
        #     env_run.rew_obs_dist_label.desc = "Obstacle Distance Punish: " + str(obs_rewards)
        #     env_run.rew_manip_label.desc = "Manipulability reward: " + str(manip_rewards)
        #     env_run.e_label.desc = "End Effector Error: " + str(e)

        # if env_run.render and code == -2:
        #     # pause to emphasize collision
        #     time.sleep(1)

        # End of trajectory handling
        if done or truncated or ep_len >= agent.max_ep_steps:
            ep_rewards.append(ep_ret)
            print("\n")
            print(f"Episode Rewards: {ep_ret}")
            print(f"Episode Len: {ep_len}")
            print(f"Average Episode Rewards: {np.mean(ep_rewards)}")
            #print(f"Termination Reason: {code_dict[code]}")

            # if env.render:
            #     # update average rewards
            #     env.avg_rewards_label.desc = "Average Rewards: " + str(np.mean(ep_rewards))

            # ee_rewards, obs_rewards, manip_rewards = 0, 0, 0

            wandb_logging.write_logs({'ep_rewards': ep_ret}, t)
            wandb_logging.write_logs({'ep_length': ep_len}, t)
            wandb_logging.write_logs({'avg_rewards': np.mean(ep_rewards)}, t)

            state, ep_ret, ep_len = env_run.reset()[0]["observation"], 0, 0

        # Update handling
        if t >= agent.update_after and t % agent.update_every == 0:
            agent.update_agent()

        # End of epoch handling
        if (t + 1) % agent.steps_per_epoch == 0:
            epoch = (t + 1) // agent.steps_per_epoch

            # Save model
            if (epoch % agent.save_freq == 0) or (epoch == agent.epochs):
                save_model()
                pass

            # Test the performance of the deterministic version of the agent.
            # env_run.mode_label.desc = "Exploring"
            metrics = evaluate_policy()
            wandb_logging.write_logs(metrics, t)
            # env_run.mode_label.desc = "Training"

            # Log info about epoch
            # logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            # logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            # logger.log_tabular('Time', time.time() - start_time)
            # logger.dump_tabular()


if __name__ == "__main__":
    # set up env parameters
    OBSTACLE_MODE = 1
    REWARD_TYPE = ""
    MAX_EP_STEPS = 500

    env = get_env(config, stage=1)
    env.reset()

    # get agent
    agent, run_name = config_sac_hybrid.get_sac_agent(env)

    save_dir = fr"{ROOT_DIR}/saved_models/{agent.alg_name}/{run_name}"

    wandb_logging.watch_agent(agent.networks)

    # run agent
    #try:
    run()
        #smtp_handler.subject = "Success"
        #logger.warning(0, "Success!")
    #except Exception as e:
        #logger.exception("Unhandled Exception!")
        #traceback.print_exception(e)

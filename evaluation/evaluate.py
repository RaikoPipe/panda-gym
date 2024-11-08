import logging
import os.path
import sys
import time

import gymnasium

from classes.train_config import TrainConfig

sys.modules["gym"] = gymnasium

from sb3_contrib import TQC

import numpy as np
#from training.train import configuration
from time import sleep
from torch.distributions.normal import Normal
import torch
import pandas as pd

import panda_gym
from tqdm import tqdm
import json
from copy import copy, deepcopy
from multiprocessing import Process
import panda_gym
import pybullet
from evaluation.ensemble_utils import action_selection
from utils import load_model_util
from classes.train_config import TrainConfig
import yaml

def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma):
    # The policy mu and sigma are from the stochastic SAC output
    # The sigma from prior is fixed
    mu = (np.power(policy_sigma, 2) * prior_mu + np.power(prior_sigma, 2) * policy_mu) / (
            np.power(policy_sigma, 2) + np.power(prior_sigma, 2))
    sigma = np.sqrt(
        (np.power(prior_sigma, 2) * np.power(policy_sigma, 2)) / (np.power(policy_sigma, 2) + np.power(prior_sigma, 2)))
    return mu, sigma


def visualize_trajectory(env, done_event, trajectory):
    try:
        env.task.sim.physics_client.removeBody(env.task.sim._bodies_idx["sphere_temp"])
    except:
        pass

    env.task.sim.remove_all_debug_text()
    # if env.task.show_goal_space:
    #     env.task.create_goal_outline()  # workaround

    # hide goal
    # env.task.sim.set_base_pose("target", np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    # change visual shape of robot

    # path_suc2 = path_success[:50]
    # done_suc2 = np.ones(50)  # metrics["done_events"][100:200]
    # done_normal = done_events[100:200]
    # path_normal = ee_pos[100:200]

    color = 1.0
    traj = deepcopy(trajectory)
    final_pos = copy(traj[-1])
    xyz1 = traj.pop()

    done_event_color = {1: np.array([0.0, 1.0, 0.0, 1.0]),
                        0: np.array([1.0, 1.0, 0.0, 1.0]),
                        -1: np.array([1.0, 0.0, 0.0, 1.0])}

    env.task.sim.create_sphere(
        body_name="sphere_temp",
        radius=0.01,
        mass=0.0,
        ghost=True,
        position=final_pos,
        rgba_color=done_event_color[done_event],
    )

    while traj:
        xyz2 = traj.pop()
        pybullet.addUserDebugLine(lineFromXYZ=xyz1, lineToXYZ=xyz2, lineColorRGB=np.array([1 - color, color, 0]),
                                  physicsClientId=env.sim.physics_client._client, lifeTime=0, lineWidth=2)
        color = len(traj) / len(trajectory)  # 1/max_ep_steps
        xyz1 = xyz2


def perform_benchmark(models, env, human=True, num_episodes=1000, deterministic=True,
                      strategy=None, scenario_name="", prior_orientation=None, pre_calc_metrics=None,
                      show_model_actions=False, configuration=TrainConfig(safety_distance=0.0)):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    # robot parameters
    # env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])
    # env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])

    episode_rewards = [0.0]

    # if goals_to_achieve:
    #     env.task.goal = goals_to_achieve.pop(0)

    done_events = []

    goals = []
    goals.append(env.task.goal)

    end_effector_positions = []
    path_success = []
    end_effector_velocities = []
    end_effector_speeds = []

    efforts = []
    jerks = []
    manipulabilities = []

    total_index_count = []
    episode_index_count = []
    ep_lengths = []
    ep_lengths_success = []

    # episode_action_sovereignty = 0
    action_sovereignty = None

    model_colors = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 0.0, 1.0]), 2: np.array([0.0, 0.0, 1.0]),
                    3: np.array([0.0, 0.0, 1.0]), 4: np.array([0.0, 0.0, 1.0])}
    seed = 0
    obs, _ = env.reset(seed=seed)
    for i in tqdm(range(num_episodes), desc=scenario_name):
        episode_reward = 0.0
        ee_pos = []
        ee_speed = []
        ep_length = 0
        total_effort = 0.0
        total_manipulability = 0.0
        jerk = []

        seed += 1
        # get next goal
        # env.task.goal = np.array(goals_to_achieve.pop(0))
        if human:
            # update goal location
            env.task.sim.set_base_pose("target", env.task.goal, np.array([0.0, 0.0, 0.0, 1.0]))
            if pre_calc_metrics:  # of precalculated metrics are given, visualize
                visualize_trajectory(env, pre_calc_metrics["done_events"][i],
                                     pre_calc_metrics["end_effector_positions"][i])

        while True:
            actions = []
            distribution_stds = []
            distribution_variances = []
            variances = []

            if models == "prior":
                #if prior_orientation == "fkine":
                prior_orientation = env.robot.inverse_kinematics(link=11, position=env.task.goal)[:7]
                action = env.robot.compute_action_neo(env.task.goal, env.task.obstacles, env.task.collision_detector,
                                                      prior_orientation, strategy)
            else:
                # get rl action distribution
                for model in models:
                    action, _states = model.predict(obs, deterministic=deterministic)

                    distribution_std = model.actor.action_dist.distribution.stddev.squeeze().cpu().detach().numpy()
                    distribution_variance = model.actor.action_dist.distribution.variance.squeeze().cpu().detach().numpy()
                    actions.append(action)
                    variances.append(distribution_variance)
                    distribution_variances.append(np.sum(distribution_variance))
                    distribution_stds.append(distribution_std)

            if strategy == "mean_actions":
                action = action_selection.mean(actions)
            elif strategy == "highest_confidence":
                action, index = action_selection.confidence(actions, variances)
                if show_model_actions:
                    env.task.sim.create_debug_text(f"Model Actions",
                                                   f">>>Action: Model {index}<<<",
                                                   color=model_colors[index])
            elif strategy == "weighted_aggregation":
                action = action_selection.weighted_aggregation(variances, actions)

            elif strategy == "bayesian_fusion":
                action = action_selection.bayesian_fusion(actions, variances)

            elif strategy == "prior_bcf":
                rl_sigma = distribution_stds[0]
                rl_mu = variances[0]

                prior_action = env.robot.compute_action_neo(env.task.goal, env.task.obstacles,
                                                            env.task.collision_detector, "")

                mu_mcf, std_mcf = fuse_controllers(prior_action, 0.4, rl_mu, rl_sigma)

                dist_hybrid = Normal(torch.tensor(mu_mcf).double().detach(), torch.tensor(std_mcf).double().detach())

                action = dist_hybrid.sample()
                action = torch.tanh(action).numpy()
            elif strategy == "prior_sum":
                prior_action = env.robot.compute_action_neo(env.task.goal, env.task.obstacles,
                                                            env.task.collision_detector, "back")
                action = np.add(actions[0], prior_action) / 2
            elif strategy == "prior":
                prior_action = env.robot.compute_action_neo(env.task.goal, env.task.obstacles,
                                                            env.task.collision_detector,
                                                            "")
                zipped = [sorted(zip(i, y)) for i, y in
                          zip([distribution_variances[0], 0.4], [actions[0], prior_action])]
                action = zipped[1][0]

            total_index_count.append(action_sovereignty)
            episode_index_count.append(action_sovereignty)

            if not isinstance(models[0], str) and show_model_actions:
                for i in range(len(models)):
                    env.task.sim.create_debug_text(f"Model Action {i}",
                                                   f">>>Model 0 Variance: {distribution_variances[i]}<<<",
                                                   color=model_colors[i])

            # if action_selection_strategy == "first_come":
            #     # The first to reach 5 has sovereignty for the rest of the episode
            #     if len(episode_index_count) == 9:
            #         if episode_index_count.count(0) >= 5:
            #             episode_action_sovereignty = 0
            #         else:
            #             episode_action_sovereignty = 1
            #     elif len(episode_index_count) < 9:
            #         episode_action_sovereignty = action_sovereignty
            #
            #     action_sovereignty = episode_action_sovereignty

            obs, reward, done, truncated, info, = env.step(action)

            if human:
                sleep(0.01/2)  # for human eval

            # add results and metrics
            episode_reward += reward

            total_effort += env.task.get_norm_effort()
            jerk.append(env.task.get_norm_jerk())

            total_manipulability += env.task.manipulability
            ee_pos.append(env.robot.get_ee_position())
            ee_speed.append(np.linalg.norm(env.robot.get_ee_velocity()))
            ep_length += 1

            if done or truncated:
                # sleep(2)
                if info["is_success"]:
                    # print("Success!")
                    done_events.append(1)
                    ep_lengths_success.append(ep_length)
                    jerks.append(np.array(jerk))
                    efforts.append(total_effort / ep_length)
                    manipulabilities.append(total_manipulability / ep_length)
                    end_effector_speeds.append(ee_speed)
                    path_success.append(ee_pos)
                elif info["is_truncated"]:
                    # print("Collision...")
                    done_events.append(-1)

                else:
                    # print("Timeout...")
                    done_events.append(0)
                obs, _ = env.reset(seed=seed)

                episode_index_count = []

                # add episode results and metrics
                episode_rewards.append(episode_reward)
                ep_lengths.append(ep_length)

                end_effector_positions.append(ee_pos)

                # finish episode
                break

            # episode_action_sovereignty = 0
        # evaluation_step(efforts, done_events, end_effector_positions, end_effector_velocities, episode_rewards,
        #                 goals, human, joint_positions, joint_velocities, manipulabilities, model, obs, goals_to_achieve)
    # Compute mean reward for the last 100 episodes

    results = {"mean_reward": np.round(np.mean(episode_rewards), 4),
               "success_rate": np.round(done_events.count(1) / len(done_events), 4),
               "collision_rate": np.round(done_events.count(-1) / len(done_events), 4),
               "timeout_rate": np.round(done_events.count(0) / len(done_events), 4),
               "num_episodes": np.round(len(done_events), 4),
               "mean_ep_length": np.round(np.mean(ep_lengths), 4),
               "mean_ep_length_success": np.round(np.mean(ep_lengths_success), 4),
               "mean_num_sim_steps": np.round(np.mean([i * configuration.n_substeps for i in ep_lengths]), 4),
               "mean_num_sim_steps_success": np.round(
                   np.mean([i * configuration.n_substeps for i in ep_lengths_success]), 4),
               "mean_effort": np.round(np.mean(efforts), 4),
               "mean_manipulability": np.round(np.mean(manipulabilities), 4),
               "mean_norm_jerk": np.round(np.mean(np.array([item for l in jerks for item in l])), 4),
               "mean_ee_speed": np.round(np.mean(np.array([item for l in end_effector_speeds for item in l])), 4)
               }

    idx = 0
    for _ in models:
        results[f"model_{idx}_action_rate"] = total_index_count.count(idx) / num_episodes
        idx = idx + 1

    metrics = {
        "end_effector_positions": end_effector_positions,
        "end_effector_speeds": end_effector_speeds,
        "end_effector_velocities": end_effector_velocities,
        "path_success": path_success,
        "goals": goals,
        "jerks": jerks,
        "done_events": done_events
    }

    return results, metrics

def evaluate_ensemble(ensemble_name, human=False, eval_type="basic", strategy="mean_actions",
                      obstacle_observation=None, num_episodes=100,
                      evaluation_scenarios=None):

    ensemble_model_paths = load_model_util.get_group_model_paths(ensemble_name)
    ensemble_yaml_paths = load_model_util.get_group_yaml_paths(ensemble_name)

    configuration = load_model_util.get_train_config_from_yaml(ensemble_yaml_paths[0])

    configuration.render = human

    # get agent type
    if obstacle_observation is None:
        obstacle_observation = {'obstacles': "vectors", 'prior': None}

    panda_gym.register_reach_ao(150)

    agent_type = None
    if len(ensemble_model_paths) > 1:
        agent_type = "ensemble"
    elif len(ensemble_model_paths) == 1:
        agent_type = "single"
    elif ensemble_name == "prior":
        agent_type = "prior"
    else:
        logging.error("No agents to evaluate")
        return

    logging.info(f"Evaluating {ensemble_name}")

    with open("scenario_goals", "r") as file:
        scenario_goals = json.load(file)

    # get env
    env = gymnasium.make(configuration.env_name)

    models = []
    for model_path in ensemble_model_paths:
        models.append(TQC.load(model_path, env=env,
                               custom_objects={"action_space": gymnasium.spaces.Box(-1.0, 1.0, shape=(7,),
                                                                                    dtype=np.float32)}))

    evaluation_results = {}
    for evaluation_scenario in evaluation_scenarios:  #
        # get env
        env = gymnasium.make(configuration.env_name,
                                  render=configuration.render,
                                  config=configuration,
                                  scenario=evaluation_scenario,
                                  ee_error_threshold=configuration.ee_error_thresholds[-1],
                                  speed_threshold=configuration.speed_thresholds[-1])

        print(f"Evaluating {evaluation_scenario}")

        # goals_to_achieve = copy(scenario_goals[evaluation_scenario])

        results, metrics = perform_benchmark(models, env, human=human, num_episodes=num_episodes, deterministic=True,
                                             strategy=strategy,
                                             scenario_name=evaluation_scenario,
                                             configuration=configuration)

        evaluation_results[evaluation_scenario] = {"results": results, "metrics": metrics}
        env.close()

    display_and_save_benchmark_results(agent_type, eval_type, evaluation_results, strategy)


def display_and_save_benchmark_results(agent_type, eval_type, evaluation_results, strategy):
    for key, value in evaluation_results.items():
        results = value["results"]
        print(f"{key}: {results}")
    results = {}
    for key, value in evaluation_results.items():
        results[key] = value["results"]
    if isinstance(strategy, dict):
        results['configuration'] = strategy
        strategy = strategy["name"]
    table = pd.DataFrame(results)
    table.index.name = "Criterias"
    print(f"{agent_type}-{strategy}")
    print(table.to_markdown())
    path = f"results/{eval_type}/{agent_type}"
    if not os.path.exists(path):
        os.makedirs(path)
    table.to_excel(f"{path}/{agent_type}-{strategy}.xlsx")


trained_models = {
    "mt_cl": ["gallant-serenity-299", "deep-frog-298", "solar-microwave-297", "revived-serenity-296",
              "glamorous-resonance-295"],
    # "mt_cl_opt_old": ["dandy-flower-328", "fancy-cloud-329", "rosy-music-336", "dauntless-leaf-337", "pious-cloud-346"],
    # "mt_cl_opt_her_old": ["lyric-dream-322", "apricot-cosmos-323", "tough-snow-324", "sunny-eon-325", "electric-serenity-326"],
    "mt_cl_opt_her": ["curious-wave-419", "glad-universe-418", "fiery-totem-417", "silver-star-416", "swept-salad-415"],
    "mt_cl_opt": ["dainty-bee-423", "vital-salad-424", "crimson-plant-425", "upbeat-gorge-427", "morning-sun-428"],
    "mt": ["solar-disco-133", "stellar-river-132", "snowy-pine-131", "comic-frost-130", "giddy-darkness-129"],
    "cl": ["firm-pond-79", "confused-firebrand-91", "rare-moon-92", "gallant-shape-95", "silver-hill-96"],
    "simple": ["dulcet-plant-105", "restful-bird-103", "graceful-dream-100", "noble-field-99", "easy-lion-98"],
    "good_ensemble": ["snowy-pine-131", "deep-frog-298", "glamorous-resonance-295", "firm-pond-79"],
    "bench": ["bench_narrow_tunnel", "bench_library2", "bench_workshop", "bench_wall"],
    "few_shot": ["few_shot_narrow_tunnel", "few_shot_library2", "few_shot_workshop", "few_shot_wall"]

    #  "opt2": ["atomic-snowflake-357"],
    #  "opt3": ["smooth-dream-362"],
    #  "mt_cl_effort": ["dutiful-fog-384", "dainty-surf-383", "helpful-night-382", "ruby-sponge-381", "dashing-dew-380"]
}

if __name__ == "__main__":
    eval_type = "base_eval"  # optimized; basic

    ensemble = "v6-large-model-test-rand-shape"

    evaluation_scenarios = ["library1", "workshop2", "library2", "narrow_tunnel", "workshop"]

    evaluate_ensemble(ensemble, human=True, eval_type="base_eval", strategy="weighted_aggregation",
                      num_episodes=200, evaluation_scenarios=evaluation_scenarios)

    # evaluate_prior(human=False, eval_type=eval_type)
    # evaluate_rl_agent(agents=trained_models["mt_cl"], human=False, eval_type=eval_type)
    # evaluate_rl_agent(agents=["easy-lion-98"], human=False, eval_type=eval_type)
    # evaluate_rl_agent(agents=["easy-lion-98"], human=False, eval_type=eval_type)

    # process_list = []
    # for key, value in trained_models.items():
    #     if key not in ["bench"]:
    #         continue
    #
    #     process_list.append(Process(target=evaluate_rl_agent, kwargs={"agents": value, "human": False, "eval_type": eval_type}))
    #
    # #process_list.append(Process(target=evaluate_prior, kwargs={"human": False, "eval_type": eval_type}))
    # #process_list.append(Process(target=evaluate_prior, kwargs={"human": False, "eval_type": "base_eval"}))
    # for p in process_list:
    #     p.start()
    #
    # for p in process_list:
    #     p.join()

    # evaluate ensemble

    # env = gym.make(config["env_name"], render=human, control_type=config["control_type"],
    #                obs_type=("ee",), goal_distance_threshold=config["goal_distance_threshold"],
    #                reward_type=config["reward_type"], limiter=config["limiter"],
    #                show_goal_space=False, scenario="library2",
    #                show_debug_labels=True, n_substeps=config["n_substeps"])
    # model2 = TQC.load(r"../run/run_data/wandb/earnest-feather-1/files/best_model.zip", env=env)
    # results1, metrics1 = evaluate_ensemble([model2], human=human, num_steps=10000, deterministic=True, strategy="variance_only", goals_to_achieve=metrics["goals"])
    # results2, metrics2 = evaluate(model1, human=human, goals_to_achieve=metrics["goals"], deterministic=True)
    # results3, metrics3 = evaluate(model2, human=human, goals_to_achieve=metrics["goals"], deterministic=True)

    # # Compare Models
    # model1 = TQC.load(r"../run/run_data/wandb/quiet-lion-122/files/model.zip", env=env)
    # model2 = TQC.load(r"../run/run_data/wandb/efficient-fog-124/files/model.zip", env=env)

    # results, metrics = evaluate(model1, model2, human=human, num_steps=50_000, deterministic=True)

    # print("Model 1:")
    # pprint.pprint(results)
    # # print("Model 2:")
    # # pprint.pprint(results1)
    # # print("library2-expert:")
    # # pprint.pprint(results3)
    #
    # # Some boilerplate to initialise things
    # sns.set()
    # plt.figure()
    #
    # # This is where the actual plot gets made
    # ax = sns.lineplot(data=metrics["end_effector_speeds"])
    #
    # # Customise some display properties
    # ax.set_title('End Effector Speeds')
    # ax.grid(color='#cccccc')
    # ax.set_ylabel('Speed')
    # ax.set_xlabel("TimeStep")
    # # ax.set_xticklabels(df["year"].unique().astype(str), rotation='vertical')
    #
    # # Ask Matplotlib to show it
    # plt.show()

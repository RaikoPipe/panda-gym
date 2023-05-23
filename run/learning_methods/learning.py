import sys
import gymnasium
sys.modules["gym"] = gymnasium


from typing import Optional, Union
import numpy as np
from stable_baselines3 import SAC, TD3, PPO, DDPG, DQN
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, EvalSuccessCallback, StopTrainingOnSuccessThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from run.learning_methods.imitation_learning import fill_replay_buffer_with_init_model, fill_replay_buffer_with_prior
from evaluate.evaluate import evaluate_ensemble
import pandas as pd

import panda_gym

import wandb
from wandb.integration.sb3 import WandbCallback


def get_env(config, n_envs, scenario, force_render=False):
    env = make_vec_env(config["env_name"], n_envs=n_envs,
                       env_kwargs={"render": config["render"] if force_render else False,
                                   "control_type": config["control_type"],
                                   "obs_type": config["obs_type"],
                                   "reward_type": config["reward_type"],
                                   "goal_distance_threshold": config["goal_distance_threshold"],
                                   "limiter": config["limiter"],
                                   "action_limiter": config["action_limiter"],
                                   "show_goal_space": False,
                                   "scenario": scenario,
                                   "show_debug_labels": False,
                                   "n_substeps": config["n_substeps"],
                                   "joint_obstacle_observation": config["joint_obstacle_observation"],
                                   "randomize_robot_pose": config["randomize_robot_pose"],
                                   "truncate_episode_on_collision" : config["truncate_episode_on_collision"],
                                   "collision_reward": config["collision_reward"]
                                   },
                       vec_env_cls=SubprocVecEnv if n_envs>1 else None
                       )
    # else:
    #     # todo: check if obsolete
    #     # env = gym.make(config["env_name"],
    #     #                render=config["render"] if not deactivate_render else False,
    #     #                #n_envs=config["n_envs"],
    #     #                control_type=config["control_type"],
    #     #                obs_type=config["obs_type"],
    #     #                goal_distance_threshold=config["goal_distance_threshold"],
    #     #                reward_type=config["reward_type"],
    #     #                limiter=config["limiter"],
    #     #                show_goal_space=False,
    #     #                scenario=stage,
    #     #                show_debug_labels=False,)
    #     env = make_vec_env(config["env_name"], n_envs=config["n_envs"],
    #                        env_kwargs={"render": config["render"] if not deactivate_render else False,
    #                                    "control_type": config["control_type"],
    #                                    "obs_type": config["obs_type"],
    #                                    "reward_type": config["reward_type"],
    #                                    "goal_distance_threshold": config["goal_distance_threshold"],
    #                                    "limiter": config["limiter"],
    #                                    "action_limiter": config["action_limiter"],
    #                                    "show_goal_space": False,
    #                                    "scenario": stage,
    #                                    "show_debug_labels": False,
    #                                    "n_substeps": config["n_substeps"]
    #                                    })

    return env


def get_model(algorithm, config, run):

    #env = get_env(config, config["stages"][0], deactivate_render=True)
    #n_actions = env.action_space.shape[0]
    # env.close()
    # env = get_env(config, config["stages"][0], deactivate_render=True)
    n_actions = 7 if config["control_type"] == "js" else 3
    # env.close()

    if config.get("noise_std"):
        normal_action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                                sigma=config["noise_std"] * np.ones(n_actions))
        vectorized_action_noise = VectorizedActionNoise(n_envs=config["n_envs"], base_noise=normal_action_noise)
        action_noise = vectorized_action_noise if config["n_env"] > 1 else normal_action_noise
    else:
        action_noise = None
    # todo: test train frequency
    if algorithm in ("TD3", "DDPG"):
        model = TD3(config["policy_type"], env=get_env(config, config["n_envs"], config["stages"][0]),
                    verbose=1, seed=config["seed"],
                    tensorboard_log=f"runs/{run.id}", device="cuda",
                    replay_buffer_class=config["replay_buffer_class"],
                    # hyperparameters
                    **config["hyperparams"]

                    )
    elif algorithm == "SAC":
        model = SAC(config["policy_type"], env=get_env(config, config["n_envs"], config["stages"][0]),
                    verbose=1, seed=config["seed"],
                    tensorboard_log=f"runs/{run.id}", device="cuda",
                    replay_buffer_class=config["replay_buffer_class"],
                    learning_starts=config["learning_starts"],

                    # hyperparameters

                    **config["hyperparams"]
                    )
    elif algorithm == "TQC":
        model = TQC(config["policy_type"], env=get_env(config,config["n_envs"], config["stages"][0]),
                    verbose=1,
                    tensorboard_log=f"runs/{run.id}", device="cuda",
                    replay_buffer_class= config["replay_buffer_class"],
                    learning_starts=config["learning_starts"],

                    # hyperparameters

                    **config["hyperparams"]

                    # learning_rate=config["learning_rate"],
                    # gamma=config["gamma"],
                    # tau=config["tau"],
                    # buffer_size=config["buffer_size"],
                    # gradient_steps=config["gradient_steps"],
                    # train_freq=config["train_freq"],
                    # use_sde=config["use_sde"],
                    # policy_kwargs=config["policy_kwargs"]
                    )

    return model


def get_tags(config):
    # set wandb tags
    tags = []
    tags.extend(str(x) for x in config["stages"])

    if len(config["stages"]) > 1:
        tags.append("curriculum_learning")
    if config["n_envs"] > 1:
        tags.append("multi_env")
    tags.append(config["algorithm"])
    tags.append(config["reward_type"])

    if config["prior_steps"] > 0:
        tags.append("prior")

    return tags


def init_wandb(config, tags):
    env_name = config["env_name"]
    project = f"{env_name}"
    run = wandb.init(
        project=f"{project}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        dir="run_data",
        tags=tags,
        group=config["stages"][-1]  # last stage is job goal
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    return run


def learn(config: dict, initial_model: Optional[OffPolicyAlgorithm] = None,
          starting_stage: Optional[str] = None, algorithm: str = "TD3"):
    panda_gym.register_envs(config["max_ep_steps"][0])

    tags = get_tags(config)
    if initial_model:
        tags.append("pre-trained")
    run = init_wandb(config, tags)

    if not initial_model:

        model= get_model(algorithm, config, run)

    else:
        model = initial_model
        stages: list = config["stages"]
        success_thresholds = config["success_thresholds"]
        if starting_stage:
            assert starting_stage in stages

            idx = stages.index(starting_stage)
            config["stages"] = stages[idx:]
            config["success_thresholds"] = success_thresholds[idx:]

    # model = TD3.load(r"run_data/wandb/run_panda_reach_evade_obstacle_stage_2_best_run/files/model.zip", env=env,
    #                  device="cuda", train_freq=n_envs, gradient_steps=2, replay_buffer=replay_buffer)

    assert len(config["stages"]) == len(config["success_thresholds"]) ==len(config["max_ep_steps"])

    # model.env.close()
    # learn for each stage until reward threshold is reached
    if config["learning_starts"]:
        if initial_model:
            model.replay_buffer = fill_replay_buffer_with_init_model(model, num_steps=config[
                "learning_starts"])
        elif config["prior_steps"] or len(config["stages"]) > 1:
            model.learn(total_timesteps=config["learning_starts"])
            model.learning_starts = 0

    if config["prior_steps"]:
        assert config["n_envs"] == 1
        env = get_env(config, 1, config["stages"][0])
        model.replay_buffer = fill_replay_buffer_with_prior(env, model, config["prior_steps"])

    for stage, success_threshold, max_ep_steps in zip(config["stages"], config["success_thresholds"], config["max_ep_steps"]):
        panda_gym.register_envs(max_ep_steps)
        if len(config["stages"]) > 1:
            model.set_env(get_env(config, config["n_envs"], stage))

        if config["render"]:
            # eval_env = gymnasium.make(config["env_name"], render=True if not config["render"] else False, control_type=config["control_type"],
            #                     obs_type=config["obs_type"],
            #                     reward_type=config["reward_type"],
            #                     show_goal_space=False, scenario=stage,
            #                     show_debug_labels=False, )
            eval_env = get_env(config, 1, scenario=stage, force_render=False)
        else:
            eval_env = get_env(config, config["n_envs"], scenario=stage)

        stop_train_callback = StopTrainingOnSuccessThreshold(success_threshold=success_threshold, verbose=1)

        eval_callback = EvalSuccessCallback(eval_env = eval_env, eval_freq=max(config["eval_freq"] // config["n_envs"], 1),
                                     callback_after_eval=stop_train_callback, verbose=1, n_eval_episodes=100,
                                     best_model_save_path=wandb.run.dir)


        model.learn(
            total_timesteps=config["max_timesteps"],
            callback=[WandbCallback(
                model_save_path=wandb.run.dir,
                model_save_freq=20_000
            ), eval_callback]
        )

        eval_env.close()

    # evaluate env
    evaluation_results = {}
    for evaluation_scenario in ["wang_3", "library2", "library1", "narrow_tunnel", "wall"]: # "wang_3", "library2", "library1", "narrow_tunnel"
        env = gymnasium.make(config["env_name"], render=False, control_type=config["control_type"],
                             obs_type=config["obs_type"], goal_distance_threshold=config["goal_distance_threshold"],
                             reward_type=config["reward_type"], limiter=config["limiter"],
                             show_goal_space=False, scenario=evaluation_scenario,
                             randomize_robot_pose=config["randomize_robot_pose"], joint_obstacle_observation=config["joint_obstacle_observation"],
                             truncate_episode_on_collision=config["truncate_episode_on_collision"],
                             show_debug_labels=True, n_substeps=config["n_substeps"])
        print(f"Evaluating {evaluation_scenario}")
        model.set_env(env)
        results, metrics = evaluate_ensemble([model], env, human=False, num_steps=10000, deterministic=True,
                                             strategy="variance_only")
        evaluation_results[evaluation_scenario] = {"results": results, "metrics" : metrics}
        env.close()

    results = {}
    for key, value in evaluation_results.items():
        results[key] = value["results"]

    table = pd.DataFrame(results)
    table.index.name = "Criterias"
    print(table.to_markdown())
    table["Criterias"] = list(results["library2"].keys())
    table = wandb.Table(dataframe=table)
    run.log({"results": table})
    for key, value in results.items():
        run.log({key: value["success_rate"]})

    run.finish()
    return model

import logging
import sys, os
from copy import deepcopy

import gymnasium

from classes.train_config import TrainConfig
from model_utils import load_model_utils

sys.modules["gym"] = gymnasium

from typing import Optional
import numpy as np
from stable_baselines3 import SAC, TD3, DDPG, PPO, HerReplayBuffer
from sb3_contrib import TQC
from sb3_extensions.replay_buffers import CustomHerReplayBuffer
from sb3_extensions.callbacks import RecordCustomMetricsCallback, StopTrainingOnSuccessThreshold, EvalSuccessCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
from training.learning_methods.imitation_learning import fill_replay_buffer_with_init_model, \
    fill_replay_buffer_with_prior
from evaluation.evaluate import perform_benchmark
import pandas as pd
from gymnasium.envs.registration import register
import wandb
from wandb.integration.sb3 import WandbCallback
import panda_gym

from dataclasses import asdict


def get_env(config, scenario, ee_error_threshold, speed_threshold, force_render=False):
    config.render = config.render if not force_render else False
    args = {
        'config': config,
        "ee_error_threshold": ee_error_threshold,
        "speed_threshold": speed_threshold,
        "scenario": scenario,
    }
    if config.n_envs > 1:
        return make_vec_env(config.env_name, n_envs=config.n_envs, seed=config.seed,
                            env_kwargs=args,
                            vec_env_cls=SubprocVecEnv if config.n_envs > 1 else None
                            )
    else:
        return gymnasium.make(config.env_name,
                              render=force_render,
                              config=config,
                              scenario=scenario,
                              ee_error_threshold=ee_error_threshold,
                              speed_threshold=speed_threshold)
    # else:
    #     # todo: check if obsolete
    #     # env = gym.make(config.env_name,
    #     #                render=config.render if not deactivate_render else False,
    #     #                #n_envs=config.n_envs,
    #     #                control_type=config.control_type,
    #     #                obs_type=config.obs_type,
    #     #                goal_distance_threshold=config.goal_distance_threshold,
    #     #                reward_type=config.reward_type,
    #     #                limiter=config.limiter,
    #     #                show_goal_space=False,
    #     #                scenario=stage,
    #     #                show_debug_labels=False,)
    #     env = make_vec_env(config.env_name, n_envs=config.n_envs,
    #                        env_kwargs={"render": config.render if not deactivate_render else False,
    #                                    "control_type": config.control_type,
    #                                    "obs_type": config.obs_type,
    #                                    "reward_type": config.reward_type,
    #                                    "goal_distance_threshold": config.goal_distance_threshold,
    #                                    "limiter": config.limiter,
    #                                    "action_limiter": config.action_limiter,
    #                                    "show_goal_space": False,
    #                                    "scenario": stage,
    #                                    "show_debug_labels": False,
    #                                    "n_substeps": config.n_substeps
    #                                    })


def get_model(config, run):
    # env = get_env(config, config.stages[0], deactivate_render=True)
    # n_actions = env.action_space.shape[0]
    # env.close()
    # env = get_env(config, config.stages[0], deactivate_render=True)
    n_actions = 7 if config.control_type == "js" else 3
    # env.close()

    if hasattr(config.hyperparams, 'noise_std'):
        noise_std = config.hyperparams.noise_std
        normal_action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                                sigma=noise_std * np.ones(n_actions))
        vectorized_action_noise = VectorizedActionNoise(n_envs=config.n_envs, base_noise=normal_action_noise)
        action_noise = vectorized_action_noise if config.n_envs > 1 else normal_action_noise
    else:
        action_noise = None

    algorithm_type = None
    match config.algorithm:
        case "DDPG":
            algorithm_type = DDPG
        case "TD3":
            algorithm_type = TD3
        case "SAC":
            algorithm_type = SAC
        case "TQC":
            algorithm_type = TQC

    if algorithm_type is None:
        logging.warning("Algorithm not found. Aborting")
        raise Exception("Algorithm not found!")

    model = algorithm_type(
        config.policy_type,
        env=get_env(config, config.stages[0], config.ee_error_thresholds[0], config.speed_thresholds[0]),
        verbose=1, seed=config.seed,
        tensorboard_log=f"runs/{run.id}", device="cuda",
        replay_buffer_class=config.replay_buffer_class,
        learning_starts=config.learning_starts,
        action_noise=action_noise,
        # hyperparameters
        **config.hyperparams.__dict__
    )

    return model


def get_tags(config):
    # set wandb tags
    tags = []
    tags.extend(str(x) for x in config.stages)

    if len(config.stages) > 1:
        tags.append("curriculum_learning")
    if config.n_envs > 1:
        tags.append("multi_env")
    tags.append(config.algorithm)
    tags.append(config.reward_type)

    if config.prior_steps > 0:
        tags.append("prior")

    return tags


def init_wandb(config, tags):
    wandb_config = asdict(config)
    wandb_config["hyperparams"] = config.hyperparams.__dict__

    env_name = config.env_name
    project = f"{env_name}"

    run_dir = fr"/beegfs2/scratch/rreider/run_data/{config.group}"

    # create run directory if not exists
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    run = wandb.init(
        project=f"{project}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        dir=run_dir,
        tags=tags,
        group=config.group,
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        job_type=config.job_type,
        name=config.name
    )
    return run


def switch_model_env(model, env) -> None:
    model.set_env(env, force_reset=True)
    if isinstance(model.replay_buffer, (HerReplayBuffer, CustomHerReplayBuffer)):
        model.replay_buffer.set_env(env)


def learn(config: TrainConfig, initial_model: Optional[OffPolicyAlgorithm] = None,
          starting_stage: Optional[str] = None):
    panda_gym.register_reach_ao(config.max_ep_steps[0])

    tags = get_tags(config)
    if initial_model:
        tags.append("pre-trained")
    run = init_wandb(config, tags)

    if not initial_model:

        model = get_model(config, run)

    else:
        model = initial_model
        stages: list = config.stages
        success_thresholds = config.success_thresholds
        if starting_stage:
            assert starting_stage in stages

            idx = stages.index(starting_stage)
            config.stages = stages[idx:]
            config.success_thresholds = success_thresholds[idx:]

    # learn for each stage until reward threshold is reached
    if config.learning_starts:
        if initial_model:
            pass
            # model.replay_buffer = fill_replay_buffer_with_init_model(model,
            #                                                          env=get_env(config,
            #                                                                      1,
            #                                                                      scenario=config.stages[0]),
            #                                                          num_steps=config.learning_starts)
        elif config.prior_steps or len(config.stages) > 1:
            model.learn(total_timesteps=config.learning_starts)
            model.learning_starts = 0

    if config.prior_steps:
        assert config.n_envs == 1
        env = get_env(config, 1, config.stages[0])
        model.replay_buffer = fill_replay_buffer_with_prior(env, model, config.prior_steps)

    iteration = 0
    model = train_model(config, iteration, model, run)

    # evaluate trained model
    # benchmark_model(config, model, run)

    return model, run


def train_model(config, iteration, model, run):
    assert len(config.stages) == len(config.success_thresholds) == len(config.max_ep_steps) == len(
        config.ee_error_thresholds)

    speed_thresholds = [None for _ in range(len(config.stages))]

    if config.goal_condition == "halt":
        assert len(config.stages) == len(config.speed_thresholds)
        speed_thresholds = config.speed_thresholds

    train_sequence = zip(config.stages,
                         config.success_thresholds,
                         config.max_ep_steps,
                         config.ee_error_thresholds,
                         speed_thresholds)

    for stage, success_threshold, max_ep_steps, ee_error_threshold, speed_threshold in train_sequence:
        panda_gym.register_reach_ao(max_ep_steps)

        switch_model_env(model, get_env(config, stage, ee_error_threshold, speed_threshold))

        callbacks = []

        # get eval callbacks for benchmark environments
        eval_benchmark_scenes = [
            "library1",
            "library2",
            "narrow_tunnel",
            "workshop",
            "workshop2",
        ]

        eval_benchmark_config = deepcopy(config)
        eval_benchmark_config.eval_freq = 10_000
        eval_benchmark_config.n_eval_episodes = 200
        eval_benchmark_config.n_envs = 8
        eval_benchmark_envs = [
            get_env(eval_benchmark_config, scene, config.ee_error_thresholds[-1], config.speed_thresholds[-1])
            for scene in eval_benchmark_scenes]

        eval_training_env = None
        if stage == config.stages[-1]: # final stage
            for eval_benchmark_scene, eval_benchmark_env in zip(eval_benchmark_scenes, eval_benchmark_envs):
                callbacks.append(
                    get_eval_success_callbacks(eval_benchmark_config, eval_benchmark_env,
                                               best_model_save_path=f"{run.dir}/{eval_benchmark_scene}",
                                               eval_log_name=f"{eval_benchmark_scene}_eval"))
        else:
            stop_train_callback = StopTrainingOnSuccessThreshold(success_threshold=success_threshold, verbose=1)
            eval_training_env = get_env(config, stage, ee_error_threshold, speed_threshold)
            callbacks.append(get_eval_success_callbacks(config, eval_training_env, stop_train_callback,
                                                        best_model_save_path=f"{run.dir}/training_scene"))

        callbacks.append(RecordCustomMetricsCallback({'stage': stage}))
        callbacks.append(WandbCallback(
            model_save_path=run.dir,
            model_save_freq=20_000
        ))

        model.learn(
            total_timesteps=config.max_timesteps,
            callback=callbacks,
            log_interval=4
        )

        # save model
        model.save(f"{run.dir}/model_{stage}_{iteration}")

        # close eval environments
        if eval_training_env:
            eval_training_env.close()
        for eval_benchmark_env in eval_benchmark_envs:
            eval_benchmark_env.close()

    return model


def get_eval_success_callbacks(config, eval_env, stop_train_callback=None, best_model_save_path=None,
                               eval_log_name=None):
    return EvalSuccessCallback(eval_env=eval_env,
                               eval_freq=max(config.eval_freq // config.n_envs, 1),
                               callback_after_eval=stop_train_callback,
                               verbose=1,
                               n_eval_episodes=config.n_eval_episodes,
                               best_model_save_path=best_model_save_path,
                               eval_log_name=eval_log_name)


def get_eval_env(config, stage, ee_error_threshold=None, speed_threshold=None):
    if config.render:
        # eval_env = gymnasium.make(config.env_name, render=True if not config.render else False, control_type=config.control_type,
        #                     obs_type=config.obs_type,
        #                     reward_type=config.reward_type,
        #                     show_goal_space=False, scenario=stage,
        #                     show_debug_labels=False, )
        eval_env = get_env(config, scenario=stage, force_render=True)
    else:
        eval_env = get_env(config, stage, ee_error_threshold, speed_threshold)
    return eval_env


def benchmark_model(config, model, run):
    evaluation_results = {}
    for evaluation_scenario in [
        "reachao1",
        "reachao2",
        "reachao3",
        "wangexp-3",
        "reachao_rand",
        "reachao_rand_start",
        "library1",
        "library2",
        "narrow_tunnel",
        "tunnel",
        "workshop",
        "industrial",
        "wall",
    ]:
        # workaround because fetching results with multiple envs is not supported (yet)
        config = deepcopy(config)
        config.n_envs = 1

        # deactivate safety distance
        config.safety_distance = 0.0

        # register max ep steps 300
        panda_gym.register_reach_ao(300)

        env = get_env(config, evaluation_scenario, config.ee_error_thresholds[-1], config.speed_thresholds[-1])
        print(f"Evaluating {evaluation_scenario}")
        best_model = model.load(path=fr"{wandb.run.dir}/model_reachao3_0.zip", env=env)

        results, metrics = perform_benchmark([best_model], env, human=False, num_episodes=100, deterministic=True,
                                             strategy="variance_only")
        evaluation_results[evaluation_scenario] = {"results": results, "metrics": metrics}
        env.close()
    results = {}
    for key, value in evaluation_results.items():
        results[key] = value["results"]
    table = pd.DataFrame(results)
    table.index.name = "Criterias"
    print(table.to_markdown())
    table["Criterias"] = results["reachao1"].keys()
    table = wandb.Table(dataframe=table)

    run.log({"results": table})
    for key, value in results.items():
        run.log({key: value["success_rate"]})


def continue_learning(model_group_name, run=None, name=None, config=None):
    # load model
    model_paths = load_model_utils.get_group_model_paths(model_group_name)
    model_yaml_paths = load_model_utils.get_group_yaml_paths(model_group_name)

    # find model with name
    model_path = None
    model_yaml_path = None
    if name is None:
        model_path = model_paths[0]
        model_yaml_path = model_yaml_paths[0]
    else:
        for path, yaml_path in zip(model_paths, model_yaml_paths):
            if name in path:
                model_path = path
                model_yaml_path = yaml_path
                break

    if config is None:
        config = load_model_utils.get_train_config_from_yaml(model_yaml_path)

    model = TQC.load(model_path, env=get_env(config, config.stages[0], config.ee_error_thresholds[0],
                                             config.speed_thresholds[0]),
                     )

    if run is None:
        tags = get_tags(config)
        tags.append("from-pre-trained")
        run = init_wandb(config, tags)

    # fill replay buffer to prevent performing training step with empty buffer
    if config.learning_starts:
        model.learn(total_timesteps=2000)

    model = train_model(config, 0, model, run)

    # evaluate trained model
    benchmark_model(config, model, run)

    return model, run

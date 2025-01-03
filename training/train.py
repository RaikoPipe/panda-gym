import os
import sys
import time

import gymnasium
import torch.multiprocessing as mp

sys.modules["gym"] = gymnasium

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import panda_gym

from classes.train_config import TrainConfig
from classes.hyperparameters import Hyperparameters

import torch

import argparse

# CUDA and PyTorch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
torch.backends.cudnn.deterministic = False  # Better performance
torch.set_float32_matmul_precision('high')

# Enable multi-GPU if available
if torch.cuda.device_count() > 1:
    torch.cuda.set_device(0)  # Use primary GPU

configuration = TrainConfig()

# # register envs to gymnasium
panda_gym.register_reach_ao(configuration.max_ep_steps[0])

# process command line arguments
parser = argparse.ArgumentParser(description='Model training')
parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds for training")

args = parser.parse_args()


def optimize_env(env):
    """Apply environment-specific optimizations"""
    if hasattr(env, 'num_envs'):
        # Vectorized environment optimizations
        env.num_envs = int(os.environ.get('SLURM_CPUS_PER_TASK', '32'))
    return env


def train_model(configs, seeds=None, pretrained_model_name=None):
    for config in configs:
        mp.set_start_method('spawn', force=True)  # Better CUDA compatibility

        if seeds is None:
            seeds = [0]

        for seed in seeds:
            config.name = f'{config.name}_{seed}'

            config.seed = seed

            model, run = learn(config=config, pretrained_model_name=pretrained_model_name)

            # Memory cleanup
            model.env.close()

            del model


if __name__ == "__main__":
    # start timer
    start_time = time.time()
    from training.utils.setup_training import learn
    import wandb

    # Initialize CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    #wandb.login(key=os.getenv("wandb_key"))

    hyperparams = Hyperparameters(algorithm="TQC")
    #hyperparams.policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300])

    train_config = TrainConfig(
        group="Benchmark-Eval",
        job_type="train",
        name="400-300-reset-test",
        #stages=["reachao3"],
        #success_thresholds=[1.0],
        #ee_error_thresholds=[0.05],
        #max_ep_steps=[100],
        max_timesteps=1_000_000,
        n_envs=8,  # Parallel environments
        n_eval_envs=32,
        n_benchmark_eval_episodes=100,
        n_eval_episodes=100,
        eval_freq=20_000,
        benchmark_eval_freq=50_000,
        algorithm="TQC",
        learning_starts=10000
    )

    train_config.hyperparams = hyperparams

    train_model(seeds=args.seeds, configs=[train_config])

    # Configure buffer size based on available memory
    # total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    # buffer_size = min(int(1e6), int((total_memory_gb * 0.6) * 1e5))  # Use 60% of GPU memory
    # train_config.hyperparams.buffer_size = buffer_size

    # end timer
    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds")

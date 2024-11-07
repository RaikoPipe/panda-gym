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


def optimize_env(env):
    """Apply environment-specific optimizations"""
    if hasattr(env, 'num_envs'):
        # Vectorized environment optimizations
        env.num_envs = int(os.environ.get('SLURM_CPUS_PER_TASK', '32'))
    return env


def train_base_model(config=TrainConfig(), iterations=None):
    mp.set_start_method('spawn', force=True)  # Better CUDA compatibility

    for seed in range(0, iterations):
        if config.name == 'default':
            name = f"base_model_{time.strftime('%Y%m%d_%H%M%S')}"
            config.name = f'{name}_{seed}'

        config.seed = seed

        model, run = learn(config=config)

        # Memory cleanup
        if hasattr(model, 'env'):
            model.env.close()
        torch.cuda.empty_cache()
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

    wandb.login(key=os.getenv("wandb_key"))

    hyperparams = Hyperparameters(algorithm="TQC")
    hyperparams.policy_kwargs = dict(log_std_init=-3, net_arch=[128, 128])

    train_config = TrainConfig(
        group="benchmark-eval-128-128",
        job_type="train",
        name="128-128",
        max_timesteps=1000,
        n_envs=32,  # Parallel environments
    )

    # Configure buffer size based on available memory
    # total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    # buffer_size = min(int(1e6), int((total_memory_gb * 0.6) * 1e5))  # Use 60% of GPU memory
    # train_config.hyperparams.buffer_size = buffer_size

    for configuration in [train_config]:
        train_base_model(config=configuration, iterations=1)

    # end timer
    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds")



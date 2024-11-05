import os
import sys
import time

import gymnasium
import torch.multiprocessing as mp

sys.modules["gym"] = gymnasium

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os

from classes.train_config import TrainConfig
from classes.hyperparameters import Hyperparameters

import torch

# CUDA and PyTorch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
torch.backends.cudnn.deterministic = False  # Better performance
torch.set_float32_matmul_precision('high')
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '32')))

# Enable multi-GPU if available
if torch.cuda.device_count() > 1:
    torch.cuda.set_device(0)  # Use primary GPU


def optimize_env(env):
    """Apply environment-specific optimizations"""
    if hasattr(env, 'num_envs'):
        # Vectorized environment optimizations
        env.num_envs = int(os.environ.get('SLURM_CPUS_PER_TASK', '32'))
    return env


def get_optimized_hyperparams():
    """Return optimized hyperparameters for TQC"""
    hyperparams = Hyperparameters(algorithm="TQC_v2")
    hyperparams.policy_kwargs = dict(
        log_std_init=-3,
        net_arch=[128, 128],
        optimizer_class=torch.optim.AdamW,  # Better optimizer
        optimizer_kwargs=dict(
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            maximize=False,
        )
    )
    return hyperparams


def train_base_model(config=TrainConfig(), iterations=None):
    mp.set_start_method('spawn', force=True)  # Better CUDA compatibility

    for seed in range(0, iterations):
        if config.name == 'default':
            name = f"base_model_{time.strftime('%Y%m%d_%H%M%S')}"
            config.name = f'{name}_{seed}'

        config.seed = seed

        # Set environment variables for better performance
        os.environ['MKL_NUM_THREADS'] = os.environ.get('SLURM_CPUS_PER_TASK', '32')
        os.environ['OMP_NUM_THREADS'] = os.environ.get('SLURM_CPUS_PER_TASK', '32')

        model, run = learn(config=config)

        # Memory cleanup
        if hasattr(model, 'env'):
            model.env.close()
        torch.cuda.empty_cache()
        del model


if __name__ == "__main__":
    from training.utils.setup_training import learn
    import wandb

    # Initialize CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    wandb.login(key=os.getenv("wandb_key"))

    hyperparams = get_optimized_hyperparams()
    train_config = TrainConfig(
        group="benchmark-eval-128-128",
        job_type="train",
        name="128-128",
        max_timesteps=2_000_000,
        hyperparams=hyperparams,
        n_envs=int(os.environ.get('SLURM_CPUS_PER_TASK', '32')),  # Parallel environments
    )

    # Configure buffer size based on available memory
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    buffer_size = min(int(1e6), int((total_memory_gb * 0.6) * 1e5))  # Use 60% of GPU memory
    train_config.hyperparams.buffer_size = buffer_size

    for configuration in [train_config]:
        train_base_model(config=configuration, iterations=5)

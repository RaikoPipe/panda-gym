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

from multiprocessing import set_start_method

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
parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
parser.add_argument("--n_eval_envs", type=int, default=4, help="Number of parallel evaluation environments")
parser.add_argument("--pretrained_model", type=str, default=None, help="Pretrained model name")
parser.add_argument("--algorithm", type=str, default="TQC", help="Algorithm to train")
parser.add_argument("--group", type=str, default="default", help="Group name for wandb")
parser.add_argument("--name", type=str, default="default", help="Name for run")
parser.add_argument("--max_timesteps", type=int, default=1_000_000, help="Maximum number of timesteps")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")

args = parser.parse_args()


def optimize_env(env):
    """Apply environment-specific optimizations"""
    if hasattr(env, 'num_envs'):
        # Vectorized environment optimizations
        env.num_envs = int(os.environ.get('SLURM_CPUS_PER_TASK', '32'))
    return env

def train_model_parallel(configs, seeds=None, pretrained_model_name=None):
    mp.set_start_method('spawn')

    processes = []
    for config in configs:
        config_name = config.name
        if seeds is None:
            seeds = [0]

        for seed in seeds:
            config.name = f'{config_name}_{seed}'

            config.seed = seed

            process = mp.Process(target=learn, args=(config, pretrained_model_name))
            process.start()
            processes.append(process)

    for process in processes:
        process.join()


def train_model(configs, seeds=None, pretrained_model_name=None):
    for config in configs:

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

    wandb.login(key=os.getenv("wandb_key"))

    algorithm = args.algorithm

    hyperparams = Hyperparameters(algorithm=algorithm)
    hyperparams.batch_size = args.batch_size
    hyperparams.policy_kwargs = dict(log_std_init=-3, net_arch=dict(pi=[400,300], qf=[1024, 1024]))
    hyperparams.gradient_steps = 20
    #hyperparams.buffer_size = 300_000

    train_config = TrainConfig(
        group=args.group,
        job_type="train",
        name=f"{args.name}-{algorithm}",
        max_timesteps=args.max_timesteps,
        n_envs=args.n_envs,  # Parallel environments
        n_eval_envs=args.n_eval_envs,
        n_benchmark_eval_episodes=100,
        n_eval_episodes=100,
        eval_freq=20_000,
        benchmark_eval_freq=50_000,
        algorithm=algorithm,
        learning_starts=10000,
        # advanced curriculum
        #stages = ["reachao1", "reachao2", "reachao3", "exp-10"],
        #success_thresholds = [0.9,0.9,0.9,1.0],
        #max_ep_steps = [50,75,100,200],
        #ee_error_thresholds=[0.05, 0.05, 0.05, 0.05],
    )

    train_config.hyperparams = hyperparams

    if args.pretrained_model:
        train_config.stages = ["reachao3"]
        train_config.success_thresholds = [1.0]
        train_config.ee_error_thresholds = [0.05]
        train_config.max_ep_steps = [200]

    train_model(seeds=args.seeds, configs=[train_config], pretrained_model_name=args.pretrained_model)

    # Configure buffer size based on available memory
    # total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    # buffer_size = min(int(1e6), int((total_memory_gb * 0.6) * 1e5))  # Use 60% of GPU memory
    # train_config.hyperparams.buffer_size = buffer_size

    # end timer
    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds")

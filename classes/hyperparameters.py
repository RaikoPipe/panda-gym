from dataclasses import dataclass
from torch import nn
import torch


# hyperparameters are from rl-baselines3 zoo and https://arxiv.org/pdf/2106.13687.pdf

class Hyperparameters:
    def __init__(self, algorithm: str = "TQC"):

        if algorithm == "TQC":
            self.learning_rate = 7.3e-4
            self.gamma = 0.98
            self.tau = 0.02
            self.buffer_size = 1_000_000
            self.batch_size = 256
            self.gradient_steps = 8
            self.train_freq = 8
            self.ent_coef = "auto"
            self.use_sde = True
            #self.top_quantiles_to_drop_per_net = 5
            self.policy_kwargs = dict(log_std_init=-3, net_arch=[256, 256])
        elif algorithm == "TQC-exp": # mix of tuned TQC fetchPush some research findings
            self.learning_rate = 1e-3
            self.gamma = 0.95
            self.tau = 0.05
            self.buffer_size = 1_000_000
            self.batch_size = 2048
            self.gradient_steps = 8
            self.train_freq = 8
            self.ent_coef = "auto"
            self.policy_kwargs = dict(log_std_init=-3, net_arch=dict(pi =[256, 256], qf= [1024, 1024]), n_critics=2,
                                      optimizer_class=torch.optim.AdamW) # Weight Decay Normalization
            # according to some research wider critic networks are easier to optimize
        elif algorithm == "CrossQ":
            # taken from SAC
            self.learning_rate = 7.3e-4
            self.buffer_size = 300_000
            self.batch_size = 256
            self.ent_coef = "auto"
            self.gamma = 0.98
            self.policy_delay = 3
            self.train_freq = 8
            self.gradient_steps = 8
            self.policy_kwargs = dict(use_expln=True, log_std_init=-3, net_arch=dict(pi=[400, 300], qf=[2048, 2048]))

        elif algorithm == "TD3":
            self.learning_rate = 1e-3
            self.gamma = 0.98
            self.buffer_size = 200_000
            self.gradient_steps = -1
            self.train_freq = (1, "episode")
            self.policy_kwargs = dict(net_arch=[256, 256])
        elif algorithm == "PPO":
            self.normalize = True
            self.n_envs = 16
            self.policy = 'MlpPolicy'
            self.batch_size = 128
            self.n_steps = 512
            self.gamma = 0.99
            self.gae_lambda = 0.9
            self.n_epochs = 20
            self.ent_coef = 0.0
            self.sde_sample_freq = 4
            self.max_grad_norm = 0.5
            self.vf_coef = 0.5
            self.learning_rate = 3e-5
            self.use_sde = True
            self.clip_range = 0.4
            self.policy_kwargs = dict(log_std_init=-2,
                                 ortho_init=False,
                                 activation_fn=nn.ReLU,
                                 net_arch=dict(pi=[256, 256], vf=[256, 256])
                                 )
        elif algorithm == "DDPG":
            self.learning_rate = 1e-3
            self.gamma = 0.98
            self.buffer_size = 200_000
            self.gradient_steps = 1
            self.train_freq = 1
            self.noise_std = 0.1
            self.policy_kwargs = dict(net_arch=[256, 256])
        else:
            raise ValueError("Invalid algorithm")

    # learning_rate: float
    # gamma: float
    # gradient_steps: int
    # train_freq: int

    # off policy
    # buffer_size: int

    # net architecture
    # policy_kwargs: dict

    # ppo specific
    # n_envs: int
    # batch_size: int
    # n_steps: int
    # gae_lambda: float
    # n_epochs: int
    # sde_sample_freq: int
    # max_grad_norm: float
    # vf_coef: float
    # clip_range: float
    # normalize: bool
    # max_grad_norm: float

    # generalized state dependend exploration
    # use_sde: bool

    # deterministic policy
    # noise_std: float

    # tqc specific
    # tau: float

    # stochastic policy
    # ent_coef: float

    # set default values
    algorithm: str = "TQC"

# TQC PandaReach-v1
# buffer_size: 1_000_000
# gamma: 0.95
# learning_rage: 0.001
# normalize: True

# HER Defaults
# buffer_size: 1_000_000
# gamma: 0.95
# learning_rate: 1e6
# tau: 0.05

# Policy kwargs decreasing neurons with each layer
# net_arch: [128, 64, 32]
# net_arch: [256, 128, 64]

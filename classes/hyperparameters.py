from dataclasses import dataclass
from torch import nn

# hyperparameters are from rl-baselines3 zoo and https://arxiv.org/pdf/2106.13687.pdf

@dataclass
class Hyperparameters:

    #learning_rate: float
    #gamma: float
    #gradient_steps: int
    #train_freq: int

    # off policy
    #buffer_size: int

    # net architecture
    #policy_kwargs: dict

    # ppo specific
    #n_envs: int
    #batch_size: int
    #n_steps: int
    #gae_lambda: float
    #n_epochs: int
    #sde_sample_freq: int
    #max_grad_norm: float
    #vf_coef: float
    #clip_range: float
    #normalize: bool
    #max_grad_norm: float

    # generalized state dependend exploration
    #use_sde: bool

    # deterministic policy
    #noise_std: float

    # tqc specific
    #tau: float

    # stochastic policy
    #ent_coef: float

    # set default values
    algorithm: str = "TQC"

    if algorithm == "TQC":
        learning_rate = 0.0007
        gamma = 0.98
        tau = 0.02
        buffer_size = 300_000
        batch_size = 256
        gradient_steps = 8
        train_freq = 8
        ent_coef = "auto"
        use_sde = True
        policy_kwargs = dict(log_std_init=-3, net_arch=[256, 256])
    elif algorithm == "TD3":
        learning_rate = 1e-3
        gamma = 0.98
        buffer_size = 200_000
        gradient_steps = -1
        train_freq = (1, "episode")
        policy_kwargs = dict(net_arch=[256, 256])
    elif algorithm == "PPO":
        normalize = True
        n_envs = 16
        policy = 'MlpPolicy'
        batch_size = 128
        n_steps = 512
        gamma = 0.99
        gae_lambda = 0.9
        n_epochs = 20
        ent_coef = 0.0
        sde_sample_freq = 4
        max_grad_norm = 0.5
        vf_coef = 0.5
        learning_rate = 3e-5
        use_sde = True
        clip_range = 0.4
        policy_kwargs = dict(log_std_init=-2,
                             ortho_init=False,
                             activation_fn=nn.ReLU,
                             net_arch=dict(pi=[256, 256], vf=[256, 256])
                             )
    elif algorithm == "DDPG":
        learning_rate = 1e-3
        gamma = 0.98
        buffer_size = 200_000
        gradient_steps = 1
        train_freq = 1
        noise_std = 0.1
        policy_kwargs = dict(net_arch=[256, 256])
    else:
        raise ValueError("Invalid algorithm")



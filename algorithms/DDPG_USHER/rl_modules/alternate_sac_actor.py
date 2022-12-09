import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal





LOG_STD_MAX = 2
LOG_STD_MIN = -20

clip_max = 3

class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, env_params['action'])
        # self.mu_layer.weight.data.fill_(0)
        # self.mu_layer.bias.data.fill_(0)
        self.log_std_layer = nn.Linear(256, env_params['action'])
        # self.log_std_layer.weight.data.fill_(0)
        # self.log_std_layer.bias.data.fill_(-1.)

    def forward(self, x, with_logprob = False, deterministic = False, forced_exploration=1):
        # with_logprob = False
        x = torch.clamp(x, -clip_max, clip_max)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        net_out = F.relu(self.fc3(x))


        mu = self.mu_layer(net_out)#/100
        log_std = self.log_std_layer(net_out)-1#/100 -1.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)*forced_exploration

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.max_action * pi_action

        if with_logprob: 
            return pi_action, logp_pi
        else: 
            return pi_action
        # return actions

class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, env_params['action'])
        # self.mu_layer.weight.data.fill_(0)
        # self.mu_layer.bias.data.fill_(0)
        self.log_std_layer = nn.Linear(256, env_params['action'])

    def forward(self, x, with_logprob = False, deterministic = False, forced_exploration=1):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # return actions
        mu = self.mu_layer(x)#/100
        log_std = self.log_std_layer(x)-1#/100 -1.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)*forced_exploration

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.max_action * pi_action

        if with_logprob: 
            return pi_action, logp_pi
        else: 
            return pi_action

            
    def set_normalizers(self, o, g): 
        self.o_norm = o
        self.g_norm = g

    # def normed_forward(self, obs, g, deterministic=False): 
    #     obs_norm = self.o_norm.normalize(obs)
    #     g_norm = self.g_norm.normalize(g)
    #     # concatenate the stuffs
    #     inputs = np.concatenate([obs_norm, g_norm])
    #     inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    #     return self.forward(inputs, deterministic=deterministic)

    def _get_norms(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        return obs_norm, g_norm

    def _get_denorms(self, obs, g):
        obs_denorm = self.o_norm.denormalize(obs)
        g_denorm = self.g_norm.denormalize(g)
        return obs_denorm, g_denorm

    def normed_forward(self, obs, g, deterministic=False): 
        obs_norm, g_norm = self._get_norms(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        # concatenate the stuffs
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)
        return self.forward(inputs, deterministic=deterministic, forced_exploration=1)


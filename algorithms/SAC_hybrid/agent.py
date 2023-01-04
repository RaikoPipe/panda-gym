import itertools

from algorithms.SAC_hybrid import core
from algorithms.SAC_hybrid.replay_buffer import ReplayBuffer
import torch
from copy import deepcopy, copy
from torch.optim import Adam
import numpy as np
import math
from torch.distributions import Normal


def inverse_sigmoid_gating_function(k, C, x):
    val = 1 / (1 + math.exp(k * (x - C)))
    return val


def compute_kld_univariate_gaussians(mu_prior, sigma_prior, mu_policy, sigma_policy):
    # Computes the analytical KL divergence between two univariate gaussians
    kl = torch.log(sigma_policy / sigma_prior) + (
                (sigma_prior ** 2 + (mu_prior - mu_policy) ** 2) / (2 * sigma_policy ** 2)) - 1 / 2
    return kl

def fuse_ensembles_deterministic(ensemble_actions):
    actions = torch.tensor([ensemble_actions[0][0]])
    mu = torch.mean(actions, dim=0)
    var = torch.var(actions, dim=0)
    sigma = np.sqrt(var)
    return mu, sigma

def fuse_ensembles_stochastic(ensemble_actions):
    mu = (np.sum(np.array([ensemble_actions[0][0]]), axis=0))
    var = (np.sum(
        np.array([(ensemble_actions[0][1] ** 2 + ensemble_actions[0][0] ** 2) - mu ** 2 ]),
        axis=0))
    sigma = np.sqrt(var)
    return torch.from_numpy(mu), torch.from_numpy(sigma)

def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma):
    # The policy mu and sigma are from the stochastic SAC_hybrid output
    # The sigma from prior is fixed
    mu = (np.power(policy_sigma, 2) * prior_mu + np.power(prior_sigma, 2) * policy_mu) / (
                np.power(policy_sigma, 2) + np.power(prior_sigma, 2))
    sigma = np.sqrt(
        (np.power(prior_sigma, 2) * np.power(policy_sigma, 2)) / (np.power(policy_sigma, 2) + np.power(prior_sigma, 2)))
    return mu, sigma


#@ray.remote(num_gpus=1)



# see SAC_hybrid example: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
class SAC_Agent():
    def __init__(self, env_fn, prior, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),
                 steps_per_epoch=4000, gamma=0.99, epochs=100, max_ep_steps=200,
                 polyak=0.995, lr=1e-3, alpha=0.2, beta=0.3, batch_size=100, start_steps=10000,
                 update_after=1000, update_every=50, num_eval_episodes=10,
                use_kl_loss=False, epsilon=1e-5, target_KL_div=0, factor_c = 0.3, lambda_max = 15.0,
                 target_entropy=0.3, sigma_prior=0.4, save_freq = 1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), method="residual",
                 ):

        """
        Soft Actor-Critic (SAC_hybrid)
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                ``act``, ``q1``, and ``q2`` should return:
                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                               | of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                               | estimate of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ===========  ================  ======================================
                Calling ``pi`` should return:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                               | actions in ``a``. Importantly: gradients
                                               | should be able to flow back into ``a``.
                ===========  ================  ======================================
            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to SAC_hybrid.
            seed (int): Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs to run and train agent.
            replay_size (int): Maximum length of replay buffer.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:
                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)
            lr (float): Learning rate (used for both policy and value learning).
            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC_hybrid paper.)
            batch_size (int): Minibatch size for SGD.
            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.
            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
            max_ep_steps (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for EpochLogger.
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """

        self.alg_name = "SAC"
        self.device = device
        self.method = method
        self.prior = prior

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = self.env.observation_space.spaces["observation"].shape[0]
        act_dim = self.env.action_space.shape[0]

        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, size=int(1e6), device=device)

        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.polyak = polyak
        self.num_eval_episodes = num_eval_episodes
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.max_ep_steps = max_ep_steps
        self.update_every = update_every
        self.update_after = update_after
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.use_kl_loss = use_kl_loss
        self.target_KL_div = target_KL_div
        self.target_entropy = target_entropy
        self.a_lr = 3e-4
        self.sigma_prior = sigma_prior

        self.save_freq=save_freq

        # CORE-RL Params
        self.factorC = factor_c
        self.lambda_max = lambda_max

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.networks = actor_critic(obs_dim, act_dim, act_limit, **ac_kwargs).to(self.device)
        self.networks_target = deepcopy(self.networks).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.networks_target.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.networks.q1.parameters(), self.networks.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.networks.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Set up automatic KL div temperature tuning for alpha
        self.alpha = torch.tensor([[10.0]], requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.alpha], lr=self.a_lr)

        # Set up automatic entropy temperature tuning for beta
        self.log_beta = torch.tensor([[-0.01]], requires_grad=True, device=self.device)
        self.beta = self.log_beta.exp()
        self.beta_optimizer = Adam([self.log_beta], lr=self.a_lr)

        # action markers
        self.last_mu_prior = None
        self.last_policy_action = None


    def compute_loss_q(self, data):
        # Set up function for computing SAC_hybrid Q-losses
        state, action, reward, new_state, done, mu_prior2 = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['mu_prior2']

        q1 = self.networks.q1(state, action)
        q2 = self.networks.q2(state, action)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, mu_policy2, sigma_policy2 = self.networks.pi(new_state)

            # Target Q-values
            q1_pi_targ = self.networks_target.q1(new_state, a2)
            q2_pi_targ = self.networks_target.q2(new_state, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            KL_loss = compute_kld_univariate_gaussians(mu_prior2, torch.tensor(self.sigma_prior).to(self.device), mu_policy2,
                                                       sigma_policy2).sum(axis=-1)

            if self.use_kl_loss:
                # KL minimisation regularisation
                backup = reward + self.gamma * (1 - done) * (q_pi_targ - self.alpha * KL_loss)
            else:
                # Maximum entropy backup
                backup = reward + self.gamma * (1 - done) * (q_pi_targ - self.beta * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, q_pi_targ

    def compute_loss_pi(self, data):
        # Set up function for computing SAC_hybrid pi loss
        o, mu_prior = data['obs'], data['mu_prior']
        pi, logp_pi, mu_policy, sigma_policy = self.networks.pi(o)
        q1_pi = self.networks.q1(o, pi)
        q2_pi = self.networks.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        KL_loss = compute_kld_univariate_gaussians(mu_prior, torch.tensor(self.sigma_prior).to(self.device), mu_policy,
                                                   sigma_policy).sum(axis=-1)

        if self.use_kl_loss:
            # Entropy-regularized policy loss
            loss_pi = (self.alpha * KL_loss - q_pi).mean()
        else:
            loss_pi = (self.beta * logp_pi - q_pi).mean()

        return loss_pi, logp_pi, KL_loss

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_pi_targ = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, logp_pi, KL_div = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.networks.parameters(), self.networks_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # KL temperature update
        alpha_loss = self.alpha * (self.target_KL_div - KL_div).detach().mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Entropy temperature update
        self.beta_optimizer.zero_grad()
        beta_loss = (-self.log_beta * (self.target_entropy + logp_pi).detach()).mean()
        beta_loss.backward()
        self.beta_optimizer.step()
        self.beta = self.log_beta.exp()

        # Record things
        return {'loss_q': loss_q.item(),
                'loss_pi': loss_pi.item(),
                'entropy': logp_pi.mean().item(),
                'KL_div': KL_div.mean().item(),
                'q_pi_targ_max': q_pi_targ.max(),
                'q_pi_targ_min': q_pi_targ.min(),
                'q_pi_targ_mean': q_pi_targ.mean(),
                'target_KL_div': self.target_KL_div}

    def update_agent(self):
        batch = self.replay_buffer.sample_batch(self.batch_size)
        metrics = self.update(batch)

        return metrics

    def add_to_replay_buffer(self, state, action, reward, new_state, done):
        mu_prior2 = self.prior.compute_action(self.env.task.goal)

        if self.method == "residual":
            self.replay_buffer.store(state, self.last_policy_action, reward, new_state, done, self.last_mu_prior, mu_prior2)
        else:
            self.replay_buffer.store(state, action, reward, new_state, done, self.last_mu_prior, mu_prior2)

    def get_policy_action(self, o, deterministic=False):
        act, mu, std = self.networks.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic)
        return act, mu, std

    def get_distr(self, state):
        state = torch.FloatTensor(copy(state)).unsqueeze(0).cuda()
        act, mu, std = self.networks.act(state, False)
        return [mu.detach().squeeze(0).cpu().numpy(), std.detach().squeeze(0).cpu().numpy()]

    def get_action_eval(self, state):
        if self.method == "residual":
            act_policy, mu, sigma = self.get_policy_action(state, True)
            act_prior = self.prior.compute_action()
            action = np.clip(act_prior + act_policy, -1, 1)
        elif self.method == "BCF":
            ensemble_actions = [self.get_distr(state)]  # ray.get([get_distr.remote(state, p.ac) for p in agents])
            mu, sigma = fuse_ensembles_stochastic(ensemble_actions)
            dist = Normal(torch.tensor(mu.detach()), torch.tensor(sigma.detach()))
            action = torch.tanh(dist.sample()).numpy()

        else:
            action, mu, sigma = self.get_policy_action(state, True)
        return action

    def get_sample(self):
        action = self.env.action_space.sample()
        mu_prior = self.prior.compute_action(self.env.task.goal)

        self.last_mu_prior = mu_prior

        if self.method == "residual":
            policy_action = action
            action = np.clip(mu_prior + policy_action, -1, 1)

        return action

    def get_action(self, state, kwargs):
        old_state = kwargs["old_state"]
        reward = kwargs["reward"]

        policy_action, mu_policy, std_policy = self.get_policy_action(state)

        mu_prior = self.prior.compute_action(self.env.task.goal)

        self.last_mu_prior = mu_prior
        self.last_policy_action = policy_action

        if self.method == "policy":
            action = policy_action

        if self.method == "CORE-RL":
            with torch.no_grad():
                act_b, _, _, _ = self.networks.pi(torch.as_tensor(old_state, dtype=torch.float32).to(self.device), True,
                                                  False)
                base_q = self.networks.q1(torch.as_tensor(old_state, dtype=torch.float32).to(self.device),
                                          act_b).cpu().numpy()

                act_t, _, _, _ = self.networks.pi(torch.as_tensor(state, dtype=torch.float32).to(self.device), True, False)
                target_q = self.networks.q1(torch.as_tensor(state, dtype=torch.float32).to(self.device),
                                            act_t).cpu().numpy()
            # Compute lambda from measured td-error
            td_error = (reward + self.gamma * target_q) - base_q
            lambda_mix = self.lambda_max * (1 - np.exp(-self.factorC * np.abs(td_error)))
            # Compute the combined action
            action = policy_action / (1 + lambda_mix) + (lambda_mix / (1 + lambda_mix)) * mu_prior

        if self.method == "BCF":
            ensemble_actions = [self.get_distr(state)]#ray.get([get_distr.remote(state, p.ac) for p in agents])
            mu_ensemble, sigma_ensemble = fuse_ensembles_stochastic(ensemble_actions)
            mu_mcf, std_mcf = fuse_controllers(mu_prior, self.sigma_prior, mu_ensemble.cpu().numpy(),
                                               sigma_ensemble.cpu().numpy())

            dist_hybrid = Normal(torch.tensor(mu_mcf).double().detach(), torch.tensor(std_mcf).double().detach())
            action = dist_hybrid.sample()
            action = torch.tanh(action).numpy()

        if self.method == "residual":
            action = np.clip(mu_prior + policy_action, -1, 1)

        return action

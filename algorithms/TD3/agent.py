import itertools

from algorithms.TD3 import core
from algorithms.TD3.replay_buffer import ReplayBuffer
import torch
from copy import deepcopy
from torch.optim import Adam
import numpy as np

class TD3_Agent:
    def __init__(self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
            steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, max_ep_steps=100,
            polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
            update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
            noise_clip=0.5, policy_delay=2, num_test_episodes=10, method = "TD3",
            logger_kwargs=dict(), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), save_freq=1):
        """
        Twin Delayed Deep Deterministic Policy Gradient (TD3)
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                these should return:
                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                               | of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                               | estimate of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ===========  ================  ======================================
            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to TD3.
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
            pi_lr (float): Learning rate for policy.
            q_lr (float): Learning rate for Q-networks.
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
            act_noise (float): Stddev for Gaussian exploration noise added to
                policy at training time. (At test time, no noise is added.)
            target_noise (float): Stddev for smoothing noise added to target
                policy.
            noise_clip (float): Limit for absolute value of target policy
                smoothing noise.
            policy_delay (int): Policy will only be updated once every
                policy_delay times for each update of the Q-networks.
            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
            max_ep_steps (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for EpochLogger.
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """
        self.alg_name= "TD3"
        self.device = device
        self.method = method

        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.spaces["observation"].shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, size=int(1e6))

        self.gamma = gamma
        self.polyak = polyak
        self.num_eval_episodes = num_test_episodes
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.max_ep_steps = max_ep_steps
        self.update_every = update_every
        self.update_after = update_after
        self.steps_per_epoch = steps_per_epoch
        self.a_lr = 3e-4
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip=noise_clip
        self.policy_delay = policy_delay
        self.save_freq = save_freq

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_steps = start_steps

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.networks = actor_critic(self.env.observation_space.spaces["observation"], self.env.action_space, **ac_kwargs)
        self.networks_target = deepcopy(self.networks)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.networks_target.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.networks.q1.parameters(), self.networks.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.networks.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

        # Set up model saving
        # logger.setup_pytorch_saver(self.ac)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.networks.pi, self.networks.q1, self.networks.q2])

        # Set up function for computing TD3 Q-losses
    def compute_loss_q(self,data):
        state, action, reward, next_state, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.networks.q1(state, action)
        q2 = self.networks.q2(state, action)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.networks_target.pi(next_state)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            # Target Q-values
            q1_pi_targ = self.networks_target.q1(next_state, a2)
            q2_pi_targ = self.networks_target.q2(next_state, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = reward + self.gamma * (1 - done) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        q1_pi = self.networks.q1(o, self.networks.pi(o))
        return -q1_pi.mean()

    def update_agent(self):
        for j in range(self.update_every):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            self.update(data=batch, timer=j)


    def update(self, data, timer):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % self.policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Record things
            # logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.networks.parameters(), self.networks_target.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)

    def get_sample(self):
        return self.env.action_space.sample()

    def get_action_eval(self, o):
        a = self.networks.act(torch.as_tensor(o, dtype=torch.float32))
        return np.clip(a, -self.act_limit, self.act_limit)
    def get_action(self, o, _):
        a = self.networks.act(torch.as_tensor(o, dtype=torch.float32))
        a += self.act_noise * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)


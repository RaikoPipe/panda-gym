from algorithms.NAF.replay_buffer import ReplayBuffer, PrioritizedReplay
from algorithms.NAF import core

import torch
import torch.nn as nn 
from torch.nn.utils import clip_grad_norm_
import numpy as np 
import torch.optim as optim


class NAF_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, env_fn, method,
                 device, tau, gamma, n_step, update_every, batch_size,
                 layer_size, seed, learning_rate, loss,
                 per, clip_grad, d2rl, update_after=1000, start_steps=0, epochs=100, q_updates = 0,
                 n_updates= 1, rb_capacity=int(1e6), num_eval_episodes=10,
                 steps_per_epoch=4000):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """

        self.method= method

        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.device = device
        self.TAU = tau
        self.GAMMA = gamma
        self.nstep = n_step
        self.update_every = update_every
        self.NUPDATES = n_updates
        self.BATCH_SIZE = batch_size
        self.Q_updates = 0
        self.per = per
        self.clip_grad = clip_grad
        self.num_eval_episodes = num_eval_episodes
        self.max_ep_steps = self.env.max_ep_steps
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_steps = start_steps
        self.update_after = update_after
        
        self.action_step = 4
        self.last_action = None

        self.networks = core.QNetwork(self.obs_dim, self.act_dim, d2rl=d2rl).to(device)

        self.optimizer = optim.Adam(self.networks.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay memory
        if per == True:
            print("Using Prioritized Experience Replay")
            self.memory = PrioritizedReplay(capacity=rb_capacity,
                                            batch_size=batch_size,
                                            seed=seed,
                                            gamma=gamma,
                                            n_step=self.nstep,
                                            beta_frames=self.steps_per_epoch*epochs)
        else:
            print("Using Regular Experience Replay")
            self.memory = ReplayBuffer(buffer_size=rb_capacity,
                                       batch_size=batch_size,
                                       device=self.device,
                                       seed=seed,
                                       gamma=gamma,
                                       nstep=self.nstep)
        
        # define loss
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "huber":
            self.loss = nn.SmoothL1Loss()
        else:
            print("Loss is not defined choose between mse and huber!")
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0 
    
    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store(state, action, reward, next_state, done)

    def update_agent(self):
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.BATCH_SIZE:
            Q_losses = []
            for _ in range(self.NUPDATES):
                experiences = self.memory.sample()
                if self.per == True:
                    loss = self.learn_per(experiences)
                else:
                    loss = self.learn(experiences)
                self.Q_updates += 1
                Q_losses.append(loss)

    def get_sample(self):
        return self.env.action_space.sample()

    def get_action_eval(self, state):
        """Acts without noise"""
        state = torch.from_numpy(state).float().to(self.device)

        self.networks.qnetwork_local.eval()
        with torch.no_grad():
            _, _, _, action = self.networks.qnetwork_local(state.unsqueeze(0))
        self.networks.qnetwork_local.train()
        return action.cpu().squeeze().numpy().reshape((self.act_dim,))


    def get_action(self, state, kwargs):
        """Calculating the action
        
        Params
        ======
            state (array_like): current state
            
        """

        state = torch.from_numpy(state).float().to(self.device)

        self.networks.qnetwork_local.eval()
        with torch.no_grad():
            action, _, _, _ = self.networks.qnetwork_local(state.unsqueeze(0))
        self.networks.qnetwork_local.train()
        return action.cpu().squeeze().numpy().reshape((self.act_dim,))



    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """

        states, actions, rewards, next_states, dones = experiences

        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, V_, _ = self.networks.qnetwork_target(next_states)

        # Compute Q targets for current states 
        V_targets = rewards + (self.GAMMA**self.nstep * V_ * (1 - dones))
        
        # Get expected Q values from local model
        _, Q, _, _ = self.networks.qnetwork_local(states, actions)

        # Compute loss
        loss = self.loss(Q, V_targets) 
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.networks.qnetwork_local.parameters(), self.clip_grad)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.networks.qnetwork_local, self.networks.qnetwork_target)
            
        return loss.detach().cpu().numpy()

    def learn_per(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones, idx, weights = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, V_, _ = self.networks.qnetwork_target(next_states)

        # Compute Q targets for current states 
        V_targets = rewards + (self.GAMMA**self.nstep * V_ * (1 - dones))
        
        # Get expected Q values from local model
        _, Q, _, _ = self.networks.qnetwork_local(states, actions)

        # Compute loss
        td_error = Q - V_targets
        loss = (self.loss(Q, V_targets)*weights).mean().to(self.device)
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.networks.qnetwork_local.parameters(), self.clip_grad)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.networks.qnetwork_local, self.networks.qnetwork_target)
        # update per priorities
        self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

        
        return loss.detach().cpu().numpy()       

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

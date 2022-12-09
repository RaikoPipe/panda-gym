import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from HER.mpi_utils.mpi_utils import sync_networks, sync_grads
from HER.mpi_utils.normalizer import normalizer
from HER.rl_modules.replay_buffer import replay_buffer
from HER.rl_modules.models import test_T_conditioned_ratio_critic as critic
from HER.rl_modules.models import actor
from HER.her_modules.her import her_sampler
import pdb
import math


"""

implementation of USHER for high-dimensional environments. We use this implementation for Fetch experiments

"""
dual = False
if dual: 
    critic_constructor = dual_critic
else: 
    critic_constructor = critic

train_on_target = True

reward_offset = lambda t : 0#math.log(t+3) -1

class ddpg_agent:
    def __init__(self, args, env, env_params):
        args.replay_k = 0
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        # self.critic_network = critic(env_params)
        self.critic_network = critic_constructor(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        # self.critic_target_network = critic(env_params)
        self.critic_target_network = critic_constructor(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, 
            self.args.replay_k, self.env.compute_reward, args.gamma, args.two_goal, False)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        self.t = 1
        self.global_count = 0
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        agent_name = "δ-DDPG"
        self.agent_name = agent_name.upper()
        key = f"name_{args.env_name}__noise_{args.action_noise}__agent_{agent_name}.txt"
        self.recording_path = "logging/recordings/" + key
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

            with open(self.recording_path, "a") as file: 
                file.write("")

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            self.t = epoch + 1
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            ev = self._eval_agent()
            success_rate, reward, value = ev['success_rate'], ev['reward_rate'], ev['value_rate']


            if MPI.COMM_WORLD.Get_rank() == 0:
                with open(self.recording_path, "a") as file: 
                    file.write(f"{epoch}, {success_rate:.3f}, {reward:.3f}, {value:.3f}\n")
                # print('[{}] epoch is: {}, eval success rate is: {:.3f}, average reward is: {:.3f}'.format(datetime.now(), epoch, success_rate, reward))
                print(f'[{datetime.now()}] epoch is: {epoch}, '
                    f'eval success rate is: {success_rate:.3f}, '
                    f'average reward is: {reward:.3f}, '
                    f'average value is: {value:.3f}')
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')


    # pre_process the inputs
    def _preproc_inputs(self, obs, g, gpi = None):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        gpi_norm = [] if type(gpi) == type(None) else [self.g_norm.normalize(gpi)]
        inputs = np.concatenate([obs_norm, g_norm] + gpi_norm)#, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        self.actor_network.set_normalizers(self.o_norm.get_torch_normalizer(), self.g_norm.get_torch_normalizer())

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)
            
    def get_input_tensor(self, obs, goal, policy_goal):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(goal)
        pol_g_norm = self.g_norm.normalize(policy_goal)

        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        inputs_norm_pol = np.concatenate([obs_norm, g_norm, pol_g_norm], axis=1)

        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_norm_tensor_pol = torch.tensor(inputs_norm_pol, dtype=torch.float32)

        return inputs_norm_tensor, inputs_norm_tensor_pol

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        # transitions['policy_g'] = transitions['g']
        gscale = 1

        o, o_next  = transitions['obs'], transitions['obs_next']
        g, sampled_g, policy_g = transitions['g'], transitions['sampled_g'], transitions['policy_g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        _, transitions['sampled_g'] = self._preproc_og(o, sampled_g)
        _, transitions['policy_g'] = self._preproc_og(o, policy_g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])*gscale
        sampled_g_norm = self.g_norm.normalize(transitions['sampled_g'])*gscale
        policy_g_norm = self.g_norm.normalize(transitions['policy_g'])*gscale
        inputs_norm = np.concatenate([obs_norm, g_norm, policy_g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])*gscale
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm, policy_g_norm], axis=1)
        inputs_goal_norm = np.concatenate([obs_norm, sampled_g_norm, policy_g_norm], axis=1)

        policy_input = torch.tensor(np.concatenate([obs_norm, g_norm], axis=1), dtype=torch.float32)
        policy_input_next = torch.tensor(np.concatenate([obs_next_norm, g_next_norm], axis=1), dtype=torch.float32)
        duplicated_g_input = torch.tensor(np.concatenate([obs_norm, g_norm, g_norm], axis=1), dtype=torch.float32)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_goal_tensor = torch.tensor(inputs_goal_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) - reward_offset(self.t)
        exact_goal_tensor = torch.tensor(transitions['exact_goal'], dtype=torch.float32) 
        t = torch.tensor(transitions['t_remaining'], dtype=torch.float32) 
        her_used = torch.tensor(transitions['her_used'], dtype=torch.float32) 
        map_t = lambda t: -1 + 2*t/self.env_params['max_timesteps'] if self.args.apply_ratio else t*0
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            deterministic_policy = True
            if deterministic_policy: 
                actions_next= self.actor_target_network(policy_input_next, deterministic=True)
                q_next_value, p_next_value = self.critic_target_network(inputs_next_norm_tensor, map_t(t), actions_next, return_p=True) 
            else:  
                actions_next, log_prob_next = self.actor_target_network(policy_input_next, with_logprob = True)
                q_next_value, p_next_value = self.critic_target_network(inputs_next_norm_tensor, map_t(t), actions_next, return_p=True) 

        q0, p0 = self.critic_network(inputs_norm_tensor, map_t(t), actions_tensor, return_p=True)

        _, on_policy_input = self.get_input_tensor(transitions['obs'], transitions['ag_next'], transitions['policy_g'])
        _ , realized_p = self.critic_network(on_policy_input, map_t(t), actions_tensor, return_p=True)
        # _, indep_goal_input = self.get_input_tensor(transitions['obs'], transitions['alt_g'], transitions['policy_g'])
        # _, indep_goal_input_next = self.get_input_tensor(transitions['obs'], transitions['alt_g'], transitions['policy_g'])

        realized_p = torch.clamp(realized_p, 0, 10e6)

        # q_indep_goal, p_indep_goal = self.critic_network(indep_goal_input, map_t(t), actions_tensor, return_p=True)
        # q_indep_goal_next, p_indep_goal_next = self.critic_target_network(indep_goal_input_next, map_t(t), actions_tensor, return_p=True)

        critic_loss = (q0).pow(2).mean()*0 #Make sure the gradients exist even if we don't use them
        critic_loss = critic_loss + ((p0).pow(2)  - (t-1)/t*(p0*p_next_value)).mean()
        # critic_loss = critic_loss + ((p_indep_goal).pow(2) - (t-1)/t*p_indep_goal*p_indep_goal_next).mean() 
        critic_loss = critic_loss - (realized_p/t).mean()/1000


        # the actor loss
        self.global_count += 1
        if self.global_count % 2 == 0:
            actions_real, log_prob = self.actor_network(policy_input, with_logprob = True)
            p_val = self.critic_network(duplicated_g_input, map_t(t), actions_real, return_p=True)[1]
            actor_loss = -p_val.mean() 
            actor_loss += self.args.entropy_regularization*log_prob.mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            sync_grads(self.actor_network)
            self.actor_optim.step()

        # start to update the network
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    def _eval_agent(self, final=False):
        total_success_rate = []
        total_reward_rate = []
        total_value_rate = []
        test_num_multiplier = 20 if final else 1
        write_file = MPI.COMM_WORLD.Get_rank() == 0 and final
        for i in range(self.args.n_test_rollouts*test_num_multiplier):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            total_r = 0
            total_value = 0

            loc = "logging/action_plans"
            if write_file:
                try: 
                    os.listdir(loc + f"/{self.args.env_name}")
                except: 
                    os.mkdir(loc + f"/{self.args.env_name}")
                filename = loc + f"/{self.args.env_name}/{self.agent_name}_epoch_{self.epoch}_plan_{i}.txt"
                f = open(filename, "w")
                f.write(f"Goal: {g.tolist()}\n")
                f.write(f"Path blocked?: {self.env.block_position}\n")
                def write_action(action):
                    f.write(f"duration 0.5: {(action*127).tolist()}\n")

            with torch.no_grad():
                pi = self.actor_network.normed_forward(obs, g, deterministic=True)
                actions_tensor = pi.detach().cpu()
                actions = actions_tensor.numpy().squeeze(axis=0)
                inputs_norm_tensor = self._preproc_inputs(obs, g, gpi=g)
                value = self.critic_network(inputs_norm_tensor, torch.tensor([[1]]), actions_tensor).mean().numpy().squeeze()
                total_value += value

            #for t in range(self.env_params['max_timesteps']):
            for t in range(int(3/(1-self.args.gamma))):
                with torch.no_grad():
                    pi = self.actor_network.normed_forward(obs, g, deterministic=True)
                    inputs_norm_tensor = self._preproc_inputs(obs, g, gpi=g)
                    actions_tensor = pi.detach().cpu()
                    actions = actions_tensor.numpy().squeeze(axis=0)
                    if write_file: write_action(actions)
                observation_new, r, _, info = self.env.step(actions)
                total_r += r*self.args.gamma**t
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
            total_reward_rate.append(total_r)
            total_value_rate.append(total_value)
            if write_file: f.close()
        total_success_rate = np.array(total_success_rate)
        total_reward_rate = np.array(total_reward_rate)
        total_value_rate = np.array(total_value_rate)

        local_success_rate = np.mean(total_success_rate[:, -1])
        local_reward_rate = np.mean(total_reward_rate)
        local_value_rate = np.mean(total_value_rate)#/self.env_params['max_timesteps']
        
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_reward_rate = MPI.COMM_WORLD.allreduce(local_reward_rate, op=MPI.SUM)
        global_value_rate = MPI.COMM_WORLD.allreduce(local_value_rate, op=MPI.SUM)

        return {
            'success_rate': global_success_rate / MPI.COMM_WORLD.Get_size(), 
            'reward_rate': global_reward_rate / MPI.COMM_WORLD.Get_size(), 
            'value_rate': global_value_rate / MPI.COMM_WORLD.Get_size(), 
            }
import numpy as np
import copy

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func, gamma, two_goal, geometric):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

        self.gamma = gamma
        self.two_goal = two_goal
        self.geometric = geometric

    # def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
    #     T = episode_batch['actions'].shape[1]
    #     rollout_batch_size = episode_batch['actions'].shape[0]
    #     batch_size = batch_size_in_transitions
    #     # select which rollouts and which timesteps to be used
    #     episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    #     t_samples = np.random.randint(T, size=batch_size)
    #     transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
    #     transitions['policy_g'] = copy.deepcopy(transitions['g'])
    #     # her idx
    #     her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
    #     future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    #     future_offset = future_offset.astype(int)
    #     future_t = (t_samples + 1 + future_offset)[her_indexes]
    #     # gamma her idx
    #     # future_offset = np.random.geometric(1-self.gamma, size=batch_size) 
    #     #     #s' is the earliest posible goal state
    #     # her_indexes = np.where(np.logical_and(
    #     #     np.random.uniform(size=batch_size) < self.future_p,
    #     #     future_offset + t_samples < T))
    #     # future_t = (t_samples + future_offset)[her_indexes]

    #     # replace go with achieved goal
    #     future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
    #     future_sampled_ag = episode_batch['ag'][episode_idxs, 
    #         (t_samples + future_offset%(T - t_samples))]
    #     transitions['g'][her_indexes] = future_ag
    #     transitions['sampled_g'] = future_sampled_ag
    #     # to get the params to re-compute reward
    #     transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
    #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

    #     # import pdb
    #     # pdb.set_trace()

    #     return transitions


    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # transitions['policy_g'] = copy.deepcopy(transitions['g'])
        # her idx
        if not self.geometric:
            her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
        elif self.geometric:
            future_offset = np.random.geometric(1-self.gamma, size=batch_size) 
                #s' is the earliest posible goal state
            her_indexes = np.where(np.logical_and(
                np.random.uniform(size=batch_size) < self.future_p,
                future_offset + t_samples < T))
            future_t = (t_samples + future_offset)[her_indexes]
            
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        future_sampled_ag = episode_batch['ag'][episode_idxs, 
            (t_samples + future_offset%(T - t_samples))]

        her_used = np.zeros_like(future_offset)
        her_used[her_indexes] = 1
        transitions['her_used'] = her_used
        mask = np.zeros_like(future_offset)
        mask[her_indexes] = 1
        transitions['exact_goal'] = np.zeros_like(future_offset)
        transitions['exact_goal'][np.where(future_offset[her_indexes] == 0)] = 1
        transitions['exact_goal'] *= mask
        transitions['t_remaining'] = T - t_samples
        transitions['alt_g'] = transitions['g'].copy() + np.random.normal(scale=0.1,size=transitions['g'].shape)
        np.random.shuffle(transitions['alt_g'])
        if self.two_goal:
            transitions['policy_g'] = transitions['g'].copy()
            transitions['g'][her_indexes] = future_ag 
        else: 
            transitions['g'][her_indexes] = future_ag 
            transitions['policy_g'] = transitions['g'].copy()
        # gamma her idx

        # replace go with achieved goal
        transitions['g'][her_indexes] = future_ag
        transitions['sampled_g'] = future_sampled_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions['alt_r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['alt_g'], None), 1)
        transitions['exact_goal'] = np.expand_dims(transitions['exact_goal'], 1)
        transitions['t_remaining'] = np.expand_dims(transitions['t_remaining'], 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        # import pdb
        # pdb.set_trace()

        return transitions

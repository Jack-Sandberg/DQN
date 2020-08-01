# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:51:59 2020

@author: Jack
"""
#From https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import numpy as np
import random
import torch
import h5py

from utils.segment_tree import SumSegmentTree, MinSegmentTree
from utils.atari_wrappers import LazyTorch

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class ReplayBufferTorch(ReplayBuffer):
    
    def __init__(self, size, device):
        """Create Replay Buffer that saves transitions as Torch tensors on 
           the specified device.
        """
        super().__init__(size)
        self.device = device
    def add(self, obs_t, action, reward, obs_tp1, done):
        # action = torch.tensor(action, device = self.device)
        # reward = torch.tensor(reward, device = self.device)
        # done = torch.tensor(done, device = self.device)
        
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(obs_t.__tensor__().unsqueeze(0))
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1.__tensor__().unsqueeze(0))
            dones.append(done)
        # return obses_t
        # cpu_count = 0
        # for obs in obses_t:
        #     if obs.device.type == 'cpu':
        #         cpu_count += 1
        # print(cpu_count/len(obses_t))
        # cpu_count = 0
        # for obs in obses_tp1:
        #     if obs is not None:
        #         if obs.device.type == 'cpu':
        #             cpu_count += 1
        # print(cpu_count/len(obses_tp1))
        # first = torch.cat(obses_t,0).permute(0,3,2,1)
        # second = torch.tensor(actions, device = self.device)
        # third = torch.tensor(rewards, device = self.device)
        # fourth =  torch.cat(obses_tp1,0).permute(0,3,2,1)
        # fifth = torch.tensor(dones, device = self.device)
        return torch.cat(obses_t,0), torch.tensor(actions, device = self.device), torch.tensor(rewards, device = self.device), torch.cat(obses_tp1,0), torch.tensor(dones, device = self.device)
        return torch.cat(obses_t,0), torch.stack(actions), torch.stack(rewards), torch.cat(obses_tp1,0), torch.stack(dones)
    def save(self, path):
        """Saves memory at path.
        """
        # _storage entries are organized as (obs_t, action, reward, obs_tp1, done)
        with h5py.File(path, 'w') as f:
            f.create_dataset('obs_t', data = [self._storage[i][0].__tensor__().cpu().numpy() for i in range(len(self))])
            f.create_dataset('action', data = [self._storage[i][1] for i in range(len(self))]) 
            f.create_dataset('reward', data = [self._storage[i][2] for i in range(len(self))])
            f.create_dataset('obs_tp1', data = [self._storage[i][3].__tensor__().cpu().numpy() for i in range(len(self))])
            f.create_dataset('done', data = [self._storage[i][4] for i in range(len(self))])
    def load(self, path):
        """Loads memory from path and deletes previous entries if they exist.
        """
        self._storage = []
        self._next_idx = 0
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        with h5py.File(path, 'r') as f:
            for i in range(f.get('done').len()):    
                obs_t = f.get('obs_t')[i]
                obs_t = LazyTorch([ torch.from_numpy(obs_t[j,:,:]).to(self.device).unsqueeze(0) for j in range(obs_t.shape[0]) ], self.device)
                action = f.get('action')[i]
                reward = f.get('reward')[i]
                obs_tp1 = f.get('obs_tp1')[i]
                obs_tp1 = LazyTorch([ torch.from_numpy(obs_tp1[j,:,:]).to(self.device).unsqueeze(0) for j in range(obs_tp1.shape[0]) ], self.device)
                done = f.get('done')[i]
                
                self.add(obs_t, action, reward, obs_tp1, done)
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super().__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
            
class PrioritizedReplayBufferTorch(ReplayBufferTorch):
    def __init__(self, size, alpha, device):
        #print(self.__mro__)
        super().__init__(size, device)
        assert alpha >= 0
        self._alpha = alpha
        it_capacity = 2
        while it_capacity < size:
            it_capacity *= 2
        
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        
    def add(self, obs_t, action, reward, obs_tp1, done):
        idx = self._next_idx
        
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
    
    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0,len(self._storage)-1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i *every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
    
    def sample(self, batch_size, beta):
        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, dtype = torch.float32, device = self.device)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])
    
    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        assert all(0 <= x < len(self._storage) for x in idxes)
        assert (priorities > 0).all()
        self._max_priority = max(self._max_priority, max(priorities))
        for idx, priority in zip(idxes, priorities):
            #assert priority > 0
            #assert 0 <= idx < len(self._storage)
            #print(priority)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            #self._max_priority = max(self._max_priority, priority)
            
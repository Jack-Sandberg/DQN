# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:07:43 2020

@author: Jack
"""

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from skimage.transform import resize
import time
import datetime
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from gameAnimation import * 
import numpy as np
from utils.replay_buffer import *

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.height = h
        self.width = w
        self.outputs = outputs
        
        self.conv1 = nn.Conv2d(4,32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32,64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64,64, kernel_size = 3, stride = 1)
        
        def conv2d_size_out(size, kernel_size , stride):
            return (size - (kernel_size -1) -1 ) // stride + 1
        w1 = conv2d_size_out(w,8,4)
        w2 = conv2d_size_out(w1,4,2)
        w3 = conv2d_size_out(w2,3,1)
        h1 = conv2d_size_out(h,8,4)
        h2 = conv2d_size_out(h1,4,2)
        h3 = conv2d_size_out(h2,3,1)
        #print(w3,h3,w3*h3)
        self.linear1 = nn.Linear(w3*h3*64,512)
        self.linear2 = nn.Linear(512,outputs)

    def forward(self,x):
        x = self.conv1(x.float()/255)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return(x)
    
    def select_action(self, state, eps, device):
        """
        Returns best_action, max_q_value, ave_q_value if greedy action is taken,
        returns random_action, None, None if random action is taken.
        """
        sample = random.random()
        if sample > eps:
            with torch.no_grad():
                #max_action_value = self.forward(state).max(1)
                action_values  = self.forward(state)
                #print(max_action_value.values)
                return action_values.max(1)[1].view(1,1), float(action_values.max(1).values), float(action_values.mean())
        else:
            return torch.tensor([[random.randrange(self.outputs)]], device = device, dtype = torch.long), None, None

class DuelingDQN(DQN):
    def __init__(self, h, w, outputs):
        super(DQN,self).__init__()
        self.height = h
        self.width = w
        self.outputs = outputs
        
        self.conv1 = nn.Conv2d(4,32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32,64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64,64, kernel_size = 3, stride = 1)
        def conv2d_size_out(size, kernel_size , stride):
            return (size - (kernel_size -1) -1 ) // stride + 1
        w1 = conv2d_size_out(w,8,4)
        w2 = conv2d_size_out(w1,4,2)
        w3 = conv2d_size_out(w2,3,1)
        h1 = conv2d_size_out(h,8,4)
        h2 = conv2d_size_out(h1,4,2)
        h3 = conv2d_size_out(h2,3,1)
        
        self.value_linear = nn.Linear(w3*h3*64, 512)
        self.value_out = nn.Linear(512,1)
        
        self.advantage_linear = nn.Linear(w3*h3*64, 512)
        self.advantage_out = nn.Linear(512, outputs)
    
    def forward(self, x):
        x = F.relu(self.conv1(x.float()/255))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim = 1)
        
        v = F.relu(self.value_linear(x))
        v = self.value_out(v)
        
        a = F.relu(self.advantage_linear(x))
        a = self.advantage_out(a)
        
        q = v + a - a.mean()
        return q
    
class DQNAgent(object):
    def __init__(self, h, w, outputs, device, game, env, memory, writer, eps_schedule, params, beta_schedule = None):
        self.policy_net = DQN(h,w,outputs).to(device)
        self.target_net = DQN(h,w,outputs).to(device)
        self.eps_schedule = eps_schedule
        self.n_actions = outputs
        self.memory = memory
        if isinstance(memory, PrioritizedReplayBufferTorch):
            self.prio = True
        else:
            self.prio = False
        self.device = device
        self.writer = writer
        self.env = env
        self.game = game
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = params.ADAM_LR)
        self.max_reward = -float('Inf')
        self.params = params
        self.beta_schedule = beta_schedule
    def initReplayBuffer(self, init_size):
        random_reward = []
        while len(self.memory) < init_size:
            frame = self.env.reset()
            cum_reward = 0
            elapsed_time = time.time()
            for t in count():
                action = random.randrange(self.n_actions)
                new_frame, reward, terminal, info = self.env.step(action)
                self.memory.add(frame, action, reward, new_frame, terminal)
                cum_reward += reward
                frame = new_frame
                if terminal and info['ale.lives'] == 0:
                    random_reward.append(cum_reward)
                    break
            elapsed_time = time.time() - elapsed_time
            print('ReplayMemory size: {}, Reward: {}, Rate: Rate: {:.2f}'.format(len(self.memory),cum_reward,t/elapsed_time))
        if len(random_reward)>0: 
            print('Random play finished, average reward: {}'.format(sum(random_reward)/len(random_reward)))
            self.writer.add_scalar('random_baseline_reward', sum(random_reward)/len(random_reward))
        else:
            print('ReplayBuffer larger than init_size')
    def load_agent(self ,pnet_path, tnet_path, logpath):
        line = None
        with open(logpath,'r') as f:
            next(f,None)
            for line in f: pass
        if line is not None:
            line = line.split(',')
            self.episodes_done = int(line[0])
            self.steps_done = int(line[1])
            self.steps_at_last_update = int(line[2])
            
        if os.path.exists(pnet_path):
            self.policy_net.load_state_dict(torch.load(pnet_path))
        if os.path.exists(tnet_path):
            self.target_net.load_state_dict(torch.load(tnet_path))
        
    def optimize(self, batch_size, beta = None):
        if self.prio:
            state_batch, action_batch, reward_batch, new_state_batch, terminal_flag_batch = self.memory.sample(batch_size, beta)
        else:
            state_batch, action_batch, reward_batch, new_state_batch, terminal_flag_batch = self.memory.sample(batch_size)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.reshape(batch_size,1).long())
        non_final_next_states = new_state_batch[~terminal_flag_batch,...]
        next_state_values = torch.zeros(batch_size, device = self.device)
        
        with torch.no_grad():
            next_state_values[~terminal_flag_batch] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values*self.params.GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values,expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    #def train(self, num_episodes, update_frequency, target_network_update_frequency, eval_frequency, eval_length, eval_eps, batch_size, logpath, save_frequency):
    def train(self, num_episodes, logpath):
        if not hasattr(self, 'episodes_done'):
            self.episodes_done = 0
        if not hasattr(self,'steps_done'):
            self.steps_done = 0
        if not hasattr(self,'steps_at_last_update'):
            self.steps_at_last_update = 0
            
        try:
            for i_episode in range(self.episodes_done+1, self.episodes_done + num_episodes + 1):
                elapsed_time = time.time()
                frame = self.env.reset()
                cum_reward = 0
                max_q = -float('Inf')
                episode_ave_q_val =0
                actions_chosen = 0
                for t in count():
                    action, q_val, ave_q_val = self.policy_net.select_action(state = frame.__tensor__().unsqueeze(0), eps = self.eps_schedule.get_eps(self.steps_done), device = self.device)
                    if (q_val is not None) and max_q < q_val:
                        max_q = q_val
                    if (ave_q_val is not None):
                        actions_chosen += 1
                        episode_ave_q_val += (ave_q_val-episode_ave_q_val)/actions_chosen
                
                    self.steps_done += 1
                    new_frame, reward, terminal, info = self.env.step(action)
                    self.memory.add(frame,int(action),reward,new_frame,terminal)
                    frame = new_frame
                    cum_reward += reward
                    
                    if self.steps_done % self.params.UPDATE_FREQUENCY == 0:
                        self.optimize(self.params.BATCH_SIZE, self.beta_schedule.get_eps(self.steps_done) if self.prio else None)
                    if (self.steps_done - self.steps_at_last_update) > self.params.TARGET_NETWORK_UPDATE_FREQUENCY:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                        self.steps_at_last_update = self.steps_done
                        print('Updated')
                    if terminal and info['ale.lives'] == 0:
                        self.writer.add_scalar('training_reward', cum_reward, i_episode)
                        self.writer.add_scalar('training_max_q', max_q, i_episode)
                        self.writer.add_scalar('training_ave_q', episode_ave_q_val, i_episode)
                        break
                if self.max_reward < cum_reward:
                    self.max_reward = cum_reward
                    self.writer.add_scalar('training_max_reward', self.max_reward, i_episode)
                elapsed_time = time.time() - elapsed_time
                if i_episode % self.params.EVAL_FREQUENCY == 0:
                    eval_rewards = 0
                    max_eval_reward = -float('Inf')
                    max_rgb_frames = []
                    for eval_episode in range(self.params.EVAL_LENGTH):
                        rgb_frames = []
                        curr_reward = 0
                        frame = self.env.reset()
                        for t in count():
                            action, _, _ = self.policy_net.select_action(state = frame.__tensor__().unsqueeze(0), eps = self.params.EVAL_EPS, device = self.device)
                            new_frame, reward, terminal, info = self.env.step(action)
                            eval_rewards += reward
                            curr_reward += reward
                            rgb_frames.append(info['rgbframe'])
                            frame = new_frame
                            if terminal and info['ale.lives'] == 0:
                                if curr_reward > max_eval_reward:
                                    max_eval_reward = curr_reward
                                    max_rgb_frames = rgb_frames
                                break
                    eval_rewards /= self.params.EVAL_LENGTH
                    filename_gif = self.game + '/eval_ep_'+str(i_episode)+ '_step_' + str(self.steps_done) + '.gif'
                    self._anim = display_frames_as_gif(max_rgb_frames, filename_gif, show = False)
                    self.writer.add_scalar('eval_reward', eval_rewards, i_episode)
                print('Episode: {}/{}, Steps: {}, Reward: {}, Time: {:.2f}, Rate: {:.2f}'.format(i_episode,self.episodes_done + num_episodes,self.steps_done, cum_reward, elapsed_time,t/elapsed_time))
                with open(logpath, 'a', newline = '') as file:
                    csv_writer = csv.writer(file, delimiter = ',')
                    timeString = datetime.datetime.now().strftime('%d/%m/%y %X')
                    csv_writer.writerow([i_episode,self.steps_done,self.steps_at_last_update,cum_reward,timeString, '{:.2f}'.format(elapsed_time), '{:.2f}'.format(t/elapsed_time)])
                if i_episode % self.params.SAVE_FREQUENCY == 0:
                    torch.save(self.policy_net.state_dict(), self.game + '/policy_net' + str(i_episode) + 'eps.pt')
                
        except KeyboardInterrupt as e:
            print(e)
            print('KeyboardInterrupt detected, initializing saving.')
            # self.writer.close()
            # self.episodes_done = i_episode
            # self.save()
        finally:
            print('Saving initialized')
            self.writer.close()
            self.episodes_done = i_episode
            self.save()
    def save(self):
        torch.save(self.policy_net.state_dict(), self.game + '/policy_netLatest.pt')
        torch.save(self.target_net.state_dict(), self.game + '/target_netLatest.pt')
        self.memory.save('E:/Programmering/'+ self.game + 'Memory/memory.h5')

class DoubleDQNAgent(DQNAgent):
    def optimize(self, batch_size, beta = None):
        if self.prio:
            state_batch, action_batch, reward_batch, new_state_batch, terminal_flag_batch, weights, idxes =  self.memory.sample(batch_size, beta)
        else:
            state_batch, action_batch, reward_batch, new_state_batch, terminal_flag_batch = self.memory.sample(batch_size)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.reshape(batch_size,1).long())
        non_final_next_states = new_state_batch[~terminal_flag_batch,...]
        next_state_values = torch.zeros(batch_size, device = self.device)
        
        with torch.no_grad():
            next_state_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(-1)
            next_state_values[~terminal_flag_batch] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)
        expected_state_action_values = (next_state_values*self.params.GAMMA) + reward_batch
        if self.prio:
            loss = (weights*F.smooth_l1_loss(state_action_values,expected_state_action_values.unsqueeze(1))).mean()
            self.memory.update_priorities(idxes, np.array(torch.abs(state_action_values-expected_state_action_values.unsqueeze(1)+1e-16).detach().cpu()))
        else:
            loss = F.smooth_l1_loss(state_action_values,expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def save(self):
        torch.save(self.policy_net.state_dict(), self.game + '/dpolicy_netLatest.pt')
        torch.save(self.target_net.state_dict(), self.game + '/dtarget_netLatest.pt')
        self.memory.save('E:/Programmering/'+ self.game + 'Memory/memory.h5')

class DuelingDoubleDQNAgent(DoubleDQNAgent):
    def __init__(self, h, w, outputs, device, game, env, memory, writer, eps_schedule, params, beta_schedule = None):
        self.policy_net = DuelingDQN(h,w,outputs).to(device)
        self.target_net = DuelingDQN(h,w,outputs).to(device)
        self.eps_schedule = eps_schedule
        self.n_actions = outputs
        self.memory = memory
        if isinstance(memory, PrioritizedReplayBufferTorch):
            self.prio = True
        else:
            self.prio = False
        self.device = device
        self.writer = writer
        self.env = env
        self.game = game
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = params.ADAM_LR)
        self.max_reward = -float('Inf')
        self.params = params
        self.beta_schedule = beta_schedule
    
    def save(self):
        torch.save(self.policy_net.state_dict(), self.game + '/ddpolicy_netLatest.pt')
        torch.save(self.target_net.state_dict(), self.game + '/ddtarget_netLatest.pt')
        self.memory.save('E:/Programmering/'+ self.game + 'Memory/memory.h5')
        
class EpsilonScheduler(object):
    def __init__(self,eps_start = 1,eps_end = 0.1,eps_number_of_frames = 10**6):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_number_of_frames = eps_number_of_frames
    
    def get_eps(self,frame_number):
        if frame_number > self.eps_number_of_frames:
            return self.eps_end
        return self.eps_start - (self.eps_start - self.eps_end)*frame_number/self.eps_number_of_frames

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:11:11 2020

@author: Jack
"""

from DQN import *

from torch.utils.tensorboard import SummaryWriter
import os

from utils.atari_wrappers import make_atari, wrap_deepmind_torch
from utils.replay_buffer import ReplayBuffer, ReplayBufferTorch, PrioritizedReplayBufferTorch
from argparse import Namespace
### Using https://ai.stackexchange.com/questions/10203/dqn-stuck-at-suboptimal-policy-in-atari-pong-task
#       https://ai.stackexchange.com/questions/10306/each-training-run-for-ddqn-agent-takes-2-days-and-still-ends-up-with-13-avg-sc/10481#10481
#       https://medium.com/@shmuma/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55

UPDATE_FREQUENCY = 4
BATCH_SIZE = 32*UPDATE_FREQUENCY
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02 # 0.1 in Deepmind paper
EPS_NUMBER_OF_FRAMES = 1*10**5 #10**6 in Deepmind paper
TARGET_NETWORK_UPDATE_FREQUENCY = (10**3) #10**4 in Deepmind paper
MEMORY_SIZE = 1*10**5 #10**6 in Deepmind paper
INITIAL_REPLAY_SIZE = 1*10**3 #5*10**4 in Deepmind paper
RMS_PROP_OPTIM = False
ADAM_LR = 10**(-4)

PRIO_ALPHA = 0.6
PRIO_BETA = 0.4
PRIO_BETA_END = 1.0
BETA_NUMBER_OF_UPDATES = EPS_NUMBER_OF_FRAMES*20

NUM_EPISODES = 10
SAVE_FREQUENCY = 600

EVAL_FREQUENCY = 300
EVAL_LENGTH = 5
EVAL_EPS = EPS_END/10

params_dict = {'UPDATE_FREQUENCY': UPDATE_FREQUENCY, 
          'BATCH_SIZE': BATCH_SIZE,
          'GAMMA': GAMMA,
          'TARGET_NETWORK_UPDATE_FREQUENCY': TARGET_NETWORK_UPDATE_FREQUENCY,
          'ADAM_LR': ADAM_LR,
          'EVAL_FREQUENCY': EVAL_FREQUENCY,
          'EVAL_LENGTH': EVAL_LENGTH,
          'EVAL_EPS': EVAL_EPS,
          'NUM_EPISODES': NUM_EPISODES,
          'SAVE_FREQUENCY': SAVE_FREQUENCY,
          'PRIO_ALPHA': PRIO_ALPHA,
          'PRIO_BETA': PRIO_BETA,
         'BETA_NUMBER_OF_UPDATES': BETA_NUMBER_OF_UPDATES}
params = Namespace(**params_dict)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = 'Pong'
    
    ENV_GYM = game + 'NoFrameskip-v4'
    env = make_atari(ENV_GYM)
    env = wrap_deepmind_torch(env, device, frame_stack = True, scale = False)
    eps_schedule = EpsilonScheduler(EPS_START, EPS_END, EPS_NUMBER_OF_FRAMES)
    beta_schedule = EpsilonScheduler(PRIO_BETA, PRIO_BETA_END, BETA_NUMBER_OF_UPDATES)
    #memory = ReplayBufferTorch(MEMORY_SIZE, device)
    memory = PrioritizedReplayBufferTorch(MEMORY_SIZE, PRIO_ALPHA, device)
    mem_path = 'E:/Programmering/'+ game + 'Memory/memory.h5'
    if os.path.exists(mem_path) and isinstance(ReplayBufferTorch,type(memory)):
        print('Loading memory')
        memory.load(mem_path)
    n_actions = env.action_space.n
    tb_path = 'tb/' + game + 'EnvLazy'
    tb_version = 1
    while os.path.exists(tb_path + str(tb_version)):
        tb_version += 1
    tb_path = tb_path + str(tb_version)
    writer = SummaryWriter(tb_path)
    
    dqn_agent = DuelingDoubleDQNAgent(h = 84, w = 84, outputs = n_actions, device = device,
                         game = game, env = env, memory = memory, writer = writer,
                         eps_schedule = eps_schedule, params = params, beta_schedule = beta_schedule)
    
    pnet_path = game + '/ddpolicy_netLatest.pt'
    tnet_path = game + '/ddtarget_netLatest.pt'
    logpath = game + '/' + game + 'log.csv'
    
    dqn_agent.load_agent(pnet_path, tnet_path, logpath)
    
    dqn_agent.initReplayBuffer(INITIAL_REPLAY_SIZE)
    dqn_agent.train(NUM_EPISODES, logpath)
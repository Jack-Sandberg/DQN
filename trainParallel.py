# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:36:48 2020

@author: Jack
"""


from DQN import *

import torch
import torch.multiprocessing as mp

import csv
import time
import os.path
import sys
import traceback
import datetime
import h5py
import shutil
from utils.adeque import AsyncDeque
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from utils.atari_wrappers import make_atari, wrap_deepmind_torch
from utils.replay_buffer import ReplayBuffer, ReplayBufferTorch
### Using https://ai.stackexchange.com/questions/10203/dqn-stuck-at-suboptimal-policy-in-atari-pong-task
#       https://ai.stackexchange.com/questions/10306/each-training-run-for-ddqn-agent-takes-2-days-and-still-ends-up-with-13-avg-sc/10481#10481
#       https://medium.com/@shmuma/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55

UPDATE_FREQUENCY = 4
PROCESSES_COUNT = 4
BATCH_SIZE = 32*UPDATE_FREQUENCY
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02 # 0.1 in Deepmind paper
EPS_NUMBER_OF_FRAMES = 1*10**4/PROCESSES_COUNT #10**6 in Deepmind paper
TARGET_NETWORK_UPDATE_FREQUENCY = (10**3) #10**4 in Deepmind paper
MEMORY_SIZE = 1*10**5 #10**6 in Deepmind paper
INITIAL_REPLAY_SIZE = 1*10**4 #5*10**4 in Deepmind paper
RMS_PROP_OPTIM = False
ADAM_LR = 10**(-4)
EVAL_FREQUENCY = 500
EVAL_LENGTH = 30
EVAL_EPS = 0.001
MODEL_SAVE_FREQUENCY = 500
num_episodes = 300
CUDA_ASYNC = True

MODEL_SYNC_FREQUENCY = 1
MODEL_SEND_FREQUENCY = 100
game = 'Pong'

USE_TRUE_HYPERPARAMS = False
if USE_TRUE_HYPERPARAMS:
    EPS_NUMBER_OF_FRAMES = 10**6 
    TARGET_NETWORK_UPDATE_FREQUENCY = 10**4 
    MEMORY_SIZE = 10**6
    INITIAL_REPLAY_SIZE = 5*10**4
def optimize_model_lazy(policy_net, target_net, optimizer, memory, device, gamma, batch_size):
    if len(memory) < batch_size:
        return
    state_batch, action_batch, reward_batch, new_state_batch, terminal_flag_batch = memory.sample(batch_size)
    if CUDA_ASYNC and device.type == 'cuda':
        state_batch = state_batch.cuda(non_blocking = CUDA_ASYNC)
        action_batch = action_batch.cuda(non_blocking = CUDA_ASYNC)
        reward_batch = reward_batch.cuda(non_blocking = CUDA_ASYNC)
        new_state_batch = new_state_batch.cuda(non_blocking = CUDA_ASYNC)
        terminal_flag_batch = terminal_flag_batch.cuda(non_blocking = CUDA_ASYNC)
    state_action_values = policy_net(state_batch).gather(1, action_batch.reshape(batch_size,1).long())
    non_final_next_states = new_state_batch[~terminal_flag_batch,...]
    next_state_values = torch.zeros(batch_size, device = device)
    #if sum(~erminal_flag_batch > 0):
    with torch.no_grad():
        next_state_values[~terminal_flag_batch] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values*gamma) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values,expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def optimize_model_buff(policy_net, target_net, optimizer, memory, device, gamma, batch_size):
    if len(memory) < batch_size:
        return
    state_batch, action_batch, reward_batch, new_state_batch, terminal_flag_batch = memory.sample(batch_size)

    state_batch = torch.from_numpy(np.moveaxis(state_batch,3,1)).to(device)
    action_batch = torch.from_numpy(action_batch).to(device)
    reward_batch = torch.from_numpy(reward_batch).to(device).to(torch.float32)
    new_state_batch = torch.from_numpy(np.moveaxis(new_state_batch,3,1)).to(device)
    terminal_flag_batch = torch.from_numpy(terminal_flag_batch).to(device)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch.reshape(batch_size,1).long())
    non_final_next_states = new_state_batch[~terminal_flag_batch,...]
    next_state_values = torch.zeros(batch_size, device = device)
    #if sum(~erminal_flag_batch > 0):
    with torch.no_grad():
        next_state_values[~terminal_flag_batch] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values*gamma) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values,expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def atari_player(replay_queue, game, param_queue, params_to_load, device, params, proc_num):
    
    try:
        num_episodes = params['num_episodes']
        steps_done = 0
        episodes_done = 0
        device = torch.device(device)
        ENV_GYM = game + 'NoFrameskip-v4'
        atari = make_atari(ENV_GYM)
        env_device = torch.device('cpu')
        atari  = wrap_deepmind_torch(atari, env_device, frame_stack = True, scale = False)
        
        n_actions = atari.env.action_space.n
        policy_net = DQN(84,84,n_actions).to(device)
        if param_queue:
            policy_net.load_state_dict(param_queue[0])
            params_to_load[proc_num] = False
        eps_schedule = EpsilonScheduler(EPS_START,EPS_END,EPS_NUMBER_OF_FRAMES)
        
        #Continues training at latest stage if saved properly
        line = None
        logpath = game + '/' + game + 'logParallel' + str(proc_num) + '.csv'
        #logpath = 'Breakout/breakoutlog.csv'
        if os.path.exists(logpath):
            with open(logpath,'r') as f:
                next(f,None)
                for line in f: pass
        if line is not None:
            line = line.split(',')
            episodes_done = int(line[0])
            steps_done = int(line[1])
        
        tb_path = 'tb/' + game + 'EnvParallel'
        tb_version = 1
        while os.path.exists(tb_path + str(tb_version)):
            tb_version += 1
        tb_path = tb_path + str(tb_version) + str(proc_num)
        writer = SummaryWriter(tb_path)
        
        episode_rewards = []
        episode_durations = []
        max_reward = -float('Inf')
        eval_list = []
        tot_put_time = 0
        for i_episode in range(episodes_done+1,episodes_done + num_episodes+1):
                    elapsed_time = time.time()
                    frame = atari.reset()
                    cum_reward = 0
                    max_q = -float('Inf')
                    episode_ave_q_val = 0
                    actions_chosen = 0
                    for t in count():
                        #print('choosing action')
                        action, q_val, ave_q_val = policy_net.select_action(state = frame.__tensor__().unsqueeze(0).to(device), eps = eps_schedule.get_eps(steps_done), device = device)
                        #print('action chosen')
                        if (q_val is not None) and max_q < q_val:
                            max_q = q_val
                        if (ave_q_val is not None):
                            actions_chosen += 1
                            episode_ave_q_val += (ave_q_val-episode_ave_q_val)/actions_chosen
                            
                        steps_done += 1
                        new_frame, reward, terminal, info = atari.step(action)
                        
                        data = (frame,int(action),reward,new_frame,terminal)
                        #print('Putting')
                        put_time = time.time()
                        replay_queue.put(data)
                        tot_put_time += time.time() - put_time
                        frame = new_frame
                        cum_reward += reward
                        
                        if steps_done % params['MODEL_SYNC_FREQUENCY'] == 0 and params_to_load[proc_num] and param_queue:
                            print('Loaded model in process ' + str(proc_num))
                            received_dict = {}
                            for k, v in param_queue[0].items():
                                received_dict[k] = v.to(device)
                            policy_net.load_state_dict(received_dict)
                            params_to_load[proc_num] = False
                        if terminal and info['ale.lives'] == 0:
                            episode_durations.append(t+1)
                            episode_rewards.append(cum_reward)
                            writer.add_scalar('training_reward' , cum_reward, i_episode)
                            writer.add_scalar('training_max_q', max_q, i_episode)
                            writer.add_scalar('training_ave_q', episode_ave_q_val, i_episode)
                            break
                    if max_reward < cum_reward:
                        max_reward = cum_reward
                        writer.add_scalar('training_max_reward', max_reward, i_episode)
                    
                    if i_episode % params['EVAL_FREQUENCY'] == 0:
                        eval_rewards = 0
                        for eval_episodes in range(EVAL_LENGTH):
                            frame = atari.reset()
                            for t in count():
                                action, _, _ = policy_net.select_action(state = frame.__tensor__().unsqueeze(0), eps = eps_schedule.get_eps(steps_done), device = device)
                                new_frame, reward, terminal, info = atari.step(action)
                                eval_rewards += reward
                                frame = new_frame
                                if terminal and info['ale.lives'] == 0:
                                    break
                        eval_list.append(eval_rewards)
                        eval_rewards /= EVAL_LENGTH
                        writer.add_scalar('eval_reward', eval_rewards, i_episode)
                    elapsed_time = time.time() - elapsed_time
                    print('Episode: {}/{}, Steps: {}, Reward: {}, Time: {:.2f}, Rate: {:.2f}, Put: {:.3f}'.format(i_episode,episodes_done + num_episodes,steps_done, cum_reward, elapsed_time,t/elapsed_time, tot_put_time))
                    with open(logpath,'a', newline = '') as file:
                        csv_writer = csv.writer(file, delimiter = ',')
                        timeString = datetime.datetime.now().strftime('%d/%m/%y %X')
                        csv_writer.writerow([i_episode,steps_done,cum_reward,timeString, '{:.2f}'.format(elapsed_time), '{:.2f}'.format(t/elapsed_time), '{:.3f}'.format(tot_put_time)])
                    if i_episode % MODEL_SAVE_FREQUENCY == 0:
                        torch.save(policy_net.state_dict(), game+'/policy_net' + str(i_episode) + 'eps.pt')
    except Exception as e:
        if 'writer' in locals():
            writer.close()
        import sys
        tb = sys.exc_info()
        replay_queue.put(e)
        return(e)
    replay_queue.put(None)
        


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except:
        print('Set start method failed')
    print('Entering main')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    ENV_GYM = game + 'NoFrameskip-v4'
    atari = make_atari(ENV_GYM)
    atari  = wrap_deepmind_torch(atari, device, frame_stack = True, scale = False)
    print('Created atari env')
    n_actions = atari.env.action_space.n
    policy_net = DQN(84,84,n_actions)
    target_net = DQN(84,84,n_actions)
    #Continues training at latest stage if saved properly
    line = None
    logpath = game + '/' + game + 'log.csv'
    #logpath = 'Breakout/breakoutlog.csv'
    if os.path.exists(logpath):
        with open(logpath,'r') as f:
            next(f,None)
            for line in f: pass
    if line is not None:
        line = line.split(',')
        episodes_done = int(line[0])
        steps_done = int(line[1])
        steps_at_last_update = int(line[2])
    print('Read logs')
    #Loads last saved policies if they exist
    pnet_path = game + '/policy_netLatest.pt'
    tnet_path = game + '/target_netLatest.pt'
    if os.path.exists(pnet_path):
        policy_net.load_state_dict(torch.load(pnet_path))
        target_net.load_state_dict(policy_net.state_dict())
    if os.path.exists(tnet_path):
        target_net.load_state_dict(torch.load(tnet_path))
    print('Loaded old nets')
    if RMS_PROP_OPTIM:
        optimizer = optim.RMSprop(policy_net.parameters(), lr = lr, momentum = momentum)
    else:
        optimizer = optim.Adam(policy_net.parameters(), lr = ADAM_LR)
    
    mem_path = 'E:/Programmering/'+ game + 'Memory/memory.h5'
    print('Creating new memory')
    memory = ReplayBufferTorch(MEMORY_SIZE, device)
    print('Finished creating new memory')
    
    # #Fills up ReplayMemory to initial size.
    random_reward = []
    while len(memory) < INITIAL_REPLAY_SIZE:
        frame = atari.reset()
        cum_reward = 0
        elapsed_time = time.time()
        for t in count():
            action = random.randrange(n_actions)
            new_frame, reward, terminal, info = atari.step(action)
            memory.add(frame,action,reward,new_frame,terminal)
            cum_reward += reward
            frame = new_frame
            if terminal and info['ale.lives'] == 0:
                random_reward.append(cum_reward)
                break
        elapsed_time = time.time()-elapsed_time
        print('ReplayMemory size: {}, Reward: {}, Rate: Rate: {:.2f}'.format(len(memory),cum_reward,t/elapsed_time))
    if len(random_reward)>0:    
        print('Random play finished, average reward: {}'.format(sum(random_reward)/len(random_reward)))
    
    policy_net.to(device)
    target_net.to(device)
    policy_net.share_memory()
    target_net.share_memory()
    replay_queue = mp.Queue(maxsize = 2*UPDATE_FREQUENCY*PROCESSES_COUNT)
    manager = mp.Manager()
    param_queue = manager.list([])
    params = {'UPDATE FREQUENCY':UPDATE_FREQUENCY,'EVAL_FREQUENCY':EVAL_FREQUENCY, 'num_episodes':num_episodes,'MODEL_SYNC_FREQUENCY':MODEL_SYNC_FREQUENCY}

    
    play_process_list = []
    
    params_to_load = manager.list()
    
    for _ in range(PROCESSES_COUNT):
        params_to_load.append(True)
    
    for i in range(PROCESSES_COUNT):
        play_process = mp.Process(target = atari_player, args = (replay_queue, game, param_queue, params_to_load, device.type, params, i))
        play_process.start()
        play_process_list.append(play_process)
    optim_index = 0
    tot_get_time = 0
    try:
        while True:
            if optim_index % (UPDATE_FREQUENCY*60) == 0:
                print('Optimization step: ' + str(optim_index) + ', Get time: ' + '{:.3f}'.format(tot_get_time))
            optim_index += UPDATE_FREQUENCY
            for _ in range(UPDATE_FREQUENCY):
                get_time = time.time()
                data = replay_queue.get()
                tot_get_time += time.time() - get_time
                if data is None:
                    for p in play_process_list:
                        p.join()
                    print('Finished reading data')
                    break
                elif issubclass(type(data),Exception) or len(data) == 3:
                    print(data)
                    for p in play_process_list:
                        p.terminate()
                        p.join()
                    
                    raise(data)
                #print('Adding data')
                frame,action,reward,new_frame,terminal = data
                frame._to(device)
                new_frame._to(device)
                memory.add(frame,action,reward,new_frame,terminal)
                #memory.add(frame,action,reward,new_frame,terminal)
            
            optimize_model_lazy(policy_net, target_net, optimizer, memory, device, gamma = GAMMA, batch_size = BATCH_SIZE)
            if optim_index % MODEL_SEND_FREQUENCY < UPDATE_FREQUENCY:
                dict_to_send = {}
                for k,v in policy_net.state_dict().items():
                    dict_to_send[k] = v.cpu()
                if len(param_queue) < 1:
                    param_queue.append(dict_to_send)
                param_queue[0] = dict_to_send
                for i in range(PROCESSES_COUNT):
                    params_to_load[i] = True
            if optim_index % TARGET_NETWORK_UPDATE_FREQUENCY < UPDATE_FREQUENCY:
                target_net.load_state_dict(policy_net.state_dict())
                print('Updated')
            
    except KeyboardInterrupt as e:
        print(e)
        print('KeyboardInterrupt detected, initializing saving.')
        torch.save(policy_net.state_dict(), game + '/policy_netLatest.pt')
        torch.save(target_net.state_dict(), game + '/target_netLatest.pt')
        # with h5py.File(mem_path,'w') as f:
        #     f.create_dataset('actions', data = memory.actions.cpu().numpy())
        #     f.create_dataset('rewards', data = memory.rewards.cpu().numpy())
        #     f.create_dataset('frames', data = memory.frames.cpu().numpy())
        #     f.create_dataset('terminal_flags', data = memory.terminal_flags.cpu().numpy())
        #     f.create_dataset('count', data = memory.count)
        #     f.create_dataset('current', data = memory.current)
        print('Save complete, exiting.')
        for p in play_process_list:
            p.terminate()
            p.join()
        
        sys.exit()
        
    print('Saving initialized')
    save_start = time.time()
    torch.save(policy_net.state_dict(), game + '/policy_netLatest.pt')
    torch.save(target_net.state_dict(), game + '/target_netLatest.pt')
    # with h5py.File(mem_path,'w') as f:
    #     f.create_dataset('actions', data = memory.actions.cpu().numpy())
    #     f.create_dataset('rewards', data = memory.rewards.cpu().numpy())
    #     f.create_dataset('frames', data = memory.frames.cpu().numpy())
    #     f.create_dataset('terminal_flags', data = memory.terminal_flags.cpu().numpy())
    #     f.create_dataset('count', data = memory.count)
    #     f.create_dataset('current', data = memory.current)
    save_end = time.time()
    minutes,seconds = divmod(save_end-save_start,60)
    for p in play_process_list:
        p.terminate()
        p.join()
    print('Save time: {:0>2}:{:05.2f}'.format(int(minutes),seconds))

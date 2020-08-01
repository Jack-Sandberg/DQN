# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:42:32 2020

@author: Jack
"""

#from DQN import *

import csv
import matplotlib.pyplot as plt
from matplotlib import animation



def display_frames_as_gif(frames, filename_gif = None, show = True):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(num = 1,figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72*5)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename_gif: 
        anim.save(filename_gif,fps=20)
    #display(display_animation(anim, default_mode='loop'))  
    if show:    
        plt.show()
    return anim
    


if __name__ == '__main__':
    EVAL_EPS = 0.02
    from DQN import * 
    game = 'Breakout'
    ENV_GYM = game + 'NoFrameskip-v4'
    env = make_atari(ENV_GYM)
    env = wrap_deepmind(env, frame_stack = True, scale = False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_actions = env.action_space.n
    policy_net = DQN(84,84,n_actions).to(device)
    pnet_path = game + '/policy_netLatest.pt'
    
    policy_net.load_state_dict(torch.load(pnet_path))
    
    frame = env.reset()
    frames = [frame.__array__(dtype=np.uint8)]
    
    cum_reward = 0
    for t in count():
        action, _ , _ = policy_net.select_action(state = torch.from_numpy(np.moveaxis(frame.__array__(dtype=np.uint8),2,0)).to(device).unsqueeze(0), eps = EVAL_EPS, device = device)
        
        frame,reward,done,info  = env.step(action)
        cum_reward += reward
        frames.append(frame.__array__(dtype=np.uint8))
        if done and info['ale.lives'] == 0:
            break
    line = None
    with open(game + '/' + game +'log.csv','r') as f:
        for line in f: pass
    if line is not None:
        line = line.split(',')
        episodes_done = line[0]
    frames_fixed = []
    for frame in frames:
        frames_fixed.append(frame[:,:,3])
    outer_anim = display_frames_as_gif(frames_fixed,game + episodes_done +'.gif')
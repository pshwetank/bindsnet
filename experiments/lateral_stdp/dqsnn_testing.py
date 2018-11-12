import os, sys
import random
import pickle
import itertools

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np

from bindsnet import *
from gym import wrappers
from time import time as T
from argparse import ArgumentParser
from collections import deque, namedtuple

# definition of DQANN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class LogPrint(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)    
    def flush(self):
        for f in self.files:
            f.flush()

def load(DQSNN_path, device_id=0):
    DQSNN = load_network(DQSNN_path)
    
    if torch.cuda.is_available():
        for c in DQSNN.connections:
            DQSNN.connections[c].w = DQSNN.connections[c].w.cuda(device_id)
            DQSNN.connections[c].update_rule = None
    else:
        for c in DQSNN.connections:
            DQSNN.connections[c].w = DQSNN.connections[c].w.cpu()
            DQSNN.connections[c].update_rule = None

    return DQSNN
    
def test(DQSNN_path, dt=1.0, runtime=500, episodes=100, epsilon=0, **args):
    if args['log']:
        f = open(os.path.join(os.path.dirname(DQSNN_path), f'{os.path.basename(DQSNN_path)[:3]}_dqsnn.log'), 'w')
        nolog = sys.stdout
        sys.stdout = LogPrint(sys.stdout, f)
    
    DQSNN = load(DQSNN_path, device_id=args['device_id'])
    
    ENV = GymEnvironment('BreakoutDeterministic-v4')
    ACTIONS = torch.tensor([0, 1, 2, 3])
    
    def policy(readout_spikes):
        q_values = torch.Tensor([readout_spikes[i].sum()
                                 for i in range(len(ACTIONS))])
        A = torch.ones(4) * epsilon / 4
        if torch.max(q_values) == 0:
            return torch.tensor([0.25, 0.25, 0.25, 0.25])
        best_action = torch.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    
    episode_rewards = torch.zeros(episodes)
    start_time = T()
    total_steps = 0
    
    for i_episode in range(episodes):
        obs = ENV.reset().cuda()
        state = torch.stack([obs] * 4, dim=2)
        new_life = True
        prev_life = 5
        steps = 0
        
        for t in itertools.count():
            encoded_state = torch.tensor([0.25, 0.5, 0.75, 1]) * state
            encoded_state = torch.sum(encoded_state, dim=2)
            encoded_state = encoded_state.view([1, -1]).repeat(runtime, 1)
            
            inpts = {'X': encoded_state}
            hidden_spikes, readout_spikes = DQSNN.run(inpts, time=runtime) 
            # TODO : this is hacky code for accumulating the total spikes during simulation
            # possible workaround would be to add NoOp connnection to accumulation layer from readout layer with inf threshold and have 1 weights
            
            action_probs = policy(torch.sum(readout_spikes, dim=0))
            action = np.random.choice(ACTIONS, p=action_probs)
            
            if new_life:
                action = 1
            
            next_obs, reward, done, info = ENV.step(action)
            next_obs = next_obs.cuda()
            steps += 1
            if prev_life - info["ale.lives"] != 0:
                new_life = True
            else:
                new_life = False
            prev_life = info["ale.lives"]
    
            next_state = torch.clamp(next_obs - obs, min=0)
            next_state = torch.cat((state[:, :, 1:], next_state.view([next_state.shape[0], next_state.shape[1], 1])), dim=2)
            episode_rewards[i_episode] += reward
            
            state = next_state
            obs = next_obs
        
            if done:
                print(f'Step {steps} ({total_steps}) @ Episode {i_episode+1:03d}/{episodes}')
                print(f'Episode Reward {episode_rewards[i_episode]}')
                sys.stdout.flush()
                break
        total_steps += steps

        
    end_time = T()
    viz_data = {'episode_rewards': episode_rewards}
    viz_data_file = os.path.join(os.path.dirname(DQSNN_path), f'{os.path.basename(DQSNN_path)[:3]}_dqsnn.dat')
    pickle.dump(viz_data, open(viz_data_file, 'wb'))
    print(f'Average Reward over {episodes} Episodes {torch.sum(episode_rewards) / episodes}')
    print(f'Saved Trained Network to {network_file}.')
    print(f'Saved Collected Data to {viz_data_file}.')
    print(f'Total time taken: {end_time - start_time}')
    
    if args['log']:
        sys.stdout = nolog
        f.close()
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runtime', type=int, default=250)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--experiment', type=int, required=True)
    args = vars(parser.parse_args())
    
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args['device_id'])
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(args['seed'])
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    experiment_dir = str(args['experiment'])
    nets = sorted([i for i in os.listdir(experiment_dir) if '.net' in i])
    for n in nets:
        test(os.path.join(experiment_dir, n), **args)
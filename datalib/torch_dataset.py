import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

Demonstrations = namedtuple('Demonstrations', ('states', 'actions', 'rewards', 'dones'))



class DemoDataset(Dataset):
    def __init__(self, demo_path, pytorch=False):
        if pytorch:
            demos = torch.load(demo_path)
            self.states = demos.states
            self.actions = demos.actions
            self.rewards = demos.rewards
        else:
            self.states = torch.tensor(np.load(demo_path + 'states.npy'))
            self.actions = torch.tensor(np.load(demo_path + 'actions.npy'))
            self.rewards = torch.tensor(np.load(demo_path + 'rewards.npy'))
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx]
    

class DemoLoader:
    def __init__(self, dataset, batch_size):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.iterator = iter(self.loader)
    
    def __iter__(self):
        for states, actions, rewards in self.loader:
            states = states.cpu().numpy()
            actions = actions.cpu().numpy()
            rewards = rewards.cpu().numpy()
            yield states, actions, rewards
    
    def __next__(self):
        try:
            states, actions, rewards = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
        states, actions, rewards = next(self.iterator)
        states = np.transpose(states.cpu().numpy(), axes=(0, 2, 3, 1))
        actions = actions.cpu().numpy()
        rewards = rewards.cpu().numpy()
        return states, actions, rewards



if __name__ == '__main__':
    demo_path = './expert_demos/'
    dataset = DemoDataset(demo_path + 'demos.pkl', pytorch=True)
    loader = DemoLoader(dataset, batch_size=1)
    state_list, action_list, reward_list = [], [], []
    for states, actions, rewards in loader:
        state_list.append(states)
        # print(states.shape, actions.shape, rewards.shape)
        action_list.append(actions[0])
        reward_list.append(rewards[0])
    
    state_list = np.concatenate(state_list)
    action_list = np.array(action_list)
    reward_list = np.array(reward_list)
    print(state_list.shape, action_list.shape, reward_list.shape)
    np.save(demo_path + 'states.npy', state_list)
    np.save(demo_path + 'actions.npy', action_list)
    np.save(demo_path + 'rewards.npy', reward_list)
    print('Done!')
    
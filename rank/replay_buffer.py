import torch
import random
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward',  'next_state', 'done'))


class ReplayBuffer(object):
    """
    ReplayBuffer - a cyclic buffer of bounded size that holds the transitions observed recently.
    It also implements a .sample() method for selecting a random batch of transitions for training.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        mini_batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done = transition
            states.append(state.tolist()[0])
            actions.append(action.tolist()[0])
            rewards.append([reward])
            next_states.append(next_state.tolist()[0])
            dones.append([done])
        return torch.tensor(states, dtype=torch.float, device=device), \
               torch.tensor(actions, dtype=torch.float, device=device), \
               torch.tensor(rewards, dtype=torch.float, device=device), \
               torch.tensor(next_states, dtype=torch.float, device=device), \
               torch.tensor(dones, device=device)

    def __len__(self):
        return len(self.memory)

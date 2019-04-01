import random
from collections import namedtuple


class ReplayMemory(object):

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'reward', 'next_state'))
        self.position = 0


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def push(self, *args):
        # saves transition
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.size


    def __len__(self):
        return len(self.memory)
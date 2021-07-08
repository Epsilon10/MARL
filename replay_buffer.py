import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.stack, zip(*batch))
        print("REW BATCH", reward_batch)
        return torch.from_numpy(state_batch), torch.from_numpy(action_batch.reshape((batch_size, 1))), torch.from_numpy(reward_batch.reshape(batch_size, 1)), torch.from_numpy(next_state_batch), torch.from_numpy(done_batch.reshape(batch_size,1))

    def __len__(self):
        return len(self.buffer)
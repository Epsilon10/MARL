# store reward machine state in replay buffer

# seperate replay buffer for each reward machine state

# sac for each reward machine state

# each rm state has seperate q function


# reward machine: initial state. intermediate state, final state

from sac_discrete import SAC_Discrete
from typing import NamedTuple, ByteString, List
import numpy as np

class RewardMachine():
    def __init__(self, num_states, reward_functions):
        self.current_state = 0
        self.reward_functions = reward_functions    
        self.num_states = num_states
    
    def get_reward(self, *args):
        return self.reward_functions[self.current_state](*args)
    
    def advance(self):
        self.current_state += 1
    
    def retreat(self):
        self.current_state -= 1

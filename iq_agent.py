from sac_discrete import SAC_Discrete
import numpy as np
from rm import RewardMachine
from replay_buffer import ReplayBuffer
from gym_multigrid.multigrid import Agent

class IqAgent(Agent):
    def __init__(self, world,color, num_rm_states, replay_size, batch_size, sac_params):
        self.q_functions = []
        self.initialize_rm(num_rm_states)
        self.replay_buffers=[]
        self.batch_size = batch_size

        for _ in range(num_rm_states):
            self.q_functions.append(SAC_Discrete(sac_params)) # fix params
            self.replay_buffers.append(ReplayBuffer(capacity=replay_size, seed=0))

        super().__init__(world=world, color=color)
        
    def initialize_rm(self, num_rm_states):
        def u1_function(at_goal):
            return float(at_goal)
        def u2_function(at_goal):
            return float(at_goal)
        
        self.rm = RewardMachine(num_rm_states, [u1_function, u2_function])
    
    def learn(self, to_log=False):
        current_state = self.rm.current_state
        batch = self.replay_buffers[current_state].sample(self.batch_size)
        self.q_functions[current_state].learn(batch, to_log)
    
    def add_experience(self, s, a, r, n, d):
        self.replay_buffers[self.rm.current_state].push(s,a,r,n,d)

    def advance_rm(self):
        self.rm.advance()
    
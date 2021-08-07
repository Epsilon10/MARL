from sac_discrete import SAC_Discrete
import numpy as np
from rm import RewardMachine
from replay_buffer import ReplayBuffer
from gym_multigrid.multigrid import Agent

class IqAgent(Agent):
    def __init__(self, world,color, num_rm_states, replay_size, batch_size, sac_params):
        self.q_functions = []
        self.initialize_rm(num_rm_states)
        self.replay_buffer = ReplayBuffer(capacity=replay_size, seed=0)
        self.batch_size = batch_size

        for _ in range(num_rm_states):
            self.q_functions.append(SAC_Discrete(sac_params)) # fix params

        super().__init__(world=world, color=color)
    def initialize_rm(self, num_rm_states):
        def u1_function(at_goal):
            return float(at_goal)
        def u2_function(at_goal):
            return float(at_goal)
        
        self.rm = RewardMachine(num_rm_states, [u1_function, u2_function])
    
    def learn(self, to_log=False):
        batch = self.replay_buffer.sample(self.batch_size)
        self.q_functions[self.rm.current_state].learn(batch, to_log)
    
    def add_experience(self, s, a, r, n, d):
        self.replay_buffer.push(s,a,r,n,d)


class IqAgentList():
    def __init__(self, num_agents, num_rm_states, sac_params):
        self.agent_list = []
        for _ in range(num_agents):
            self.agent_list.append(IqAgent(num_rm_states, sac_params))
    
    def advance_agent_rm(self, agent_id):
        return self.agent_list[agent_id].rm.advance()
    
    def add_agent_experience(self, agent_id, s, a, r, n, d):
        self.agent_list[agent_id].add_experience(s,a,r,n,d)

    def learn(self):
        for agent in self.agent_list:
            agent.learn()
    
    def pop(self, agent_id):
        return self.agent_list.pop(agent_id)
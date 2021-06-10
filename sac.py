import torch

from replay_buffer import ReplayBuffer
from models import ActorCritic

from copy import deepcopy
import itertools

class SAC:
    def __init__(self, num_observations, actions, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.act_limit = args.act_limit

        self.ac = ActorCritic(num_observations, actions, self.act_limit)
        self.ac_target = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for param in self.ac_target.parameters():
            param.requires_grad = False
        
        self.q_parameters = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        replay_buffer = ReplayBuffer(num_observations, actions, args.replay_size)
        
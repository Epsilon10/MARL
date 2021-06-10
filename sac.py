import torch
from torch.optim import Adam

from replay_buffer import ReplayBuffer
from models import ActorCritic

from copy import deepcopy
import itertools

class SAC:
    def __init__(self, num_observations, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.act_limit = args.act_limit

        actions = action_space[0]

        self.ac = ActorCritic(num_observations, actions, self.act_limit)
        self.ac_target = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for param in self.ac_target.parameters():
            param.requires_grad = False
        
        self.q_parameters = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        replay_buffer = ReplayBuffer(num_observations, actions, args.replay_size)

        self.policy_optimizer = Adam(self.ac.policy.parameters(), lr=args.lr)
        self.q_optimizer = Adam(self.q_parameters, lr=args.lr)

        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        if self.automatic_entropy_tuning:
            self.target_entropy = torch.

    def compute_loss_q(self, state, action, reward, next_state, done):
        q1 = self.ac.q1(state, action)
        q2 = self.ac.q2(state, action)

        # Bellman equation
        with torch.no_grad():
            a2, logp_a2 = self.ac.pi(next_state)

            # Target q values
            q1_policy_target = self.ac_target.q1(next_state, a2)
            q2_policy_target = self.ac_target.q2(next_state, a2)
            
            q_policy_target = torch.min(q1_policy_target, q2_policy_target)

            # https://spinningup.openai.com/en/latest/algorithms/sac.html
            y = reward + self.gamma* (1-done) * (q_policy_target - self.alpha * logp_a2)

        # MSE error loss
        loss_q1 = ((q1 - y)**2).mean()
        loss_q2 = ((q2 - y)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

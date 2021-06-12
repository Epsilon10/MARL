import torch
from torch.optim import Adam
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from models import ActorCritic

from copy import deepcopy
import itertools

class SAC:
    def __init__(self, num_observations, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval

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
            # Target Entropy = −dim(A)
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)
        
        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        done_batch = torch.FloatTensor(done_batch)

        with torch.no_grad():
            next_action, next_state_log_pi = self.ac.policy.sample(next_state_batch)
            q1_target = self.ac_target.q1(next_state_batch, next_action)
            q2_target = self.ac_target.q2(next_state_batch, next_action)

            q_targ = torch.min(q1_target, q2_target)
            next_q_val = reward_batch + self.gamma * (1 - done_batch) * (q_targ - self.alpha * next_state_log_pi)

        q1 = self.ac.q1(state_batch, action_batch)
        q2 = self.ac.q2(state_batch, action_batch)

        q1_loss = F.mse_loss(q1, next_q_val)
        q2_loss = F.mse_loss(q2, next_q_val)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        pi, logp_pi = self.ac.policy.sample(state_batch)

        q1_pi = self.ac.q1(state_batch, pi)
        q2_pi = self.ac.q2(state_batch, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = ((self.alpha * logp_pi) - q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        # update networks with polyak averaging
        with torch.no_grad():
            for param, target_param in zip(self.ac.parameters(), self.ac_target.parameters()):
                target_param.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

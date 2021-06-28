from math import log
import torch
import numpy as np
from models import CategoricalPolicy, VisualQNetworkPair
from torch.optim import Adam
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import datetime


# based on https://github.com/ku2482/sac-discrete.pytorch/blob/40c9d246621e658750e0a03001325006da57f2d4/sacd/agent/sacd.py

def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()

class SAC_Discrete():
    def __init__(self, observation_shape, num_actions, hidden_dim, gamma=0.99, lr=.0003, automatic_entropy_tuning=True):
        self.critic = VisualQNetworkPair(observation_shape, num_actions, hidden_dim, use_conv=True)
        self.target_critic = VisualQNetworkPair(observation_shape, num_actions, hidden_dim, use_conv=True).eval()

        self.policy = CategoricalPolicy(observation_shape, num_actions, hidden_dim, True)
        
        self.target_critic.load_state_dict(self.critic.state_dict())

        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.q2.parameters(), lr=lr)

        self.target_entropy = -np.log(1.0 / num_actions) * 0.98

        # optimize log alpha instead of alpha
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self.gamma = gamma

        self.writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "GridWorld",
                                                            "Categorical", "autotune" if automatic_entropy_tuning else ""))

    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))

        return q1_loss, q2_loss, mean_q1, mean_q2

    def calc_policy_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        policy_loss = (-q - self.alpha * entropies).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies):
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies))
        return entropy_loss           
    
    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def learn(self, batch, to_log=False):
        q1_loss, q2_loss, mean_q1, mean_q2 = self.calc_critic_loss(batch)
        policy_loss, entropies = self.calc_policy_loss(batch)
        entropy_loss = self.calc_entropy_loss(entropies)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()            

        if to_log:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)
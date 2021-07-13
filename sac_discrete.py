from math import log, log10
import torch
import numpy as np
from torch.serialization import save
from models import CategoricalPolicy, VisualQNetworkPair
from torch.optim import Adam
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import datetime

import torch.nn as nn
import torch.nn.functional as F
import os
# based on https://arxiv.org/pdf/1910.07207.pdf

def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()

def soft_update_of_target_network(local_model, target_model, tau):
    """Updates the target network in the direction of the local network but by taking a step size
    less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class SAC_Discrete():
    def __init__(self, observation_shape, num_actions, hidden_dim, gamma=0.9, lr=3e-4, automatic_entropy_tuning=True, cuda=False, tau=0.005):
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.critic = VisualQNetworkPair(input_shape=observation_shape, num_actions=num_actions, hidden_dim=hidden_dim, use_conv=True).to(device=self.device)
        self.target_critic = VisualQNetworkPair(input_shape=observation_shape, num_actions=num_actions, hidden_dim=hidden_dim, use_conv=True).to(device=self.device).eval()

        self.policy = CategoricalPolicy(observation_shape, num_actions, hidden_dim, True).to(device=self.device)
        
        self.target_critic.load_state_dict(self.critic.state_dict())

        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.q2.parameters(), lr=lr)

        self.target_entropy  = -np.log(1.0 / num_actions) * .98

        # optimize log alpha instead of alpha

        #self.alpha = 0.078
        #self.log_alpha = np.log(self.alpha)

        # optimize log alpha instead of alpha
        GRIDWORLD_STARTING_ENTROPY = -2.659

        self.log_alpha = torch.tensor([-2.659], requires_grad=True) # good starting log alpha
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        
        self.gamma = gamma
        self.tau = tau

        self.writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "GridWorld",
                                                            "Categorical", "autotune" if automatic_entropy_tuning else ""))
        self.learning_steps = 0
    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def explore(self, states):
        with torch.no_grad():
            actions, _, _ = self.policy.sample(states)
        return actions
    
    def act(self, states):
        with torch.no_grad():
            actions = self.policy.act(states)
        return actions

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            min_qf_next_target = action_probs * (torch.min(next_q1, next_q2) - self.alpha * log_action_probs)
            
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            target_q = rewards + (1.0 - dones.byte()) * self.gamma * (min_qf_next_target)
        return target_q

    def calc_critic_loss(self, batch):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        return q1_loss, q2_loss, mean_q1, mean_q2

    def calc_policy_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            q1, q2 = self.critic(states)
            min_q = torch.min(q1, q2)

        inside_term = self.alpha * log_action_probs - min_q
        policy_loss = (action_probs * inside_term).sum(dim=1).mean()
        log_action_probs = torch.sum(log_action_probs * action_probs, dim=1)
        return policy_loss, log_action_probs

    def calc_entropy_loss(self, log_probs):
        entropy_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        return entropy_loss
    
    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def learn(self, batch, to_log=False):
        self.learning_steps +=1
        q1_loss, q2_loss, mean_q1, mean_q2 = self.calc_critic_loss(batch)
        policy_loss, log_action_probs = self.calc_policy_loss(batch)
        entropy_loss = self.calc_entropy_loss(log_action_probs)

        
        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        soft_update_of_target_network(self.critic, self.target_critic, self.tau)
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
    
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))

        

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class QNetwork(nn.Module):
    def __init__(self, num_observations, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 network
        self.q1_linear1 = nn.Linear(num_observations+num_actions, hidden_dim)
        self.q1_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_linear3 = nn.Linear(hidden_dim, 1)

        #Q2 network
        self.q2_linear1 = nn.Linear(num_observations+num_actions, hidden_dim)
        self.q2_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        input = torch.cat([state, action], 1)

        x1 = F.relu(self.q1_linear1(input))
        x1 = F.relu(self.q1_linear2O(x1))
        x1 = self.q1_linear3(x1)

        x2 = F.relu(self.q2_linear1(input))
        x2 = F.relu(self.q2_linear2(x2))
        x2 = self.q2_linear3(x2)

        return x1, x2


class GuassianPolicy(torch.nn.Module):
    def __init__(self, num_observations, num_actions, hidden_dim, act_limit):
        super().__init__()

        # 3 layer network, with 2 outputs

        self.layer1 = nn.Linear(num_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, num_actions)
        self.log_stddev_layer = nn.Linear(hidden_dim, num_actions)

        self.act_limit = act_limit

        # apply initial weights
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_layer(x)
        log_stddev = self.log_stddev_layer(x)

        return mean, log_stddev

    def sample(self, state):
        mean, log_stddev = self.forward(state)
        stddev = log_stddev.exp()
        normal = Normal(mean, stddev)
        action = torch.tanh(normal.rsample())

        log_prob = normal.log_prob(action).sum(axis=1)
        log_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)


        return action * self.act_limit, log_prob

class ActorCritic(nn.Module):
    def __init__(self, num_observations, num_actions, act_limit, hidden_dim=(256,256)):
        super().__init__()

        self.policy = GuassianPolicy(num_observations, num_actions, hidden_dim, act_limit)
        self.q1 = QNetwork(num_observations, num_actions, hidden_dim)
        self.q2 = QNetwork(num_observations, num_actions, hidden_dim)
    
    def act(self, observation):
        with torch.nograd():
            action, _ = self.policy(observation)
            return action.numpy()
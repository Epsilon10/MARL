import torch
import torch.nn as nn
import torch.nn.functional as F

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

        x1 = F.relu()




class GuassianPolicy(torch.nn.Module):
    def __init__(self, num_observations, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_observations, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.cov_linear = nn.Linear(hidden_dim, num_actions)
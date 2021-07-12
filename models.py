import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from math import floor
from typing import Tuple

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class FullCNN(BaseNetwork):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(initialize_weights_he)
    
    def forward(self, states):
        return self.net(states.permute(0,3,1,2))
    
    @staticmethod
    def conv_output_shape(
        h_w: Tuple[int, int],
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
    ):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        w = floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        return h, w

class VisualQNetwork(BaseNetwork):
    def __init__(self, input_shape, num_actions, hidden_dim, use_conv=False):
        super().__init__()

        self.use_conv = use_conv

        height = input_shape[0]
        width = input_shape[1]
        in_channels = input_shape[2]

        conv1_hw = FullCNN.conv_output_shape((height, width), 8, 4)
        conv2_hw = FullCNN.conv_output_shape(conv1_hw, 4, 2)
        conv3_hw = FullCNN.conv_output_shape(conv2_hw, 3, 1)

        if self.use_conv:
            self.conv = FullCNN(in_channels)       

        self.net = nn.Sequential(
            nn.Linear(conv3_hw[0]*conv3_hw[1]*64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions)
        ) 
    
    def forward(self, states):
        if self.use_conv:
            states = self.conv(states)
        
        return self.net(states)

class VisualQNetworkPair(BaseNetwork):
    def __init__(self, input_shape, num_actions, hidden_dim, use_conv=False):
        super().__init__()

        self.q1 = VisualQNetwork(input_shape, num_actions, hidden_dim, use_conv)
        self.q2 = VisualQNetwork(input_shape, num_actions, hidden_dim, use_conv)

    def forward(self, states):
        q1 = self.q1(states)
        q2 = self.q2(states)

        return q1, q2

class CategoricalPolicy(BaseNetwork):
    def __init__(self, input_shape, num_actions, hidden_dim, use_conv=False):
        super().__init__()

        in_channels = input_shape[2]
        height = input_shape[0]
        width = input_shape[1]

        self.use_conv = use_conv

        if self.use_conv:
            self.conv = FullCNN(in_channels)

        conv1_hw = FullCNN.conv_output_shape((height, width), 8, 4)
        conv2_hw = FullCNN.conv_output_shape(conv1_hw, 4, 2)
        conv3_hw = FullCNN.conv_output_shape(conv2_hw, 3, 1)
        
        self.net = nn.Sequential(
            nn.Linear(conv3_hw[0]*conv3_hw[1]*64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def sample(self, states, epsilon=0.1):
        if self.use_conv:
            states = self.conv(states)
        out = self.net(states)
        action_probs = F.softmax(out,dim=1)
        action_distro = Categorical(action_probs)
        actions = action_distro.sample().view(-1,1)

        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
    
    def act(self, states, epsilon=0.1):
        if self.use_conv:
            states = self.conv(states)
        action_logits = self.net(states)
        action_logits += epsilon * torch.rand(action_logits.shape[0], action_logits.shape[1])
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions



class GuassianPolicy(torch.nn.Module):
    def __init__(self, num_observations, num_actions, hidden_size, act_limit):
        super().__init__()

        # 3 layer network, with 2 outputs

        self.linear1 = nn.Linear(num_observations, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_layer = nn.Linear(hidden_size, num_actions)
        self.log_stddev_layer = nn.Linear(hidden_size, num_actions)

        self.act_limit = act_limit

        # apply initial weights
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_layer(x)
        log_stddev = self.log_stddev_layer(x)
        log_stddev = torch.clamp(log_stddev, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_stddev

    def sample(self, state):
        mean, log_stddev = self.forward(state)
        stddev = log_stddev.exp()
        normal = Normal(mean, stddev)

        print(mean, stddev)

        sample = normal.rsample()
        log_prob = normal.log_prob(sample).sum(axis=-1)
        #log_prob -= (2*(np.log(2) - sample - F.softplus(-2*sample))).sum(axis=0)
        
        action = torch.tanh(sample) * self.act_limit
        return action, log_prob

"""
class ActorCritic(nn.Module):
    def __init__(self, num_observations, num_actions, hidden_dim, act_limit=1):
        super().__init__()

        self.policy = GuassianPolicy(num_observations, num_actions, hidden_dim, act_limit)
        self.q1 = QNetwork(num_observations, num_actions, hidden_dim)
        self.q2 = QNetwork(num_observations, num_actions, hidden_dim)
    
    def act(self, state):
        with torch.nograd():
            action, _ = self.policy.sample(state)
            return action.numpy()
"""

import torch

class QNetwork(torch.nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

class GuassianPolicy(torch.nn.Module):
    def __init__(self, num_observations, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = torch.nn.Linear(num_observations, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = torch.nn.Linear(hidden_dim, num_actions)
        self.cov_linear = torch.nn.Linear(hidden_dim, num_actions)
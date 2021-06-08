import torch
from models import QNetwork

class SAC:
    def __init__(self, observations, actions, args):
        """
        observations: list of observations from environment
        actions: list of possible actions for agent
        args: gamma (discount factor), alpha (temperature coeff)
        """

        self.gamma = args.gamma
        self.alpha = args.alpha

        self.critic = QNetwork()
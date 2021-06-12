from mlagents_envs.environment import ActionTuple, BaseEnv
from sac import SAC

class Trainer:
    @staticmethod
    def generate_trajectories(env, buffer_size):
        agent = SAC()
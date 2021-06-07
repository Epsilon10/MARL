from mlagents_envs.environment import ActionTuple, BaseEnv

class Trainer:
    @staticmethod
    def generate_trajectories(
        env: BaseEnv
    ):
        buffer: Buffer = []
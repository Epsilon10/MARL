from mlagents_envs.registry import default_registry
from sac import SAC

env = default_registry["GridWorld"].make()
env.reset()

BEHAVIOR_NAME = list(env.behavior_specs)[0]

behavior_spec = env.behavior_specs[BEHAVIOR_NAME]
observation_specs = behavior_spec.observation_specs
action_spec = behavior_spec.action_spec

decision_steps, terminal_steps = env.get_steps(BEHAVIOR_NAME)

agent = SAC(observation_specs[0].shape[0], action_spec.discrete_branches, args)

env.close()
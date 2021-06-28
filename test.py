from mlagents_envs.base_env import ActionTuple
from mlagents_envs.registry import default_registry
from sac_discrete import SAC_Discrete
import torch

env = default_registry["GridWorld"].make()
env.reset()

BEHAVIOR_NAME = list(env.behavior_specs)[0]

behavior_spec = env.behavior_specs[BEHAVIOR_NAME]
observation_specs = behavior_spec.observation_specs
action_spec = behavior_spec.action_spec

decision_steps, terminal_steps = env.get_steps(BEHAVIOR_NAME)

print(observation_specs[0].shape, action_spec.discrete_branches[0])

agent = SAC_Discrete(observation_specs[0].shape, action_spec.discrete_branches[0], 512, .0003, .99)

print(decision_steps[0].obs[0].shape)
state = torch.from_numpy(decision_steps.obs[0])

a,b,c = agent.policy.sample(state)

at = ActionTuple()
at.add_discrete(a.numpy())
env.set_actions(BEHAVIOR_NAME, at)

print(a,b,c)
env.close()
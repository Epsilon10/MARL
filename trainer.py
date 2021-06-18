import argparse
import itertools

from mlagents_envs.environment import ActionTuple, BaseEnv
from mlagents_envs.registry import default_registry
from sac import SAC
from replay_buffer import ReplayBuffer

from typing import Dict
import random

import numpy as np

parser = argparse.ArgumentParser(description='SAC Unity')


args = parser.parse_args()

buffer = ReplayBuffer(args.capacity, args.seed)

env = default_registry["GridWorld"].make()
env.reset()

# train many agents at one time

total_numsteps = 0

NUM_AGENTS = 9
BEHAVIOR_NAME = list(env.behavior_specs)[0]

behavior_spec = env.behavior_specs[BEHAVIOR_NAME]
observation_specs = behavior_spec.observation_specs
action_spec = behavior_spec.action_spec

num_observations = observation_specs[0].shape[0]

action_spec = action_spec.discrete_branches if action_spec.is_discrete() else (action_spec.continuous_size)

agent = SAC(observation_specs[0].shape[0], action_spec.discrete_branches, args)

replay_buffer = ReplayBuffer(args.replay_size, args.seed)

total_numsteps = 0
updates = 0

def get_actions_from_policy(state):
    actions = np.empty((NUM_AGENTS, action_spec.discrete_size))
    for agent_i in NUM_AGENTS:
        for j in range(action_spec.discrete_size):
            actions[agent_i][j] = agent.get_action(state)
    return actions

def update(states, done, decision_steps, terminal_steps):
    for agent_id in decision_steps:
        states[agent_id] = decision_steps[agent_id].obs[0]
        done[agent_id][0] = False
    for agent_id in terminal_steps:
        states[agent_id] = terminal_steps[agent_id].obs[0]
        done[agent_id][0] = not terminal_steps[agent_id].interrupted
    
for episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    env.reset()
    
    states = np.empty((NUM_AGENTS, num_observations))
    done = np.zeros((NUM_AGENTS, 1), dtype=bool)

    while len(buffer) < args.buffer_size:
        decision_steps, terminal_steps = env.get_steps(BEHAVIOR_NAME)

        update(states, done, decision_steps, terminal_steps)

        if args.start_steps > total_numsteps:
            actions = action_spec.random_action(NUM_AGENTS)
        else:
            actions = get_actions_from_policy(states)

        for i in range(args.updates_per_step):
            agent.update_parameters(replay_buffer, args.replay_size, updates)
            updates +=1

        env.step()
        
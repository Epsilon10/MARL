import argparse
import itertools
import torch

from mlagents_envs.environment import ActionTuple, BaseEnv
from mlagents_envs.registry import default_registry
from sac import SAC
from replay_buffer import ReplayBuffer

from typing import Dict
import random

import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='SAC Unity')
parser.add_argument('--env-name', default="GridWorld", help='Unity environment (defaut: GridWorld)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')

args = parser.parse_args()

writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             "Gaussian", "autotune" if args.automatic_entropy_tuning else ""))

buffer = ReplayBuffer(args.replay_size, args.seed)

env = default_registry["GridWorld"].make()
env.reset()

# train many agents at one time

total_numsteps = 0

NUM_AGENTS = 9
BEHAVIOR_NAME = list(env.behavior_specs)[0]

behavior_spec = env.behavior_specs[BEHAVIOR_NAME]
observation_specs = behavior_spec.observation_specs
action_spec = behavior_spec.action_spec

num_observations = 64 * 84 * 3

agent = SAC(num_observations, action_spec.discrete_branches, args)

replay_buffer = ReplayBuffer(args.replay_size, args.seed)

total_numsteps = 0
updates = 0

def get_actions_from_policy(state):
    actions = np.empty((NUM_AGENTS, action_spec.discrete_size))
    for agent_i in NUM_AGENTS:
        for j in range(action_spec.discrete_size):
            actions[agent_i][j] = agent.get_action(state)
    return actions

def get_states_and_termination():
    states = np.empty((NUM_AGENTS, num_observations))
    done = np.empty(NUM_AGENTS)
    rewards = np.empty(NUM_AGENTS)

    decision_steps, terminal_steps = env.get_steps(BEHAVIOR_NAME)

    for agent_id in decision_steps:
        states[agent_id] = decision_steps[agent_id].obs[0].flatten()
        done[agent_id] = False
        rewards[agent_id] = decision_steps[agent_id].reward

    for agent_id in terminal_steps:
        states[agent_id] = terminal_steps[agent_id].obs[0].flatten()
        done[agent_id] = not terminal_steps[agent_id].interrupted
        rewards[agent_id] = terminal_steps[agent_id].reward 

    return states, rewards, done

states, rewards, done = get_states_and_termination()

action, log_prob = agent.ac.policy.sample(torch.from_numpy(states[0]).float())
print(action)

env.close()

"""
for episode in itertools.count(1):
    episode_rewards = np.zeroes((1, NUM_AGENTS))
    episode_steps = 0
    env.reset()

    states, done = get_states_and_termination()

    while len(buffer) < args.buffer_size:
        if args.start_steps > total_numsteps:
            actions = action_spec.random_action(NUM_AGENTS) # sample random action
        else:
            actions = get_actions_from_policy(states) # sample action from policy

        for i in range(args.updates_per_step):
            q1_loss, q2_loss, policy_loss, entropy_loss, alpha = agent.update_parameters(replay_buffer, args.replay_size, updates)

            writer.add_scalar("loss/q1", q1_loss, updates)
            writer.add_scalar("loss/q2", q2_loss, updates)
            writer.add_scalar("loss/policy", policy_loss, updates)
            writer.add_scalar("loss/entropy_loss", entropy_loss, updates)
            writer.add_scalar("entropy_temperature/alpha", alpha, updates)
            updates +=1

        env.step()

        next_states, rewards, done = get_states_and_termination()

        episode_steps += 1
        total_numsteps += 1 

        episode_rewards += rewards
        # episode_reward += 1

        for agent_id in range(NUM_AGENTS):
            state = states[agent_id]
            reward = rewards[agent_id]
            action = actions[agent_id]
            next_state = next_states[agent_id]
            replay_buffer.push(state, action, reward, next_state, done[agent_id])
    
    if total_numsteps > args.num_steps:
        break

env.close()
"""
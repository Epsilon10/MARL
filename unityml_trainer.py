import argparse
import itertools
from math import log
import torch

from mlagents_envs.environment import ActionTuple

from sac_discrete import SAC_Discrete
from replay_buffer import ReplayBuffer

from typing import Dict
import random

import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import NamedTuple, List


# A Trajectory is an ordered sequence of Experiences

class UnityMLTrainer():
    # Discrete action spaces only right now, will impl continuous in futre
    def __init__(self, env, num_steps=100000, batch_size=64,
                 replay_size=1000, gamma=0.99,
                 target_entropy_ratio=0.98, start_steps=200,
                 update_interval=4, target_update_interval=8000,
                 num_eval_steps=125000, max_episode_steps=27000,
                 log_interval=10, eval_interval=1000, cuda=True, seed=0, num_agents=9, automatic_entropy_tuning=True):
        
        self.env = env
        self.env.reset()
        
        self.log_interval = log
        self.target_update_interval = target_update_interval
        self.update_interval = update_interval

        self.behavior_name = list(self.env.behavior_specs)[0]
        behavior_spec = self.env.behavior_specs[self.behavior_name]
        
        self.action_spec = behavior_spec.action_spec
        self.observation_specs = behavior_spec.observation_specs
    
        self.agent = SAC_Discrete(self.observation_specs[0].shape, self.action_spec.discrete_branches[0], 512, .0003, .99)
        self.replay_buffer = ReplayBuffer(replay_size, seed)
        self.batch_size = batch_size

        self.last_observations = {}
        self.dones = {}
        self.last_actions = {}
        self.cum_rewards = {}
        self.trajectories = {}

        self.start_steps = start_steps
        self.steps = 0

        self.replay_size = replay_size

    def train_episode(self):
        while len(self.replay_buffer) < self.replay_size:
            print("GOING")
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

            for agent_id in terminal_steps:            
                self.replay_buffer.push(
                    state=self.last_observations[agent_id],
                    action=self.last_actions[agent_id],
                    reward=terminal_steps[agent_id].reward,
                    next_state=terminal_steps[agent_id].obs[0],
                    done=not terminal_steps[agent_id].interrupted
                )

                self.last_observations.pop(agent_id)
                self.last_actions.pop(agent_id)
            
            num_active_agents = len(decision_steps)

            if self.start_steps > self.steps:
                actions = self.action_spec.random_action(num_active_agents).discrete
            else:
                actions, _, _ = self.agent.policy.sample(decision_steps.obs[0])
                actions = actions.numpy()

            for action, agent_id in zip(actions, decision_steps):
                if agent_id in self.last_observations:
                    self.replay_buffer.push(
                        state=self.last_observations[agent_id] ,
                        action=self.last_actions[agent_id],
                        reward=decision_steps[agent_id].reward,
                        next_state=decision_steps[agent_id].obs[0],
                        done=False
                    )
                self.last_observations[agent_id] = decision_steps[agent_id].obs[0]
                self.last_actions[agent_id] = action.item()
            
            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions)
            self.env.set_actions(self.behavior_name, action_tuple)

            self.env.step()

            self.steps += 1
            
            if self.steps % self.update_interval == 0 and self.steps >= self.start_steps:
                batch = self.replay_buffer.sample(self.batch_size)
                to_log = self.steps % self.log_interval == 0
                self.agent.learn(batch, to_log)
            
            if self.steps % self.target_update_interval:
                self.agent.update_target()
    

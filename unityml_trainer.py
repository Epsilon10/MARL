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
from torch import tensor

# A Trajectory is an ordered sequence of Experiences

class UnityMLTrainer():
    # Discrete action spaces only right now, will impl continuous in futre
    def __init__(self, env, num_steps=10000, batch_size=64,
                 replay_size=100000, gamma=0.99,
                 target_entropy_ratio=0.98, start_steps=1000,
                 update_interval=4, target_update_interval=800,
                 num_eval_steps=12500, max_episode_steps=2700,
                 log_interval=10, eval_interval=100, cuda=True, seed=0, num_agents=9, automatic_entropy_tuning=True):
        
        self.env = env
        self.env.reset()
        
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.update_interval = update_interval
        self.max_episode_steps = max_episode_steps
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.num_steps = num_steps
        self.episodes = 0

        self.behavior_name = list(self.env.behavior_specs)[0]
        behavior_spec = self.env.behavior_specs[self.behavior_name]
        
        self.action_spec = behavior_spec.action_spec
        self.observation_specs = behavior_spec.observation_specs
    
        self.agent = SAC_Discrete(self.observation_specs[0].shape, self.action_spec.discrete_branches[0], 512)
        self.replay_buffer = ReplayBuffer(replay_size, seed)
        self.batch_size = batch_size

        self.last_observations = {}
        self.dones = {}
        self.last_actions = {}

        self.start_steps = start_steps
        self.steps = 0
    
    def set_actions_for_agents(self, actions):
        action_tuple = ActionTuple()
        action_tuple.add_discrete(actions)
        self.env.set_actions(self.behavior_name, action_tuple)

    def train_episode(self):
        all_done = False
        episode_steps = 0
        self.episodes += 1 
        self.env.reset()
        print("EPISODE", self.episodes)
        while not all_done and episode_steps <= self.max_episode_steps:
            print("STEPS:", self.steps)
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

            all_done = len(decision_steps) == 0
            print("NUM NOT DONE", len(decision_steps))

            for agent_id in terminal_steps:            
                self.replay_buffer.push(
                    state=self.last_observations[agent_id].copy(),
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
                actions = self.agent.explore(torch.from_numpy(decision_steps.obs[0]))
                actions = actions.numpy()

            for action, agent_id in zip(actions, decision_steps):
                if agent_id in self.last_observations:
                    self.replay_buffer.push(
                        state=self.last_observations[agent_id].copy(),
                        action=self.last_actions[agent_id],
                        reward=decision_steps[agent_id].reward,
                        next_state=decision_steps[agent_id].obs[0],
                        done=False
                    )

                self.last_observations[agent_id] = decision_steps[agent_id].obs[0]
                self.last_actions[agent_id] = action.item()
            
            self.set_actions_for_agents(actions)

            self.env.step()

            self.steps += 1
            episode_steps += 1
            
            if self.steps % self.update_interval == 0 and self.steps >= self.start_steps:
                print("LEARN")
                batch = self.replay_buffer.sample(self.batch_size)
                to_log = self.steps % self.log_interval == 0
                self.agent.learn(batch, to_log)
            
            if self.steps % self.target_update_interval == 0:
                print("UPDATE TARGET")
                self.agent.update_target()
            
            if self.steps % self.eval_interval == 0 and False:
                print("EVAL")
                self.evaluate()
                self.agent.save_models(save_dir="models/")
                
    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0
        for i in range(self.num_eval_steps):
            self.env.reset()
            episode_return = 0.0
            all_done = False

            for n in range(self.max_episode_steps):
                decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
                actions = self.agent.act(torch.from_numpy(decision_steps.obs[0])).numpy()
                self.set_actions_for_agents(actions)
                print("ACTIONS", actions)
                self.env.step()
                num_steps += 1
                episode_return += decision_steps.reward.mean()
                print(f"EP: {i}, STEPS: {n}, NUM FINISHED: {9 - len(decision_steps)}")
                print("EP return", episode_return)
            
            num_episodes += 1
            total_return += episode_return

            
    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break




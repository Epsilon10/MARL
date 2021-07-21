import argparse
import itertools
from math import log
import torch

from mlagents_envs.environment import ActionTuple

from sac_discrete import SAC_Discrete, update_params
from replay_buffer import ReplayBuffer

from typing import Dict
import random

import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import NamedTuple, List
from torch import tensor

# A Trajectory is an ordered sequence of Experiences

class GridWorldTrainData():
    def __init__(self, eval):
        self.eval = eval
        if not eval:
            self.last_observations = {}
            self.last_actions = {}
        else:
            self.cum_rew_agent = {}
            self.cum_rew = []
    
    def clear(self):
        if not self.eval:
            self.last_observations.clear()
            self.last_actions.clear()
        else:
            self.cum_rew_agent.clear()
            self.cum_rew.clear()
    
class UnityMLTrainer():
    # Discrete action spaces only right now, will impl continuous in futre
    def __init__(self, env,num_steps=100000, batch_size=64,
                 lr=3e-4, replay_size=60000, gamma=0.9, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=400,
                 update_interval=2, target_update_interval=800,
                 use_per=False, dueling_net=False, num_eval_steps=12000,
                 max_episode_steps=2700, log_interval=5, eval_interval=1000,
                 cuda=True, seed=0, num_agents=9, automatic_entropy_tuning=True):
        
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
        self.num_agents = num_agents
        self.behavior_name = list(self.env.behavior_specs)[0]
        behavior_spec = self.env.behavior_specs[self.behavior_name]
        self.best_eval_score = -np.inf
        self.action_spec = behavior_spec.action_spec
        self.observation_specs = behavior_spec.observation_specs
    
        self.agent = SAC_Discrete(self.observation_specs[0].shape, self.action_spec.discrete_branches[0], hidden_dim=512, starting_entropy=-2.659, gamma=gamma, lr=lr)
        self.replay_buffer = ReplayBuffer(replay_size, 123456)
        self.batch_size = batch_size

        self.start_steps = start_steps
        self.steps = 0
        self.replay_size = replay_size
    
    
    def set_actions_for_agents(self, actions):
        action_tuple = ActionTuple()
        action_tuple.add_discrete(actions)
        self.env.set_actions(self.behavior_name, action_tuple)

    def execute(self, train_data, eval):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        for agent_id in terminal_steps:   
            if not eval:         
                self.replay_buffer.push(
                    state=train_data.last_observations[agent_id].copy(),
                    action=train_data.last_actions[agent_id],
                    reward=terminal_steps[agent_id].reward,
                    next_state=terminal_steps[agent_id].obs[0],
                    done=not terminal_steps[agent_id].interrupted
                )

                train_data.last_observations.pop(agent_id)
                train_data.last_actions.pop(agent_id)
            else:
                train_data.cum_rew.append(train_data.cum_rew_agent.pop(agent_id) + terminal_steps[agent_id].reward)
                
        num_active_agents = len(decision_steps)

        if eval:
            actions = self.agent.act(torch.from_numpy(decision_steps.obs[0])).numpy()
        elif self.start_steps > self.steps:
            actions = self.action_spec.random_action(num_active_agents).discrete
        else:
            actions = self.agent.explore(torch.from_numpy(decision_steps.obs[0])).numpy()

        for action, agent_id in zip(actions, decision_steps):
            if not eval:
                if agent_id in train_data.last_observations:
                    self.replay_buffer.push(
                        state=train_data.last_observations[agent_id].copy(),
                        action=train_data.last_actions[agent_id],
                        reward=decision_steps[agent_id].reward,
                        next_state=decision_steps[agent_id].obs[0],
                        done=False
                    )
                train_data.last_observations[agent_id] = decision_steps[agent_id].obs[0]
                train_data.last_actions[agent_id] = action.item()
            else:
                if agent_id not in train_data.cum_rew_agent:
                    train_data.cum_rew_agent[agent_id] = 0

                train_data.cum_rew_agent[agent_id] += decision_steps[agent_id].reward

        self.set_actions_for_agents(actions)
        self.env.step()
            
            
    def train_episode(self):
        episode_steps = 0
        self.episodes += 1 
        self.env.reset()

        train_data = GridWorldTrainData(eval=False)

        while episode_steps < self.max_episode_steps:
            if len(self.replay_buffer) < self.replay_size:
                self.execute(train_data=train_data, eval=False)
            self.steps += 1

            if self.steps % self.update_interval and self.steps > self.start_steps:
                batch = self.replay_buffer.sample(self.batch_size)
                to_log = self.steps % self.log_interval == 0
                self.agent.learn(batch, to_log)
            
            episode_steps += 1
            
        self.evaluate()
    
    def close(self):
        self.env.close()
    
    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0
        
        while True:
            self.env.reset()
            if num_steps > self.num_eval_steps:
                break
            eval_train_data = GridWorldTrainData(eval=True)
            epsisode_steps = 0

            while epsisode_steps < self.max_episode_steps:
                self.execute(train_data=eval_train_data, eval=True)
                num_steps += 1
                epsisode_steps += 1
            
            num_episodes += 1
            total_return += np.mean(eval_train_data.cum_rew)
            print(f"EP: {num_episodes}, REW: {np.mean(eval_train_data.cum_rew)}")

        mean_return = total_return / num_episodes
        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.agent.save_models(save_dir='models/best')
        self.agent.writer.add_scalar('reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break




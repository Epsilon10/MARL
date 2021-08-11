import gym
from gym.envs.registration import register
import time
import itertools

from torch.cuda.profiler import start
from marl import IqAgentList

register(
    id='mutltiagent-gw-v0',
    entry_point='gym_envs.envs:MAGW_Env10x15'
)

env = gym.make('mutltiagent-gw-v0')

total_numsteps = 0
updates = 0
start_steps = 20000
replay_size = 100000
max_episode_steps = 200000
NUM_AGENTS = 2

iq_agents = env.agents

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    states = env.reset()

    while not done:
        if start_steps > total_numsteps:
            actions = [env.action_space.sample() for _ in range(NUM_AGENTS)]
        else:
            actions = iq_agents.get_actions(states)
        
        for agent in iq_agents:
            if agent.buf_at_capacity():
                agent.learn()
        updates += 1
        next_states, rewards, done, _ = env.step(actions)
        episode_steps += 1
        total_numsteps += 1
        #episode_reward += reward

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        for i, agent in enumerate(iq_agents):
            agent.add_experience(states[i],actions[i], rewards[i],next_states[i], done)
        

import gym
from gym.envs.registration import register
import time

register(
    id='mutltiagent-gw-v0',
    entry_point='gym_envs.envs:MAGW_Env10x15'
)

env = gym.make('mutltiagent-gw-v0')

_ = env.reset()

nb_agents = len(env.agents)

while True:
    env.render(mode='human', highlight=True)
    time.sleep(.1)

    ac = [env.action_space.sample() for _ in range(nb_agents)]

    obs, _, done, _ = env.step(ac)
    print("OBS SHAPE", obs[0].shape)
    if done:
        break
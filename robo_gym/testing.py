import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
import numpy as np

target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

# initialize environment
env = gym.make('NoObstacleNavigationQueenieSim-v0', ip=target_machine_ip, gui=True)
env = ExceptionHandling(env)

num_episodes = 100

for episode in range(num_episodes):
    done = False
    env.reset()
    while not done:
        # random step in the environment
        action = np.zeros(2)
        action[0] = 1
        action[1] = 0
        # state, reward, done, info = env.step(env.action_space.sample())
        state, reward, done, info = env.step(action)
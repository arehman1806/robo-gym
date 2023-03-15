# import gym
# import robo_gym
# from robo_gym.wrappers.exception_handling import ExceptionHandling
# import numpy as np

# target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

# # initialize environment
# env = gym.make('GraspQueenieSim-v2', ip=target_machine_ip, gui=False)
# env = ExceptionHandling(env)

# num_episodes = 10000

# for episode in range(num_episodes):
#     done = False
#     env.reset()
#     while not done:
#         # random step in the environment
#         # action = np.zeros(2)
#         # action[0] = 1
#         # action[1] = 0
#         state, reward, done, info = env.step(env.action_space.sample())
#         # state, reward, done, info = env.step(action)







import gym
import robo_gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CheckpointCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from explore import ExploreObject

target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'
checkpoint_callback = CheckpointCallback(
  save_freq=40000,
  save_path="./logs/grasp/",
  name_prefix="queenie_grasp_varying_handle_2",
  save_replay_buffer=False,
  save_vecnormalize=False,
)

# model = SAC.load("./logs/grasp/queenie_grasp_varying_handle_40000_steps.zip")
env = gym.make('GraspQueenieSim-v2', ip=target_machine_ip, gui=False)
# model.set_env(env)
model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="./queenie_grasp_varying_handle_2/", buffer_size=200000)

try:
  model.learn(total_timesteps=500_000, progress_bar=True, callback=checkpoint_callback, tb_log_name="SAC")
except:
  model.save("./trained_queenie_grasp_varying_handles_2")
  model.save_replay_buffer("./trained_queenie_grasp_varying_handles_rb_2")
print("model has learned")

model.save("./trained_queenie_grasp_varying_handles_2")
model.save_replay_buffer("./trained_queenie_grasp_varying_handles_rb_2")
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render()
#     # VecEnv resets automatically
#     # if done:
#     #   obs = env.reset()

env.close()

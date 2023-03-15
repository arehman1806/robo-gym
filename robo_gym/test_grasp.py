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

model = SAC.load("./logs/grasp/queenie_grasp_varying_handle_200000_steps.zip")
env = gym.make('GraspQueenieSim-v2', ip=target_machine_ip, gui=False)
# model.set_env(env)
# model.learn(total_timesteps=100_000, progress_bar=True, callback=checkpoint_callback)
# model.save("./trained_queenie_grasp_small_focal_finetuned/")
# model.save_replay_buffer("./grasp_finetuned_rb")


# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=15, deterministic=True)
obs = env.reset()
done = False
while True:
    action = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action[0])
    if done:
        print(info)
        obs = env.reset()
        # obs, reward, done, info = env.step(np.array([-999, 0]))
        done = False
       
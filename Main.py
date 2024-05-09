##########################################
### To run this file: & PYTHONPATH PATH
##########################################
### Libraries
##########################################

from Env import BenchmarkReplication
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np
import random
import os

##########################################
### Train/load model
##########################################

Benv = BenchmarkReplication(W = 1, N = 10, Nsim = 100, Dynamics = 'BS', start_time = 0, T = 26, dT = 1, r = 0, mu = [0.03,0.01], sigma = [0.3, 0.2])
#check_env(Benv) #check environment is accepted by SB3
Benv.seed(seed=random.seed(10))

steps = 100000

try:
    path = f"C:/Users/yoshi/OneDrive/Desktop/Research/Benchmark_RL/BS_PPO_{str(steps)}.zip"
    model = PPO.load(path, env = DummyVecEnv([lambda: Benv]), print_system_info=True)
    print("Existing model loaded")
except:
    print("Training model...")
    model = PPO('MlpPolicy', DummyVecEnv([lambda: Benv]), learning_rate=0.001, verbose=1)
    model.learn(total_timesteps=steps)
    path = f"C:/Users/yoshi/OneDrive/Desktop/Research/Benchmark_RL/BS_PPO_{str(steps)}"
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(path)

##########################################
### Experiment
##########################################

Nepisodes = 1000
rew = []

vec_env = model.get_env()
for i in range(Nepisodes):
    obs = vec_env.reset()
    cont = True
    i = 0
    while cont:
        action = model.predict(obs)
        obs, reward, terminated, truncated = vec_env.step(action[0])
        i += 1
        if any([terminated,truncated]):
            cont = False
            rew.append(reward)

print(np.mean(rew))




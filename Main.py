##########################################
### To run this file: & PYTHONPATH PATH
### PYTHONPATH is the path to python.exe
### in the BencEnv conda environment
### PATH is this python file's path
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

W = 1
N = 10
Nsim = 100
Dynamics  = 'BS'
star_time = 0
T = 26
dT = 1
r = 0
mu = [0.03,0.01]
sigma = [0.03,0.01]

Benv = BenchmarkReplication(W = W, N = N, Nsim = Nsim,
                             Dynamics = Dynamics, start_time = star_time,
                               T = T, dT = dT, r = r, mu = mu, sigma = sigma)
Benv.seed(seed=random.seed(10))

steps = 200000

path_folder = f"C:/Users/yoshi/OneDrive/Desktop/Research/Benchmark_RL/BS_PPO" # PATH to the BS_PPO_Models folder
path = f"{path_folder}/BS_PPO_{str(steps)}_{str(int(sigma[0]*100))}{str(int(sigma[1]*100))}"
try:
    model = PPO.load(path, env = DummyVecEnv([lambda: Benv]), print_system_info=True)
except:
    print("Training model...")
    model = PPO('MlpPolicy', DummyVecEnv([lambda: Benv]), learning_rate=0.001, verbose=1)
    model.learn(total_timesteps=steps)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    model.save(f"{path}.zip")

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




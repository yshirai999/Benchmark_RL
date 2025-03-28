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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from Loggers import TensorboardCallback
import numpy as np
import random
import os
import matplotlib.pyplot as plt

##########################################
### Train/load model
##########################################

N = 10
Nsim = 100
Dynamics  = 'BS'
star_time = 0
T = 0.5
dT = 1/52
r = 0
mu = [0.03,0.01]
sigma = [0.3,0.1]

Benv = BenchmarkReplication(N = N, Nsim = Nsim,
                             Dynamics = Dynamics, start_time = star_time,
                               T = T, dT = dT, r = r, mu = mu, sigma = sigma)
Benv.seed(seed=random.seed(10))

steps = 5000
env = gym.wrappers.TimeLimit(Benv, max_episode_steps=steps)
env = Monitor(env, allow_early_resets=True)

path_folder = f"C:/Users/yoshi/OneDrive/Desktop/Research/Benchmark_RL/BS_PPO" # PATH to the BS_PPO_Models folder
path = f"{path_folder}/BS_PPO_{str(steps)}_{str(int(sigma[0]*100))}{str(int(sigma[1]*100))}"

eval_callback = EvalCallback(env, best_model_save_path=path_folder,
                             log_path=path_folder, eval_freq=500,
                             deterministic=True, render=False)

try:
    model = PPO.load(path, env = DummyVecEnv([lambda: env]), print_system_info=True)
except:
    print("Training model...")
    if not os.path.exists(f"{path_folder}/tensorboard/"):
        os.makedirs(f"{path_folder}/tensorboard/")
    model = PPO('MlpPolicy', DummyVecEnv([lambda: env]), learning_rate=0.001, verbose=1,
                tensorboard_log=f"{path_folder}/tensorboard/")
    model.learn(total_timesteps=steps, callback=eval_callback, log_interval = 100)
    model.save(f"{path}.zip")
    # To open tensorboard window, run tensorboard --logdir C:/Users/yoshi/OneDrive/Desktop/Research/Benchmark_RL/BS_PPO/tensorboard/PPO_1

##########################################
### Experiment
##########################################

Nepisodes = 25
rew = []
Pi = []

vec_env = model.get_env()
for i in range(Nepisodes):
    obs = Benv.reset()
    obs = [[obs[0][i] for i in range(len(obs[0]))]]
    cont = True
    i = 0
    while cont:
        action, _states = model.predict(obs, deterministic = True)
        if len(action) == 1: 
            obs, reward, terminated, truncated, info = Benv.step(action[0])
        else:
            obs, reward, terminated, truncated, info = Benv.step(action)
        i += 1
        if any([terminated,truncated]):
            cont = False
            Pi.append(Benv.Pi)
            rew.append(reward)

# Visualization
M = int(T/dT)-1
n = min(100,Nepisodes)
Pi = np.array(random.sample(Pi,n)).T

time = np.linspace(0,T,M)/dT
tt = np.full(shape=(n,M), fill_value=time).T
fig = plt.figure()
plt.plot(tt,Pi)
if not os.path.exists(f"{path_folder}/plots/"):
        os.makedirs(f"{path_folder}/plots/")
plt.savefig(f"{path_folder}/plots//BS_PPO_{str(steps)}_{str(int(sigma[0]*100))}{str(int(sigma[1]*100))}")
plt.show()

print(np.mean(rew),np.std(rew))




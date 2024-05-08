from Env import BenchmarkReplication
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np
import random

Benv = BenchmarkReplication(W = 100, N = 10, Nsim = 100, Dynamics = 'BS', start_time = 0, T = 26, dT = 1, r = 0, mu = [0.03,0.01], sigma = [0.3, 0.2])
#check_env(Benv) #check environment is accepted by SB3
Benv.seed(seed=random.seed(10))
model = PPO('MlpPolicy', DummyVecEnv([lambda: Benv]), learning_rate=0.001, verbose=1)

steps = 100000
try:
    model.load("BS_PPO_"+str(steps))
except:
    model.learn(total_timesteps=steps)
    model.save("BS_PPO_"+str(steps))

Nepisodes = 1
rew = []
for i in range(Nepisodes):
    vec_env = model.get_env()
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
import numpy as np
from numpy.random import PCG64DXSM, Generator
import scipy as sp
from Pricers import BSprice, BGprice
from MUFs import Var, Welford_Var
import gymnasium as gym
from gymnasium.spaces import Box
from typing import TYPE_CHECKING, Optional

class BenchmarkReplication(gym.Env):

    def __init__(self, N: int = 10, Nsim: int = 100, Dynamics: str = 'BS', start_time: float = 0, T: float = 0.5, dT: float = 0.5/52, r: float = 0, mu: list[float] = [0.03,0.01], sigma: list[float] = [0.3, 0.2]):
        self.start_time = start_time
        self.T = T
        self.dT = dT
        self.r = r
        
        self.N = N # There are N calls and N put options for each underlying. Then the player needs to make 4N-1 decisions.
        
        self.Dynamics = Dynamics
        self.S0 = [500,25]
        self.sigma = sigma
        self.mu = mu
        self.bp = [0.1,0.1]
        self.bn = [0.05,0.05]
        self.cp = [0.1,0.1]
        self.cn = [0.05,0.05]

        self.action_space = Box(low = -100, high = 100, shape = (4,N))
        self.observation_space = Box(low = 0, high = 10000, shape = (2,1), dtype = np.float64), # current prices of 2 underlying assets
        self.observation_space = self.observation_space[0]

        np.random.seed(9001) #For MC integration
        self.Nsim = Nsim
        if self.Dynamics == 'BS':
            self.X = np.array(self.dT*[self.mu+self.sigma*np.random.randn(2) for i in range(Nsim)])
        
    
    def seed(self, seed:int) -> None:
        self.np_random = Generator(PCG64DXSM(seed=seed)) #For ts creation

    def step(
        self,
        action
    ):
        T = self.T
        N = self.N

        S = self.ts[:][self.time]
        kmin = [0.7*S[i] for i in range(len(S))]
        kmax = [1.3*S[i] for i in range(len(S))]
        k0 = np.linspace(kmin[0],kmax[0],N)
        k1 = np.linspace(kmin[1],kmax[1],N)
        if self.Dynamics == 'BS':
            O = BSprice(S,k0,k1,self.r,self.dT,self.sigma)
        else:
            O = BGprice(S,k0,k1,self.r,self.dT,self.bp,self.cp,self.bn,self.cn)

        Cost = sum([np.dot(action[i],O[i]) for i in range(4)]) - action[3][-1]*O[3][-1]
        action[3][-1] = - Cost/O[3][-1]

        self.reward += - Welford_Var(S,self.dT,action,N,self.Nsim,k0,k1,self.mu,self.sigma,self.X,self.W0)

        self.time += 1
        
        if self.time == T:
            self.terminated = True
            self.truncated = True
        
        self.info = {'terminal_observation': [self.reward, self.time, self.terminated, self.truncated]}
        obs = np.array([[S[i]] for i in range(len(S))])

        return obs, self.reward, self.terminated, self.truncated, self.info
        
    def reset(
        self,
        *,
        seed: Optional[int] = None
    ):
        T = self.T
        N = self.N
        self.time = self.start_time # the current time is zero
        self.p = np.zeros(N)
        self.reward = 0
        self.p = np.zeros(4*N) # current position in each option
        
        if self.Dynamics == 'BS':
            eps0 = self.dT*[self.np_random.normal(self.mu[0],self.sigma[0]) for i in range(T)]
            eps1 = self.dT*[self.np_random.normal(self.mu[1],self.sigma[1]) for i in range(T)]
        
        self.ts = [[self.S0[0]*np.exp(sum(eps0[:i])), self.S0[1]*np.exp(sum(eps1[:i]))] for i in range(T)]

        S = self.ts[:][0]
        obs = np.array([[S[i]] for i in range(len(S))])

        self.terminated = False
        self.truncated = False

        return obs, {}

    def render(self):
        pass
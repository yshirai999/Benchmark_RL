import numpy as np
from numpy.random import PCG64DXSM, Generator
import scipy as sp
from Pricers import BSprice, BGprice
from MUFs import Var, Welford_Var
import gymnasium as gym
from gymnasium.spaces import Box
from typing import TYPE_CHECKING, Optional

class BenchmarkReplication(gym.Env):

    def __init__(self, N: int = 10, Nsim: int = 100, Dynamics: str = 'BS', start_time: float = 0, T: float = 1, dT: float = 1/52, r: float = 0, mu: list[float] = [0.03,0.01], sigma: list[float] = [0.3, 0.2]):
        self.start_time = start_time
        self.T = T
        self.dT = dT
        self.M = int(T/dT)
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

        self.Pi = [] # Portfolio value

        self.action_space = Box(low = -100, high = 100, shape = (4,N))
        self.observation_space = Box(low = 0, high = 10000, shape = (2,1), dtype = np.float64), # current prices of 2 underlying assets
        self.observation_space = self.observation_space[0]

        np.random.seed(9001) #For MC integration
        self.Nsim = Nsim
        if self.Dynamics == 'BS':
            self.X = np.array([self.mu+self.sigma*np.random.randn(2) for i in range(Nsim)])*self.dT
        
    
    def seed(self, seed:int) -> None:
        self.np_random = Generator(PCG64DXSM(seed=seed)) #For ts creation

    def step(
        self,
        action
    ):
        T = self.T
        M = self.M
        N = self.N

        S = self.ts[:][self.time]

        if self.time < M-1:
            kmin = [0.7*S[i] for i in range(len(S))]
            kmax = [1.3*S[i] for i in range(len(S))]
            k0 = np.linspace(kmin[0],kmax[0],N)
            k1 = np.linspace(kmin[1],kmax[1],N)
            if self.Dynamics == 'BS':
                O = BSprice(S,k0,k1,self.r,self.dT,self.sigma)
            else:
                O = BGprice(S,k0,k1,self.r,self.dT,self.bp,self.cp,self.bn,self.cn)
            
            Cost = sum([np.dot(action[i],O[i]) for i in range(4)]) - action[1][int(N/2)]*O[1][int(N/2)]
            action[1][int(N/2)] = - Cost/O[1][int(N/2)]

            self.reward += Welford_Var(S,self.dT,action,N,self.Nsim,k0,k1,self.mu,self.sigma,self.X)

            self.time += 1

            Snext = self.ts[:][self.time]
            
            xi = (Snext[0]**2+Snext[1]**2)/(Snext[0]+Snext[1])#-(S[0]**2+S[1]**2)/(S[0]+S[1])

            self.Pi.append(sum([(action[0][n]*max(Snext[0]-k0[n],0)+action[1][n]*max(k0[n]-Snext[0],0)) for n in range(N)])-xi) # actual result from the strategy

        if self.time == M-1:
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
        dT = self.dT
        N = self.N
        M = self.M
        self.time = self.start_time # the current time is zero
        self.p = np.zeros(N)
        self.reward = 0
        self.p = np.zeros(4*N) # current position in each option
        self.Pi = []
        mu = self.mu
        sigma = self.sigma
        
        if self.Dynamics == 'BS':
            eps0 = [dT*(mu[0]+self.np_random.normal(0,sigma[0]**2)) for i in range(M)]
            eps1 = [dT*(mu[1]+self.np_random.normal(0,sigma[1]**2)) for i in range(M)]
        
        self.ts = [[self.S0[0]*np.exp(sum(eps0[:i])), self.S0[1]*np.exp(sum(eps1[:i]))] for i in range(M)]

        S = self.ts[:][0]
        obs = np.array([[S[i]] for i in range(len(S))])

        self.terminated = False
        self.truncated = False

        return obs, {}

    def render(self):
        pass
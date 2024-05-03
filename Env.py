import numpy as np
import scipy as sp
from Pricers import BSprice, BGprice
import gymnasium as gym
from gymnasium.spaces import Box

class BenchmarkReplication(gym.Env):

    def __init__(self, W: float = 100, N: int = 10, Dynamics: str = 'BS', start_time: float = 0, T: float = 26, dT: float = 1, r: float = 0, mu: list[float] = [0.03,0.01], sigma: list[float] = [0.3, 0.2]):
        self.start_time = start_time
        self.T = T
        self.dT = dT
        
        self.W0 = W # Initial Wealth
        self.W = W # Current Wealth
        self.N = N # There are N calls and N put options for each underlying. Then the player needs to make 4N-1 decisions.
        
        self.Dynamics = Dynamics
        self.S0 = [500,25]
        self.sigma = sigma
        self.mu = mu
        self.bp = [0.1,0.1]
        self.bn = [0.05,0.05]
        self.cp = [0.1,0.1]
        self.cn = [0.05,0.05]
        # self.sigma = kwargs.get('sigma',None)
        # self.mu = kwargs.get('mu',None)
        # self.bp = kwargs.get('bp',None)
        # self.cp = kwargs.get('cp',None)
        # self.bn = kwargs.get('bn',None)
        # self.cn = kwargs.get('cn',None)

        self.action_spaces = Box(low = -np.inf, high = np.inf, shape = (4,N))
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = (2,)), # current prices of 2 underlying assets
        
    def step(self, action):
        T = self.T
        N = self.N
        p = self.p

        if self.time == T-1:
            terminated = True
        
        if self.W == 0:
            truncated = True

        if not all([terminated, truncated]):
            S = self.time_series[:][self.time]
            kmin = 0.7*S 
            kmax = 1.3*S
            k0 = np.linspace(kmin[0],kmax[0],N/2)
            k1 = np.linspace(kmin[1],kmax[0],N/2)
            if self.Dynamics == 'BS':
                O = BSprice(S,k0,k1,self.r,self.dT,self.sigma)
            else:
                O = BGprice(S,k0,k1,self.r,self.dT,self.sigma)
            
            Cost = sum([np.dot(action[i],O[i]) for i in range(4)])-action[3][-1]*O[3][-1]
            action[3][-1] = self.W - Cost

            self.W = 0
            for n in range(N):
                self.W = self.W \
                        + action[0][n]*max(S[0] - self.k0[n],0) \
                        + action[1][n]*max(self.k0[n] - S[0],0) \
                        + action[2][n]*max(self.k1[n] - S[1],0) \
                        + action[3][n]*max(S[1] - self.k1[n],0) 

            self.reward = self.W - S[0]*S[0]/sum(S) - S[1]*S[1]/sum(S) #The benchmark formed by SPY and XLE is subtracted
            self.time += 1
        
        info = {}

        return S, self.reward, terminated, truncated, info
        
    def reset(self):
        T = self.T
        N = self.N
        self.time = self.start_time # the current time is zero
        self.p = np.zeros(N)
        self.W = self.W0 # the initial cash
        self.reward = 0
        self.p = np.zeros(4*N) # current position in each option
        
        if self.Dynamics == 'BS':
            eps0 = self.dT*[self.mu[0]+self.sigma[0]*np.random.randn(1) for i in range(T)]
            eps1 = self.dT*[self.mu[1]+self.sigma[1]*np.random.randn(1) for i in range(T)]
        
        self.time_series = [[sum(eps0[:i]), sum(eps1[:i])] for i in range(T)]

        S = self.time_series[:][0]

        if self.time == T-1:
            terminated = True
        
        if self.W == 0:
            truncated = True

        info = {}

        return S, self.reward, terminated, truncated, info

    def render(self):
        pass
import numpy as np
import time

def Var(S: list[float], dT: float, a: np.ndarray, N: int, Nsim: int, k0: np.ndarray, k1: np.ndarray, mu: float, sigma: float, X: np.ndarray) -> float:
    mu = [mu[i]*dT for i in range(len(mu))]
    sigma = [sigma[i]*np.sqrt(dT) for i in range(len(sigma))]
    x = []
    for nsim in range(Nsim):
        Snew = S*np.exp(mu + sigma*X[nsim])
        xn = (Snew[0]**2+Snew[1]**2)/(Snew[0]+Snew[1])-(S[0]**2+S[1]**2)/(S[0]+S[1])
        xn = xn + sum([(a[0][n]*max(Snew[0]-k0[n],0)+a[1][n]*max(k0[n]-Snew[0],0)) for n in range(N)])
        xn = xn + sum([(a[2][n]*max(Snew[1]-k1[n],0)+a[3][n]*max(k1[n]-Snew[1],0)) for n in range(N)])
        x.append(xn)
    mu_sim = np.mean(x)
    sigma_sim = np.mean([(x[i]-mu_sim)**2 for i in range(Nsim)])
    return mu_sim - sigma_sim

def Welford_Var(S: list[float], dT: float, a: np.ndarray, N: int, Nsim: int, k0: np.ndarray, k1: np.ndarray, mu: float, sigma: float, X: np.ndarray) -> float:
    mu = [mu[i]*dT for i in range(len(mu))]
    sigma = [sigma[i]*np.sqrt(dT) for i in range(len(sigma))]
    x = []
    mean = 0
    count = 0
    M2 = 0
    for nsim in range(Nsim):
        count += 1
        Snew = S*np.exp(mu + sigma*X[nsim])
        xn = (Snew[0]**2+Snew[1]**2)/(Snew[0]+Snew[1])
        xn = xn + sum([(a[0][n]*max(Snew[0]-k0[n],0)+a[1][n]*max(k0[n]-Snew[0],0)) for n in range(N)])
        xn = xn + sum([(a[2][n]*max(Snew[1]-k1[n],0)+a[3][n]*max(k1[n]-Snew[1],0)) for n in range(N)])
        x.append(xn)
        mu_sim = np.mean(x)
        delta = xn-mean
        mean += delta/count
        delta2 = xn-mean
        M2 += delta * delta2
    sigma_sim = M2/count
    return mu_sim - sigma_sim

def Naive_Var(S: list[float], dT: float, a: np.ndarray, N: int, Nsim: int, k0: np.ndarray, k1: np.ndarray, mu: float, sigma: float, X: np.ndarray) -> float:
    mu = [mu[i]*dT for i in range(len(mu))]
    sigma = [sigma[i]*np.sqrt(dT) for i in range(len(sigma))]
    x = []
    count = 0
    for nsim in range(Nsim):
        count+=1
        Snew = S*np.exp(mu + sigma*X[nsim])
        xn = (Snew[0]**2+Snew[1]**2)/(Snew[0]+Snew[1])
        xn = xn + sum([(a[0][n]*max(Snew[0]-k0[n],0)+a[1][n]*max(k0[n]-Snew[0],0)) for n in range(N)])
        xn = xn + sum([(a[2][n]*max(Snew[1]-k1[n],0)+a[3][n]*max(k1[n]-Snew[1],0)) for n in range(N)])
        x.append(xn)
    sum_x  = np.sum(x)
    mu_sim = sum_x/Nsim
    sumsq = np.sum(x[i]**2 for i in range(Nsim))
    sigma_sim = (sumsq-(sum_x**2)/count)/count
    return mu_sim-sigma_sim

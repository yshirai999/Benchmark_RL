import numpy as np
import time

def Var(S: list[float], dT: float, a: np.ndarray, N: int, Nsim: int, k0: np.ndarray, k1: np.ndarray, mu: float, sigma: float, X: np.ndarray, W0: float) -> float:
    mu = [mu[i]*dT for i in range(len(mu))]
    sigma = [sigma[i]*np.sqrt(dT) for i in range(len(sigma))]
    x = []
    for nsim in range(Nsim):
        Snew = S*np.exp(mu + sigma*X[nsim])
        xn = (Snew[0]**2+Snew[1]**2)/(Snew[0]+Snew[1])
        xn = xn + sum([(a[0][n]*max(Snew[0]-k0[n],0)+a[1][n]*max(k0[n]-Snew[0],0))/W0 for n in range(N)])
        xn = xn + sum([(a[2][n]*max(Snew[1]-k1[n],0)+a[3][n]*max(k1[n]-Snew[1],0))/W0 for n in range(N)])
        x.append(xn)
    mu_sim = np.mean(x)
    sigma_sim = np.mean([(x[i]-mu_sim)**2 for i in range(nsim)])
    return sigma_sim

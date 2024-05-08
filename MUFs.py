import numpy as np
import time

def Var(S: list[float], a: np.ndarray, N: int, Nsim: int, k0: np.ndarray, k1: np.ndarray, mu: float, sigma: float, X: np.ndarray, W0: float) -> float:
    csi = [0,0]
    for nsim in range(Nsim):
        Snew = S*np.exp(mu + sigma*X[nsim])
        csi = csi - W0*(Snew[0]**2+Snew[1]**2)/(sum(Snew))
        for n in range(N):
            csi = csi + a[0][n]*max(Snew[0]-k0[n],0)+a[1][n]*max(k0[n]-Snew[0],0)
            csi = csi + a[2][n]*max(Snew[1]-k1[n],0)+a[3][n]*max(k1[n]-Snew[1],0)
        csi = csi*csi
    return np.mean(csi)

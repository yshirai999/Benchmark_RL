import numpy as np
import scipy as sp

def Var(S: list[float], a: np.ndarray, k0: np.ndarray, k1: np.ndarray, r: float, T: float, mu: float, sigma: float, X: np.ndarray) -> float:
    csi = [0,0]
    for n in range(len(X)):
        Snew = S*np.exp(mu + sigma*X[n])
        csi = csi + (Snew[0]**2+Snew[1]**2)/(sum(Snew))
        for i in range(len(k0)):
            csi = csi + a[0][i]*max(Snew[0]-k0[i],0)+a[1][i]*max(k0[i]-Snew[0])
        for i in range(len(k1)):
            csi = csi + a[2][i]*max(Snew[1]-k1[i],0)+a[3][i]*max(k1[i]-Snew[1])
        csi = csi*csi
    return np.mean(csi)

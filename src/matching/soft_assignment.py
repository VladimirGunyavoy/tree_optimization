import numpy as np
from dataclasses import dataclass

@dataclass
class SinkhornConfig:
    eps: float = 0.05          # температура (энтропия)
    n_iter: int = 150          # итераций нормировки
    big_cost: float = 1e6      # барьер для запретов (диагональ)
    annea_schedule: tuple = (0.1, 0.05, 0.02, 0.01, 0.005)  # по эпохам

def pairwise_sqdist(X: np.ndarray) -> np.ndarray:
    # X: (N,d)
    G = X @ X.T
    sq = np.diag(G)[:,None] + np.diag(G)[None,:] - 2*G
    return np.maximum(sq, 0.0)

def sinkhorn(C: np.ndarray, cfg: SinkhornConfig) -> np.ndarray:
    # C: (N,N) — стоимости; диагональ будет заменена на big_cost
    K = np.exp(-C / cfg.eps)
    u = np.ones(C.shape[0])
    v = np.ones(C.shape[0])
    for _ in range(cfg.n_iter):
        u = 1.0 / (K @ v)
        v = 1.0 / (K.T @ u)
    P = np.diag(u) @ K @ np.diag(v)
    return P  # ~двойная стохастичность

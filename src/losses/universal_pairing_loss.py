import numpy as np
from dataclasses import dataclass
from src.matching.soft_assignment import pairwise_sqdist, sinkhorn, SinkhornConfig
from typing import Dict, List, Optional

@dataclass
class UniversalLossConfig:
    eps: float = 0.05
    margin: float = 0.02
    lam_push: float = 0.3
    forbid_self: bool = True

def universal_loss(X: np.ndarray, 
                   loss_cfg: UniversalLossConfig,
                   sk_cfg: SinkhornConfig,
                   pairing_map: Optional[Dict[int, List[int]]] = None
                   ) -> dict:
    """
    X: (N,d) — конечные точки.
    pairing_map: dict {idx_внука -> [разрешенные_партнеры]}
    Возвращает dict с полями: total, pull, push, P, C.
    """
    N = X.shape[0]
    C = pairwise_sqdist(X)
    
    # Применяем маску разрешенных пар, если она есть
    if pairing_map:
        # Создаем маску запретов: True, если пара ЗАПРЕЩЕНА
        forbidden_mask = np.ones_like(C, dtype=bool)
        for i in range(N):
            # Важно: получаем копию, чтобы не изменять исходный словарь в дереве
            allowed_partners = pairing_map.get(i, []).copy()
            # Разрешаем самого себя (потом запретим диагональю)
            allowed_partners.append(i)
            forbidden_mask[i, allowed_partners] = False
        
        # Применяем маску
        C[forbidden_mask] = 1e6

    if loss_cfg.forbid_self:
        np.fill_diagonal(C, 1e6)

    # мягкое соответствие
    P = sinkhorn(C, sk_cfg)

    # 1) тянем парные расстояния
    L_pull = float((P * C).sum())

    # 2) отталкиваем от третьих (margin)
    L_push = 0.0
    for i in range(N):
        for j in range(N):
            if i == j: 
                continue
            pij = P[i, j]
            if pij <= 1e-12: 
                continue
            dij = np.sqrt(C[i, j])
            # усредненный hinge по третьим
            for k in range(N):
                if k == i or k == j: 
                    continue
                dik = np.sqrt(C[i, k])
                L_push += pij * max(0.0, loss_cfg.margin - (dik - dij))

    total = L_pull + loss_cfg.lam_push * L_push
    return {"total": total, "pull": L_pull, "push": L_push, "P": P, "C": C}
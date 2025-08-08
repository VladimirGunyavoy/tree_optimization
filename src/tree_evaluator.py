# ───────── tree_evaluator.py ────────────────────────────────────────
import numpy as np
from spore_tree import SporeTree

class TreeEvaluator:
    def __init__(self, tree: SporeTree):
        self.tree = tree
        self._last_dt = None
        self._initialised = False     # порядок внуков ещё не зафиксирован

    # ---------------------------------------------------------------
    def _build_if_needed(self, dt_all: np.ndarray):
        dt_all = np.asarray(dt_all).ravel()

        if not self._initialised:
            # первый вызов → создаём детей, внуков, фиксируем порядок
            self.tree.create_children(dt_children=dt_all[:4])
            self.tree.create_grandchildren(dt_grandchildren=dt_all[4:])
            self.tree.sort_and_pair_grandchildren()
            self.tree.calculate_mean_points()
            self._initialised = True
        elif not np.allclose(dt_all, self._last_dt):
            # только обновляем позиции, порядок уже зафиксирован
            self.tree.update_positions(
                dt_children      = dt_all[:4],
                dt_grandchildren = dt_all[4:]
            )

        self._last_dt = dt_all.copy()

    # площадь четырёхугольника
    def area(self, dt_all: np.ndarray) -> float:
        self._build_if_needed(dt_all)
        mp = self.tree.mean_points           # (4,2)
        x, y = mp[:, 0], mp[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                            np.dot(y, np.roll(x, 1)))

    # расстояния в 4 парах
    def pair_distances(self, dt_all: np.ndarray) -> np.ndarray:
        self._build_if_needed(dt_all)

        gc = self.tree.sorted_grandchildren  # ← всегда актуальный список
        d  = np.zeros(4)
        for k in range(4):
            i1, i2 = 2*k, 2*k+1
            p1 = gc[i1]['position']
            p2 = gc[i2]['position']
            d[k] = np.linalg.norm(p1 - p2)
        return d
# ────────────────────────────────────────────────────────────────────

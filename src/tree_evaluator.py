# ───────── tree_evaluator.py ────────────────────────────────────────
import numpy as np
from .spore_tree import SporeTree

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
        
    @staticmethod
    def _softmin(values, alpha=10.0):
        """Вычисляет softmin для массива значений."""
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        
        exp_vals = np.exp(-alpha * values)
        return -1/alpha * np.log(np.sum(exp_vals))

    def dynamic_loss(self, dt_all: np.ndarray, alpha: float = 10.0, 
                     w_distance: float = 1.0, w_velocity: float = 0.0,
                     w_repulsion: float = 0.0, w_time: float = 0.0, 
                     p_norm: float = 1.0) -> float:
        """
        Вычисляет общую взвешенную функцию потерь.

        Args:
            dt_all: Вектор временных шагов.
            alpha: Параметр softmin.
            w_distance: Вес для компоненты расстояния.
            w_velocity: Вес для компоненты скорости сближения.
            w_repulsion: Вес для компоненты отталкивания.
            w_time: Вес для бонуса за время (регуляризации).
            p_norm: Параметр p-нормы для бонуса за время.

        Returns:
            Итоговое скалярное значение функции потерь.
        """
        components = self.calculate_loss_components(dt_all, alpha, p_norm)
        
        total_loss = (
            w_distance * components['distance_loss'] +
            w_velocity * components['velocity_loss'] +
            w_repulsion * components['repulsion_loss'] -
            w_time * components['time_bonus']
        )
        return total_loss

    def calculate_loss_components(self, dt_all: np.ndarray, alpha: float = 10.0, p_norm: float = 1.0) -> dict:
        """
        Вычисляет и возвращает все "сырые" (невзвешенные) компоненты функции потерь.
        """
        self._build_if_needed(dt_all)
        
        # 1. Расчет компонентов встреч
        grandchildren = self.tree.grandchildren
        positions = {gc['global_idx']: gc['position'] for gc in grandchildren}
        velocities = {}
        for gc in grandchildren:
            state = gc['position']
            dot_v = self.tree.pendulum.pendulum_dynamics(state=state, control=gc['control'])
            velocities[gc['global_idx']] = dot_v
            
        distance_loss = 0.0
        velocity_loss = 0.0
        candidate_map = self.tree.pairing_candidate_map
        
        for i in range(len(grandchildren)):
            candidate_ids = candidate_map.get(i, [])
            if not candidate_ids:
                continue
                
            v_i = positions[i]
            dot_v_i = velocities[i]
            
            distance_costs = []
            velocity_costs = []
            
            for j in candidate_ids:
                v_j = positions[j]
                dot_v_j = velocities[j]
                r_ij = v_i - v_j
                dot_r_ij = dot_v_i - dot_v_j
                
                distance_costs.append(np.dot(r_ij, r_ij))
                velocity_costs.append(np.dot(r_ij, dot_r_ij))
            
            distance_loss += self._softmin(np.array(distance_costs), alpha=alpha)
            velocity_loss += self._softmin(np.array(velocity_costs), alpha=alpha)

        # 2. Расчет бонуса за время (ранее "регуляризация")
        epsilon = 1e-8
        time_bonus = np.sum(np.power(np.abs(dt_all) + epsilon, p_norm))
        
        # 3. Расчет потерь отталкивания
        repulsion_loss = 0.0
        mean_points = self.tree.mean_points
        if mean_points is not None and len(mean_points) == 4:
            from scipy.spatial.distance import pdist
            distances = pdist(mean_points)
            
            epsilon_rep = 1e-6
            repulsion_loss = np.sum(1.0 / (distances + epsilon_rep))

        return {
            'distance_loss': distance_loss,
            'velocity_loss': velocity_loss,
            'time_bonus': time_bonus,
            'repulsion_loss': repulsion_loss,
            'sum_abs_dt': np.sum(np.abs(dt_all))
        }
# ────────────────────────────────────────────────────────────────────

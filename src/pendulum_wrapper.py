"""
Изолированная обертка над функцией маятника из основного проекта.
"""
import sys
import os
import numpy as np

# Подключаемся к основному проекту только для PendulumSystem
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from logic.pendulum import PendulumSystem


class OptimizationPendulum:
    """
    Простая обертка над PendulumSystem только для оптимизации dt.
    Изолирует нас от изменений в основном проекте.
    """
    
    def __init__(self, g=9.81, l=2.0, m=1.0, damping=0.1, max_control=2.0):
        """Создает систему маятника с базовыми параметрами."""
        self.pendulum = PendulumSystem(
            g=g, l=l, m=m, 
            damping=damping, 
            max_control=max_control
        )
        
        # Кэшируем границы управления
        self.u_min, self.u_max = self.pendulum.get_control_bounds()
    
    def step(self, state, control, dt):
        """
        Один шаг интеграции маятника.
        
        Args:
            state: np.array([theta, theta_dot])
            control: float - управление 
            dt: float - временной шаг
            
        Returns:
            np.array([theta_new, theta_dot_new])
        """
        return self.pendulum.scipy_rk45_step(state, control, dt)
    
    def get_control_bounds(self):
        """Возвращает (u_min, u_max)"""
        return self.u_min, self.u_max
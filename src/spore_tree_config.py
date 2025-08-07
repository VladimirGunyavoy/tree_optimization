from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional

@dataclass
class SporeTreeConfig:
    """
    Конфигурация для дерева спор и оптимизации.
    """
    # Параметры дерева
    initial_position: np.ndarray = None  # [theta, theta_dot] - будет установлено в __post_init__
    dt_base: float = 0.001
    dt_grandchildren_factor: float = 0.05  # dt_внуков = dt_родителя * factor (1/20 вместо 1/10)
    
    # Параметры оптимизации  
    dt_bounds: Tuple[float, float] = (0.001, 0.2)
    epsilon: float = 1e-3  # точность сходимости пар
    max_iterations: int = 1000
    optimization_method: str = 'SLSQP'
    tolerance: float = 1e-6
    
    # Параметры валидации
    assert_pairing: bool = True  # проверять правильность пар после сортировки
    
    # Параметры визуализации (на будущее)
    figure_size: Tuple[int, int] = (14, 8)
    root_size: int = 100
    child_size: int = 80
    grandchild_size: int = 40
    show_debug: bool = False  # отладочная информация по умолчанию
    
    def __post_init__(self):
        """Устанавливает дефолтное начальное положение если не задано."""
        if self.initial_position is None:
            self.initial_position = np.array([np.pi, 0.0])
    
    def get_default_dt_vector(self) -> np.ndarray:
        """
        Возвращает дефолтный вектор времен для оптимизации.
        
        Returns:
            np.array из 12 элементов: [4 dt для детей] + [8 dt для внуков]
        """
        dt_children = np.ones(4) * self.dt_base
        dt_grandchildren = np.ones(8) * self.dt_base * self.dt_grandchildren_factor
        
        return np.hstack((dt_children, dt_grandchildren))
    
    def validate(self) -> bool:
        """
        Валидирует корректность параметров конфига.
        
        Returns:
            True если все параметры корректны
            
        Raises:
            ValueError при некорректных параметрах
        """
        if self.dt_base <= 0:
            raise ValueError(f"dt_base должен быть положительным, получен: {self.dt_base}")
            
        if self.dt_grandchildren_factor <= 0:
            raise ValueError(f"dt_grandchildren_factor должен быть положительным, получен: {self.dt_grandchildren_factor}")
            
        if self.dt_bounds[0] >= self.dt_bounds[1]:
            raise ValueError(f"Некорректные границы dt: {self.dt_bounds}")
            
        if self.epsilon <= 0:
            raise ValueError(f"epsilon должен быть положительным, получен: {self.epsilon}")
            
        if len(self.initial_position) != 2:
            raise ValueError(f"initial_position должен содержать 2 элемента [theta, theta_dot], получен: {self.initial_position}")
            
        return True
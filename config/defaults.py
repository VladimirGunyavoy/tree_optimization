"""
Базовые настройки для оптимизации dt.
"""

DEFAULT_CONFIG = {
    # Основные параметры оптимизации
    "dt_base": 0.1,
    "epsilon": 1e-3,
    "dt_bounds": (0.001, 0.2),
    
    # Параметры маятника  
    "pendulum": {
        "g": 9.81,
        "l": 2.0,
        "m": 1.0,
        "damping": 0.1,
        "max_control": 2.0
    },
    
    # Отладка
    "debug": {
        "show_topology_creation": False,
        "show_calculations": False,
        "show_optimization": False,
        "show_progress": False
    }
}

def create_config(preset="default", **overrides):
    """
    Создает конфигурацию с возможностью переопределения.
    
    Args:
        preset: str - пока только "default"
        **overrides - параметры для переопределения
    
    Returns:
        dict - готовая конфигурация
    """
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return config
import numpy as np
from src.spore_tree_config import SporeTreeConfig
from src.spore_tree import SporeTree
from src.pendulum import PendulumSystem
from src.pairs.find_optimal_pairs import find_optimal_pairs
from src.pairs.create_tree_from_pairs import create_tree_from_pairs
from src.area_opt.optimize_tree_area import optimize_tree_area


def find_optimal_dt_vector(initial_position, 
                          pendulum=None,
                          config=None,
                          constraint_distance=None,
                          area_optimization_dt_bounds=None,
                          show=False):
    """
    Находит оптимальный вектор времен dt через полный пайплайн оптимизации.
    
    Выполняет:
    1. Создание дерева из initial_position
    2. Поиск оптимальных пар внуков  
    3. Создание дерева с парными оптимизациями
    4. Оптимизацию площади дерева при constraint'ах расстояний
    
    Args:
        initial_position: np.array [theta, theta_dot] - начальная позиция корня
        pendulum: PendulumSystem - объект маятника (если None, создает дефолтный)
        config: SporeTreeConfig - конфигурация с параметрами дерева и поиска пар
        constraint_distance: float - максимальное расстояние между парами (если None, берет 1e-4)
        area_optimization_dt_bounds: tuple - границы dt для оптимизации площади (если None, берет config.dt_bounds)
        show: bool - вывод отладочной информации
        
    Returns:
        np.array: оптимальный вектор dt из 12 элементов [4 детей + 8 внуков]
        None: при ошибке на любом этапе
    """
    
    try:
        if show:
            print("ПОИСК ОПТИМАЛЬНОГО ВЕКТОРА DT")
            print("="*50)
            print(f"Начальная позиция: {initial_position}")
        
        # ================================================================
        # ПОДГОТОВКА ОБЪЕКТОВ
        # ================================================================
        
        # Используем глобальные объекты или создаем дефолтные
        if pendulum is None:
            if show:
                print("Создание дефолтного маятника...")
            pendulum = PendulumSystem(
                g=9.81, l=2.0, m=1.0, damping=0.05, max_control=2.0
            )
        
        if config is None:
            if show:
                print("Создание дефолтной конфигурации...")
            config = SporeTreeConfig(
                initial_position=initial_position,
                dt_base=0.1,
                dt_grandchildren_factor=0.1,
                figure_size=(10, 15)
            )
        else:
            # Обновляем позицию в существующем конфиге
            config.initial_position = initial_position.copy()
        
        config.validate()
        
        # Берем параметры оптимизации из конфига
        if constraint_distance is None:
            constraint_distance = 1e-4  # Дефолт, так как в SporeTreeConfig этого параметра нет
        
        # Границы dt для поиска пар (из config)
        pair_dt_bounds = config.dt_bounds
        
        # Границы dt для оптимизации площади (отдельный параметр или из config)
        if area_optimization_dt_bounds is None:
            area_dt_bounds = config.dt_bounds
        else:
            area_dt_bounds = area_optimization_dt_bounds
        
        max_iterations = config.max_iterations  
        optimization_method = config.optimization_method
        
        if show:
            print(f"Параметры оптимизации из конфига:")
            print(f"  constraint_distance: {constraint_distance}")
            print(f"  pair_dt_bounds (для поиска пар): {pair_dt_bounds}")
            print(f"  area_dt_bounds (для оптимизации площади): {area_dt_bounds}")
            print(f"  max_iterations: {max_iterations}")
            print(f"  optimization_method: {optimization_method}")
        
        # ================================================================
        # ЭТАП 1: СОЗДАНИЕ ИСХОДНОГО ДЕРЕВА
        # ================================================================
        
        if show:
            print(f"\nЭтап 1: Создание исходного дерева...")
        
        tree = SporeTree(pendulum, config)
        children = tree.create_children(show=show and False)
        grandchildren = tree.create_grandchildren(show=show and False)
        tree.sort_and_pair_grandchildren()
        tree.calculate_mean_points()
        
        if show:
            print(f"Дерево создано: {len(children)} детей, {len(grandchildren)} внуков")
        
        # ================================================================
        # ЭТАП 2: ПОИСК ОПТИМАЛЬНЫХ ПАР
        # ================================================================
        
        if show:
            print(f"\nЭтап 2: Поиск оптимальных пар...")
        
        pairs = find_optimal_pairs(tree, show=show)
        
        if pairs is None:
            if show:
                print("ОШИБКА: Не удалось найти оптимальные пары!")
            return None
        
        if show:
            print(f"Найдено {len(pairs)} оптимальных пар")
            print(pairs)
        
        # ================================================================
        # ЭТАП 3: СОЗДАНИЕ ДЕРЕВА С ПАРНЫМИ ОПТИМИЗАЦИЯМИ
        # ================================================================
        
        if show:
            print(f"\nЭтап 3: Создание дерева из пар...")
        
        result = create_tree_from_pairs(tree, pendulum, config, show=show and False)
        
        if not result or not result['success']:
            if show:
                print(f"ОШИБКА: Создание дерева из пар не удалось!")
                print(f"Ошибка: {result['error'] if result else 'критическая ошибка'}")
            return None
        
        paired_tree = result['optimized_tree']
        
        if show:
            print(f"Дерево с парными оптимизациями создано")
            print(f"Использовано пар: {result['stats']['pairs_found']}")
        
        # ================================================================
        # ЭТАП 4: ОПТИМИЗАЦИЯ ПЛОЩАДИ
        # ================================================================
        
        if show:
            print(f"\nЭтап 4: Оптимизация площади дерева...")
        
        optimization_result = optimize_tree_area(
            tree=paired_tree,
            pairs=pairs, 
            pendulum=pendulum,
            constraint_distance=constraint_distance,
            dt_bounds=area_dt_bounds,  # Используем специальные границы для оптимизации площади
            max_iterations=max_iterations,
            optimization_method=optimization_method,
            show=show and False  # Детальный дебаг только при необходимости
        )
        
        if not optimization_result or not optimization_result['success']:
            if show:
                print(f"ОШИБКА: Оптимизация площади не удалась!")
                if optimization_result:
                    print(f"Причина: {optimization_result.get('error', 'неизвестная ошибка')}")
            return None
        
        # ================================================================
        # ФИНАЛЬНЫЙ РЕЗУЛЬТАТ
        # ================================================================
        
        optimal_dt_vector = optimization_result['optimized_dt_vector']
        
        if show:
            print(f"\nОПТИМИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
            print(f"Оптимальные времена найдены:")
            print(f"  dt_children: {optimal_dt_vector[:4]}")
            print(f"  dt_grandchildren: {optimal_dt_vector[4:12]}")
            print(f"Улучшение площади: {optimization_result['improvement']:.6f}")
            print(f"Финальная площадь: {optimization_result['optimized_area']:.6f}")
        
        return optimal_dt_vector
        
    except Exception as e:
        if show:
            print(f"КРИТИЧЕСКАЯ ОШИБКА в find_optimal_dt_vector: {e}")
            import traceback
            traceback.print_exc()
        return None
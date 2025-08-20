def build_grandchild_distance_tables(tree, pendulum, dt_bounds=(0.001, 0.1), show=False):
    """
    Строит таблицы минимальных расстояний и оптимальных времен для пар внуков.
    
    Args:
        tree: SporeTree - объект дерева с созданными внуками
        pendulum: PendulumSystem - объект маятника
        dt_bounds: tuple - границы поиска |dt|
        show: bool - показать процесс построения
        
    Returns:
        dict: {
            'distance_table': DataFrame - минимальные расстояния,
            'time_table_i': DataFrame - оптимальные времена для первого внука,
            'time_table_j': DataFrame - оптимальные времена для второго внука,
            'convergence_table': DataFrame - скорости сближения,
            'optimization_results': dict - детальные результаты оптимизации
        }
    """
    import numpy as np
    import pandas as pd
    
    if not tree._grandchildren_created:
        raise RuntimeError("Сначала создайте внуков через tree.create_grandchildren()")
    
    if show:
        print("Построение таблиц расстояний и времен для пар внуков")
        print("=" * 60)
    
    # Импортируем функции (предполагаем что они в том же модуле или импортированы)
    from .compute_convergence_tables import compute_distance_derivative_table
    from .find_converging_pairs import find_converging_grandchild_pairs
    from .optimize_grandchild_pairs import optimize_grandchild_pair_distance
    
    # Шаг 1: Вычисляем таблицу скоростей сближения
    convergence_df = compute_distance_derivative_table(
        tree.grandchildren, pendulum, show=show
    )
    
    # Шаг 2: Находим сближающиеся пары
    converging_pairs = find_converging_grandchild_pairs(convergence_df, show=show)
    
    # Шаг 3: Инициализируем таблицы результатов
    n = len(tree.grandchildren)
    distance_table = np.full((n, n), np.nan)
    time_table_i = np.full((n, n), np.nan)
    time_table_j = np.full((n, n), np.nan)
    
    optimization_results = {}
    
    if show:
        print(f"\nОптимизация {len(converging_pairs)} сближающихся пар:")
    
    # Шаг 4: Оптимизируем каждую сближающуюся пару
    for pair in converging_pairs:
        gc_i_idx = pair['gc_i']
        gc_j_idx = pair['gc_j']
        pair_name = pair['pair_name']
        
        if show:
            print(f"\n  Пара {pair_name} (скорость: {pair['velocity']:.5f}):")
        
        result = optimize_grandchild_pair_distance(
            gc_i_idx, gc_j_idx, 
            tree.grandchildren, tree.children, pendulum,
            dt_bounds=dt_bounds, show=show
        )
        
        optimization_results[pair_name] = result
        
        if result['success']:
            # Заполняем таблицы ПРАВИЛЬНО для симметричности
            distance_table[gc_i_idx, gc_j_idx] = result['min_distance']
            distance_table[gc_j_idx, gc_i_idx] = result['min_distance']
            
            # ИСПРАВЛЕНО: правильная интерпретация времен
            # time_table_i[row, col] = оптимальное время для внука row при встрече с внуком col
            # time_table_j[row, col] = оптимальное время для внука col при встрече с внуком row
            
            # Для пары (i,j):
            time_table_i[gc_i_idx, gc_j_idx] = result['optimal_dt_i']  # время для внука i
            time_table_j[gc_i_idx, gc_j_idx] = result['optimal_dt_j']  # время для внука j
            
            # Для симметричной пары (j,i):
            time_table_i[gc_j_idx, gc_i_idx] = result['optimal_dt_j']  # время для внука j
            time_table_j[gc_j_idx, gc_i_idx] = result['optimal_dt_i']  # время для внука i
            
            if show:
                print(f"    Успех: расстояние={result['min_distance']:.5f}")
                print(f"    dt_i={result['optimal_dt_i']:+.5f}, dt_j={result['optimal_dt_j']:+.5f}")
        else:
            if show:
                print(f"    Неудача: оптимизация не сошлась")
    
    # Шаг 5: Создаем DataFrame
    row_names = [f"gc_{i}" for i in range(n)]
    col_names = [f"gc_{i}" for i in range(n)]
    
    distance_df = pd.DataFrame(distance_table, index=row_names, columns=col_names)
    time_i_df = pd.DataFrame(time_table_i, index=row_names, columns=col_names)
    time_j_df = pd.DataFrame(time_table_j, index=row_names, columns=col_names)
    
    if show:
        print(f"\nРезультирующие таблицы:")
        print("=" * 40)
        
        print(f"\nТаблица минимальных расстояний:")
        with pd.option_context('display.precision', 6):
            print(distance_df)
        
        print(f"\nТаблица времен для внука i:")
        with pd.option_context('display.precision', 5):
            print(time_i_df)
        
        print(f"\nТаблица времен для внука j:")
        with pd.option_context('display.precision', 5):
            print(time_j_df)
        
        # Статистика
        valid_distances = distance_df.values[~np.isnan(distance_df.values)]
        if len(valid_distances) > 0:
            print(f"\nСтатистика:")
            print(f"  Успешных оптимизаций: {len(valid_distances)}")
            print(f"  Минимальное расстояние: {np.min(valid_distances):.6f}")
            print(f"  Среднее расстояние: {np.mean(valid_distances):.6f}")
    
    return {
        'distance_table': distance_df,
        'time_table_i': time_i_df,
        'time_table_j': time_j_df,
        'convergence_table': convergence_df,
        'optimization_results': optimization_results,
        'converging_pairs': converging_pairs
    }


def build_grandchild_parent_distance_tables(tree, pendulum, dt_bounds=(0.001, 0.1), show=False):
    """
    Строит таблицы минимальных расстояний и оптимальных времен для пар внук-родитель.
    
    Args:
        tree: SporeTree - объект дерева с созданными внуками
        pendulum: PendulumSystem - объект маятника
        dt_bounds: tuple - границы поиска |dt|
        show: bool - показать процесс построения
        
    Returns:
        dict: {
            'distance_table': DataFrame - минимальные расстояния,
            'time_table': DataFrame - оптимальные времена,
            'convergence_table': DataFrame - скорости сближения,
            'optimization_results': dict - детальные результаты оптимизации
        }
    """
    import numpy as np
    import pandas as pd
    
    if not tree._grandchildren_created:
        raise RuntimeError("Сначала создайте внуков через tree.create_grandchildren()")
    
    if show:
        print("Построение таблиц расстояний и времен для пар внук-родитель")
        print("=" * 60)
    
    # Импортируем функции
    from .compute_convergence_tables import compute_grandchild_parent_convergence_table
    from .find_converging_pairs import find_converging_grandchild_parent_pairs
    from .optimize_grandchild_pairs import optimize_grandchild_parent_distance
    
    # Шаг 1: Вычисляем таблицу скоростей сближения
    convergence_df = compute_grandchild_parent_convergence_table(
        tree.grandchildren, tree.children, pendulum, show=show
    )
    
    # Шаг 2: Находим сближающиеся пары
    converging_pairs = find_converging_grandchild_parent_pairs(convergence_df, show=show)
    
    # Шаг 3: Инициализируем таблицы результатов
    n_grandchildren = len(tree.grandchildren)
    n_parents = len(tree.children)
    distance_table = np.full((n_grandchildren, n_parents), np.nan)
    time_table = np.full((n_grandchildren, n_parents), np.nan)
    
    optimization_results = {}
    
    if show:
        print(f"\nОптимизация {len(converging_pairs)} сближающихся пар внук-родитель:")
    
    # Шаг 4: Оптимизируем каждую сближающуюся пару
    for pair in converging_pairs:
        gc_idx = pair['gc_idx']
        parent_idx = pair['parent_idx']
        pair_name = pair['pair_name']
        
        if show:
            print(f"\n  Пара {pair_name} (скорость: {pair['velocity']:.5f}):")
        
        result = optimize_grandchild_parent_distance(
            gc_idx, parent_idx,
            tree.grandchildren, tree.children, pendulum,
            dt_bounds=dt_bounds, show=show
        )
        
        optimization_results[pair_name] = result
        
        if result['success']:
            distance_table[gc_idx, parent_idx] = result['min_distance']
            time_table[gc_idx, parent_idx] = result['optimal_dt']
            
            if show:
                print(f"    Успех: расстояние={result['min_distance']:.5f}")
                print(f"    optimal_dt={result['optimal_dt']:+.5f}")
        else:
            if show:
                print(f"    Неудача: оптимизация не сошлась")
    
    # Шаг 5: Создаем DataFrame
    row_names = [f"gc_{i}" for i in range(n_grandchildren)]
    col_names = [f"parent_{i}" for i in range(n_parents)]
    
    distance_df = pd.DataFrame(distance_table, index=row_names, columns=col_names)
    time_df = pd.DataFrame(time_table, index=row_names, columns=col_names)
    
    if show:
        print(f"\nРезультирующие таблицы:")
        print("=" * 40)
        
        print(f"\nТаблица минимальных расстояний внук-родитель:")
        with pd.option_context('display.precision', 6):
            print(distance_df)
        
        print(f"\nТаблица оптимальных времен внук-родитель:")
        with pd.option_context('display.precision', 5):
            print(time_df)
        
        # Статистика
        valid_distances = distance_df.values[~np.isnan(distance_df.values)]
        if len(valid_distances) > 0:
            print(f"\nСтатистика:")
            print(f"  Успешных оптимизаций: {len(valid_distances)}")
            print(f"  Минимальное расстояние: {np.min(valid_distances):.6f}")
            print(f"  Среднее расстояние: {np.mean(valid_distances):.6f}")
    
    return {
        'distance_table': distance_df,
        'time_table': time_df,
        'convergence_table': convergence_df,
        'optimization_results': optimization_results,
        'converging_pairs': converging_pairs
    }
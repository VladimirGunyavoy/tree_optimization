def compute_distance_derivative_table(grandchildren, pendulum, show=False):
    """
    Составляет таблицу первых производных расстояний между всеми парами внуков.
    
    Args:
        grandchildren: list - список внуков с полями 'position', 'control', 'dt'
        pendulum: PendulumSystem - объект маятника для вычисления скоростей
        show: bool - выводить таблицу
        
    Returns:
        pandas.DataFrame: симметричная таблица значений d/dt|r_i - r_j|
            Отрицательные: сближаются (чем меньше, тем быстрее)
            Положительные: расходятся (чем больше, тем быстрее)
    """
    import numpy as np
    import pandas as pd
    
    n = len(grandchildren)
    values_table = np.zeros((n, n))
    
    # Вычисляем "сырые" скорости всех внуков (без учета направления времени)
    raw_velocities = []
    time_directions = []
    for i, gc in enumerate(grandchildren):
        pos = gc['position']
        control = gc['control']
        
        # Сохраняем направление времени отдельно
        time_directions.append(np.sign(gc['dt']))
        
        # Получаем "сырую" динамику маятника (всегда для времени вперед)
        dynamics = pendulum.pendulum_dynamics(pos, control)  # [theta_dot, theta_ddot]
        raw_velocities.append(dynamics)
        
    if show:
        print("Отладочная информация первых 3 внуков:")
        for i in range(min(3, n)):
            gc = grandchildren[i]
            direction = "forward" if time_directions[i] > 0 else "backward"
            print(f"  Внук {i}: dt={gc['dt']:+.5f} ({direction})")
            print(f"    raw_dynamics={raw_velocities[i]}, time_direction={time_directions[i]:+1.0f}")
    
    # Заполняем только верхний треугольник (оптимизация в 2 раза)  
    for i in range(n):
        for j in range(i+1, n):  # Только j > i
            # Позиции внуков
            r1 = grandchildren[i]['position']
            r2 = grandchildren[j]['position']
            
            # Сырые скорости (без учета направления времени)
            v1_raw = raw_velocities[i]
            v2_raw = raw_velocities[j]
            
            # Направления времени
            sign1 = time_directions[i]
            sign2 = time_directions[j]
            
            # Вектор между точками
            r_diff = r1 - r2
            
            # ИСПРАВЛЕНО: правильный учет направлений времени
            # Формула v_diff = sign1 * v1_raw - sign2 * v2_raw работает универсально:
            # - Если sign1 == sign2: получаем sign1 * (v1_raw - v2_raw) 
            # - Если sign1 != sign2: получаем скорости с учетом встречного движения во времени
            v_diff = sign1 * v1_raw - sign2 * v2_raw
            
            # Текущее расстояние
            distance = np.linalg.norm(r_diff)
            
            if distance < 1e-10:
                derivative_value = 0.0
            else:
                # Производная расстояния: d/dt |r1 - r2| = (r1-r2)·(v1-v2) / |r1-r2|
                derivative_value = np.dot(r_diff, v_diff) / distance
            
            # Заполняем симметрично
            values_table[i, j] = derivative_value
            values_table[j, i] = derivative_value
    
    # Создаем pandas DataFrame
    df = pd.DataFrame(values_table, 
                     index=[f"gc_{i}" for i in range(n)],
                     columns=[f"gc_{i}" for i in range(n)])
    
    if show:
        print("Таблица первых производных расстояний d/dt|r_i - r_j|:")
        print("   < 0: сближаются (чем меньше, тем быстрее)")
        print("   = 0: стационарно") 
        print("   > 0: расходятся (чем больше, тем быстрее)")
        print()
        # Форматируем вывод с 5 знаками после запятой
        with pd.option_context('display.precision', 5):
            print(df)
        
        # Дополнительная статистика (только верхний треугольник)
        upper_triangle = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
        valid_values = upper_triangle.stack().values
        
        negative_count = (valid_values < -1e-6).sum()
        zero_count = ((valid_values >= -1e-6) & (valid_values <= 1e-6)).sum()
        positive_count = (valid_values > 1e-6).sum()
        
        print(f"\nСтатистика:")
        print(f"  Сближающихся пар: {negative_count}")
        print(f"  Стационарных пар: {zero_count}")
        print(f"  Расходящихся пар: {positive_count}")
        print(f"  Всего уникальных пар: {len(valid_values)}")
        
        if negative_count > 0:
            min_val = valid_values[valid_values < -1e-6].min()
            print(f"  Максимальная скорость сближения: {min_val:.5f}")
        if positive_count > 0:
            max_val = valid_values[valid_values > 1e-6].max()
            print(f"  Максимальная скорость расхождения: {max_val:.5f}")
    
    return df


def compute_grandchild_parent_convergence_table(grandchildren, children, pendulum, show=False):
    """
    Составляет таблицу первых производных расстояний между внуками и ЧУЖИМИ родителями.
    
    Args:
        grandchildren: list - список внуков с полями 'position', 'control', 'dt', 'parent_idx'
        children: list - список родителей с полями 'position', 'control', 'dt'
        pendulum: PendulumSystem - объект маятника для вычисления скоростей
        show: bool - выводить таблицу
        
    Returns:
        pandas.DataFrame: таблица d/dt|r_внук - r_родитель|
            Строки: внуки (gc_0, gc_1, ...)
            Столбцы: родители (parent_0, parent_1, ...)
            Значения < 0: сближаются
            Значения > 0: расходятся
            NaN: свой родитель (исключен)
    """
    import numpy as np
    import pandas as pd
    
    n_grandchildren = len(grandchildren)
    n_parents = len(children)
    values_table = np.full((n_grandchildren, n_parents), np.nan)
    
    # Вычисляем сырые скорости внуков (без учета направления времени)
    grandchild_raw_velocities = []
    grandchild_time_directions = []
    for gc in grandchildren:
        pos = gc['position']
        control = gc['control']
        grandchild_time_directions.append(np.sign(gc['dt']))
        dynamics = pendulum.pendulum_dynamics(pos, control)
        grandchild_raw_velocities.append(dynamics)
    
    # РОДИТЕЛИ СТАТИЧНЫ - их скорость равна 0 (они не эволюционируют во времени)
    
    # Заполняем таблицу
    for gc_idx, gc in enumerate(grandchildren):
        own_parent_idx = gc['parent_idx']
        
        for parent_idx, parent in enumerate(children):
            # Пропускаем своего родителя
            if parent_idx == own_parent_idx:
                continue
            
            # Позиции
            gc_pos = gc['position']
            parent_pos = parent['position']
            
            # Сырая скорость внука и направление времени
            gc_vel_raw = grandchild_raw_velocities[gc_idx]
            gc_time_sign = grandchild_time_directions[gc_idx]
            
            # Вектор между точками
            r_diff = gc_pos - parent_pos
            
            # ИСПРАВЛЕНО: родители статичны (скорость = 0)
            # v_diff = скорость_внука - 0 = gc_time_sign * gc_vel_raw
            v_diff = gc_time_sign * gc_vel_raw
            
            # Текущее расстояние
            distance = np.linalg.norm(r_diff)
            
            if distance < 1e-10:
                derivative_value = 0.0
            else:
                # Производная расстояния: d/dt |r_внук - r_родитель|
                derivative_value = np.dot(r_diff, v_diff) / distance
            
            values_table[gc_idx, parent_idx] = derivative_value
    
    # Создаем pandas DataFrame
    df = pd.DataFrame(values_table,
                     index=[f"gc_{i}" for i in range(n_grandchildren)],
                     columns=[f"parent_{i}" for i in range(n_parents)])
    
    if show:
        print("Таблица сближения внуков с ЧУЖИМИ родителями d/dt|r_внук - r_родитель|:")
        print("   < 0: внук сближается с родителем")
        print("   = 0: стационарно")
        print("   > 0: внук отдаляется от родителя")
        print("   NaN: свой родитель (исключен)")
        print()
        
        # Форматируем вывод
        with pd.option_context('display.precision', 5):
            print(df)
        
        # Статистика по сближениям
        valid_values = df.values[~np.isnan(df.values)]
        
        approaching_count = (valid_values < -1e-6).sum()
        stationary_count = ((valid_values >= -1e-6) & (valid_values <= 1e-6)).sum()
        receding_count = (valid_values > 1e-6).sum()
        
        print(f"\nСтатистика:")
        print(f"  Внуков сближается с чужими родителями: {approaching_count}")
        print(f"  Стационарных: {stationary_count}")
        print(f"  Внуков отдаляется от чужих родителей: {receding_count}")
        print(f"  Всего связей внук-чужой_родитель: {len(valid_values)}")
        
        if approaching_count > 0:
            min_val = valid_values[valid_values < -1e-6].min()
            print(f"  Максимальная скорость сближения: {min_val:.5f}")
        if receding_count > 0:
            max_val = valid_values[valid_values > 1e-6].max()
            print(f"  Максимальная скорость отдаления: {max_val:.5f}")
        
        # Показываем какие внуки к каким родителям сближаются
        print(f"\nВнуки, сближающиеся с чужими родителями:")
        for gc_idx in range(n_grandchildren):
            approaching_parents = []
            for parent_idx in range(n_parents):
                value = df.iloc[gc_idx, parent_idx]
                if not np.isnan(value) and value < -1e-6:
                    approaching_parents.append(f"parent_{parent_idx}({value:.5f})")
            
            if approaching_parents:
                print(f"  gc_{gc_idx}: {', '.join(approaching_parents)}")
    
    return df
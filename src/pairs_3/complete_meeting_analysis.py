def complete_meeting_analysis(tree, pendulum, dt_bounds=(0.001, 0.1), 
                              export_results=False, output_dir="results", show=True):
    """
    Полный анализ всех возможных встреч в дереве спор.
    
    Создает:
    1. Таблицу минимальных расстояний внук-внук
    2. Таблицу оптимальных времен внук-внук  
    3. Таблицу минимальных расстояний внук-родитель
    4. Таблицу оптимальных времен внук-родитель
    5. Хронологию встреч для каждого внука
    
    Args:
        tree: SporeTree - объект дерева с созданными внуками
        pendulum: PendulumSystem - объект маятника
        dt_bounds: tuple - границы поиска |dt| (учитывает направления времени)
        export_results: bool - экспортировать результаты в CSV
        output_dir: str - директория для экспорта
        show: bool - показать весь процесс анализа
        
    Returns:
        dict: полные результаты анализа
    """
    import numpy as np
    import pandas as pd
    import os
    
    if not tree._grandchildren_created:
        raise RuntimeError("Сначала создайте внуков через tree.create_grandchildren()")
    
    if show:
        print("ПОЛНЫЙ АНАЛИЗ ВСТРЕЧ В ДЕРЕВЕ СПОР")
        print("=" * 60)
        print(f"Внуков: {len(tree.grandchildren)}")
        print(f"Родителей: {len(tree.children)}")
        print(f"Границы dt: {dt_bounds}")
        
        # Показываем направления времени внуков
        forward_count = sum(1 for gc in tree.grandchildren if gc['dt'] > 0)
        backward_count = len(tree.grandchildren) - forward_count
        print(f"Forward внуков: {forward_count}, Backward внуков: {backward_count}")
    
    # ========================================================================
    # ЭТАП 1: АНАЛИЗ ВСТРЕЧ ВНУК-ВНУК
    # ========================================================================
    
    if show:
        print(f"\nЭТАП 1: АНАЛИЗ ВСТРЕЧ ВНУК-ВНУК")
        print("-" * 50)
    
    # Вычисляем скорости сближения внуков
    n_gc = len(tree.grandchildren)
    gc_gc_convergence = np.zeros((n_gc, n_gc))
    
    # Вычисляем скорости всех внуков
    velocities = []
    for gc in tree.grandchildren:
        time_sign = np.sign(gc['dt'])
        dynamics = pendulum.pendulum_dynamics(gc['position'], gc['control'])
        velocity_vector = time_sign * dynamics
        velocities.append(velocity_vector)
    
    # Заполняем таблицу скоростей сближения
    for i in range(n_gc):
        for j in range(i+1, n_gc):
            r_diff = tree.grandchildren[i]['position'] - tree.grandchildren[j]['position']
            v_diff = velocities[i] - velocities[j]
            distance = np.linalg.norm(r_diff)
            
            if distance < 1e-10:
                derivative_value = 0.0
            else:
                derivative_value = np.dot(r_diff, v_diff) / distance
            
            gc_gc_convergence[i, j] = derivative_value
            gc_gc_convergence[j, i] = derivative_value
    
    gc_gc_convergence_df = pd.DataFrame(
        gc_gc_convergence,
        index=[f"gc_{i}" for i in range(n_gc)],
        columns=[f"gc_{i}" for i in range(n_gc)]
    )
    
    # Находим сближающиеся пары внук-внук
    gc_gc_converging_pairs = []
    for i in range(n_gc):
        for j in range(i+1, n_gc):
            velocity = gc_gc_convergence[i, j]
            if velocity < -1e-6:
                gc_gc_converging_pairs.append({
                    'gc_i': i, 'gc_j': j, 'velocity': velocity,
                    'pair_name': f"gc_{i}-gc_{j}"
                })
    
    gc_gc_converging_pairs.sort(key=lambda x: x['velocity'])
    
    if show:
        print(f"Найдено {len(gc_gc_converging_pairs)} сближающихся пар внук-внук")
    
    # Оптимизируем встречи внук-внук
    gc_gc_distance_table = np.full((n_gc, n_gc), np.nan)
    gc_gc_time_i_table = np.full((n_gc, n_gc), np.nan)
    gc_gc_time_j_table = np.full((n_gc, n_gc), np.nan)
    gc_gc_optimization_results = {}
    
    for pair in gc_gc_converging_pairs:
        gc_i_idx, gc_j_idx = pair['gc_i'], pair['gc_j']
        gc_i = tree.grandchildren[gc_i_idx]
        gc_j = tree.grandchildren[gc_j_idx]
        
        # Определяем ограничения времени
        if gc_i['dt'] > 0:
            dt_i_bounds = dt_bounds
        else:
            dt_i_bounds = (-dt_bounds[1], -dt_bounds[0])
            
        if gc_j['dt'] > 0:
            dt_j_bounds = dt_bounds
        else:
            dt_j_bounds = (-dt_bounds[1], -dt_bounds[0])
        
        # Оптимизируем
        def distance_function(dt_params):
            dt_i, dt_j = dt_params
            try:
                parent_i_pos = tree.children[gc_i['parent_idx']]['position']
                parent_j_pos = tree.children[gc_j['parent_idx']]['position']
                pos_i = pendulum.step(parent_i_pos, gc_i['control'], dt_i)
                pos_j = pendulum.step(parent_j_pos, gc_j['control'], dt_j)
                return np.linalg.norm(pos_i - pos_j)
            except:
                return 1e6
        
        from scipy.optimize import minimize
        
        x0 = [(dt_i_bounds[0] + dt_i_bounds[1]) / 2, 
              (dt_j_bounds[0] + dt_j_bounds[1]) / 2]
        bounds = [dt_i_bounds, dt_j_bounds]
        
        try:
            result = minimize(distance_function, x0=x0, bounds=bounds, method='L-BFGS-B')
            if result.success:
                gc_gc_distance_table[gc_i_idx, gc_j_idx] = result.fun
                gc_gc_distance_table[gc_j_idx, gc_i_idx] = result.fun
                gc_gc_time_i_table[gc_i_idx, gc_j_idx] = result.x[0]
                gc_gc_time_j_table[gc_i_idx, gc_j_idx] = result.x[1]
                gc_gc_time_i_table[gc_j_idx, gc_i_idx] = result.x[1]
                gc_gc_time_j_table[gc_j_idx, gc_i_idx] = result.x[0]
                
                gc_gc_optimization_results[pair['pair_name']] = {
                    'success': True, 'min_distance': result.fun,
                    'optimal_dt_i': result.x[0], 'optimal_dt_j': result.x[1]
                }
        except:
            gc_gc_optimization_results[pair['pair_name']] = {'success': False}
    
    # ========================================================================
    # ЭТАП 2: АНАЛИЗ ВСТРЕЧ ВНУК-РОДИТЕЛЬ
    # ========================================================================
    
    if show:
        print(f"\nЭТАП 2: АНАЛИЗ ВСТРЕЧ ВНУК-РОДИТЕЛЬ")
        print("-" * 50)
    
    # Вычисляем скорости сближения внук-родитель
    n_parents = len(tree.children)
    gc_parent_convergence = np.full((n_gc, n_parents), np.nan)
    
    # Вычисляем скорости родителей
    parent_velocities = []
    for parent in tree.children:
        time_sign = np.sign(parent['dt'])
        dynamics = pendulum.pendulum_dynamics(parent['position'], parent['control'])
        parent_velocities.append(time_sign * dynamics)
    
    # Заполняем таблицу сближения внук-родитель
    for gc_idx, gc in enumerate(tree.grandchildren):
        for parent_idx in range(n_parents):
            if parent_idx == gc['parent_idx']:  # Пропускаем своего родителя
                continue
            
            r_diff = gc['position'] - tree.children[parent_idx]['position']
            v_diff = velocities[gc_idx] - parent_velocities[parent_idx]
            distance = np.linalg.norm(r_diff)
            
            if distance < 1e-10:
                derivative_value = 0.0
            else:
                derivative_value = np.dot(r_diff, v_diff) / distance
            
            gc_parent_convergence[gc_idx, parent_idx] = derivative_value
    
    gc_parent_convergence_df = pd.DataFrame(
        gc_parent_convergence,
        index=[f"gc_{i}" for i in range(n_gc)],
        columns=[f"parent_{i}" for i in range(n_parents)]
    )
    
    # Находим сближающиеся пары внук-родитель
    gc_parent_converging_pairs = []
    for gc_idx in range(n_gc):
        for parent_idx in range(n_parents):
            velocity = gc_parent_convergence[gc_idx, parent_idx]
            if not np.isnan(velocity) and velocity < -1e-6:
                gc_parent_converging_pairs.append({
                    'gc_idx': gc_idx, 'parent_idx': parent_idx, 'velocity': velocity,
                    'pair_name': f"gc_{gc_idx}-parent_{parent_idx}"
                })
    
    gc_parent_converging_pairs.sort(key=lambda x: x['velocity'])
    
    if show:
        print(f"Найдено {len(gc_parent_converging_pairs)} сближающихся пар внук-родитель")
    
    # Оптимизируем встречи внук-родитель
    gc_parent_distance_table = np.full((n_gc, n_parents), np.nan)
    gc_parent_time_table = np.full((n_gc, n_parents), np.nan)
    gc_parent_optimization_results = {}
    
    for pair in gc_parent_converging_pairs:
        gc_idx, parent_idx = pair['gc_idx'], pair['parent_idx']
        gc = tree.grandchildren[gc_idx]
        
        # Определяем ограничения времени для внука
        if gc['dt'] > 0:
            dt_bounds_signed = dt_bounds
        else:
            dt_bounds_signed = (-dt_bounds[1], -dt_bounds[0])
        
        # Оптимизируем
        def distance_function(dt):
            try:
                gc_parent_pos = tree.children[gc['parent_idx']]['position']
                target_pos = tree.children[parent_idx]['position']
                final_pos = pendulum.step(gc_parent_pos, gc['control'], dt)
                return np.linalg.norm(final_pos - target_pos)
            except:
                return 1e6
        
        from scipy.optimize import minimize_scalar
        
        try:
            result = minimize_scalar(distance_function, bounds=dt_bounds_signed, method='bounded')
            if result.success:
                gc_parent_distance_table[gc_idx, parent_idx] = result.fun
                gc_parent_time_table[gc_idx, parent_idx] = result.x
                
                gc_parent_optimization_results[pair['pair_name']] = {
                    'success': True, 'min_distance': result.fun, 'optimal_dt': result.x
                }
        except:
            gc_parent_optimization_results[pair['pair_name']] = {'success': False}
    
    # ========================================================================
    # ЭТАП 3: СОЗДАНИЕ ХРОНОЛОГИИ
    # ========================================================================
    
    if show:
        print(f"\nЭТАП 3: СОЗДАНИЕ ХРОНОЛОГИИ ВСТРЕЧ")
        print("-" * 50)
    
    chronology = {}
    
    for gc_idx in range(n_gc):
        meetings = []
        gc = tree.grandchildren[gc_idx]
        
        # Встречи с другими внуками
        for other_gc_idx in range(n_gc):
            if gc_idx == other_gc_idx:
                continue
            
            distance = gc_gc_distance_table[gc_idx, other_gc_idx]
            if not np.isnan(distance):
                meeting = {
                    'type': 'grandchild',
                    'partner': f"gc_{other_gc_idx}",
                    'partner_idx': other_gc_idx,
                    'distance': distance,
                    'time_for_gc': gc_gc_time_i_table[gc_idx, other_gc_idx],
                    'time_for_partner': gc_gc_time_j_table[gc_idx, other_gc_idx],
                    'quality': 1.0 / (distance + 1e-8),
                    'convergence_velocity': gc_gc_convergence[gc_idx, other_gc_idx]
                }
                meetings.append(meeting)
        
        # Встречи с чужими родителями
        for parent_idx in range(n_parents):
            if parent_idx == gc['parent_idx']:
                continue
            
            distance = gc_parent_distance_table[gc_idx, parent_idx]
            if not np.isnan(distance):
                meeting = {
                    'type': 'parent',
                    'partner': f"parent_{parent_idx}",
                    'partner_idx': parent_idx,
                    'distance': distance,
                    'time_for_gc': gc_parent_time_table[gc_idx, parent_idx],
                    'time_for_partner': None,
                    'quality': 1.0 / (distance + 1e-8),
                    'convergence_velocity': gc_parent_convergence[gc_idx, parent_idx]
                }
                meetings.append(meeting)
        
        # Сортируем по качеству
        meetings.sort(key=lambda x: x['quality'], reverse=True)
        chronology[gc_idx] = meetings
        
        if show:
            direction = "forward" if gc['dt'] > 0 else "backward"
            print(f"gc_{gc_idx} ({direction}): {len(meetings)} встреч")
            for i, meeting in enumerate(meetings[:3]):  # Топ-3
                time_info = f"t={meeting['time_for_gc']:+.4f}с"
                if meeting['time_for_partner'] is not None:
                    time_info += f" (партнер: {meeting['time_for_partner']:+.4f}с)"
                print(f"  {i+1}. {meeting['partner']}: "
                      f"расст={meeting['distance']:.5f}, {time_info}")
    
    # ========================================================================
    # ЭТАП 4: СОЗДАНИЕ ИТОГОВЫХ ТАБЛИЦ
    # ========================================================================
    
    # Создаем DataFrame для всех таблиц
    results = {
        'gc_gc_tables': {
            'distance_table': pd.DataFrame(gc_gc_distance_table, 
                                         index=[f"gc_{i}" for i in range(n_gc)],
                                         columns=[f"gc_{i}" for i in range(n_gc)]),
            'time_table_i': pd.DataFrame(gc_gc_time_i_table,
                                       index=[f"gc_{i}" for i in range(n_gc)],
                                       columns=[f"gc_{i}" for i in range(n_gc)]),
            'time_table_j': pd.DataFrame(gc_gc_time_j_table,
                                       index=[f"gc_{i}" for i in range(n_gc)],
                                       columns=[f"gc_{i}" for i in range(n_gc)]),
            'convergence_table': gc_gc_convergence_df,
            'optimization_results': gc_gc_optimization_results
        },
        'gc_parent_tables': {
            'distance_table': pd.DataFrame(gc_parent_distance_table,
                                         index=[f"gc_{i}" for i in range(n_gc)],
                                         columns=[f"parent_{i}" for i in range(n_parents)]),
            'time_table': pd.DataFrame(gc_parent_time_table,
                                     index=[f"gc_{i}" for i in range(n_gc)],
                                     columns=[f"parent_{i}" for i in range(n_parents)]),
            'convergence_table': gc_parent_convergence_df,
            'optimization_results': gc_parent_optimization_results
        },
        'chronology': chronology,
        'summary': {
            'total_grandchildren': n_gc,
            'total_gc_gc_meetings': len([r for r in gc_gc_optimization_results.values() if r['success']]),
            'total_gc_parent_meetings': len([r for r in gc_parent_optimization_results.values() if r['success']]),
            'grandchildren_with_meetings': sum(1 for meetings in chronology.values() if meetings)
        }
    }
    
    # Экспорт в CSV если требуется
    if export_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # Экспортируем основные таблицы
        results['gc_gc_tables']['distance_table'].to_csv(
            os.path.join(output_dir, "gc_gc_distances.csv"))
        results['gc_parent_tables']['distance_table'].to_csv(
            os.path.join(output_dir, "gc_parent_distances.csv"))
        
        # Экспортируем хронологию
        chronology_data = []
        for gc_idx, meetings in chronology.items():
            for rank, meeting in enumerate(meetings, 1):
                chronology_data.append({
                    'grandchild': f"gc_{gc_idx}",
                    'rank': rank,
                    'partner': meeting['partner'],
                    'partner_type': meeting['type'],
                    'distance': meeting['distance'],
                    'time_for_gc': meeting['time_for_gc'],
                    'time_for_partner': meeting['time_for_partner'],
                    'quality': meeting['quality']
                })
        
        chronology_df = pd.DataFrame(chronology_data)
        chronology_df.to_csv(os.path.join(output_dir, "chronology.csv"), index=False)
        
        if show:
            print(f"\nРезультаты экспортированы в: {output_dir}")
    
    if show:
        print(f"\nАНАЛИЗ ЗАВЕРШЕН")
        print("=" * 30)
        print(f"Встреч внук-внук: {results['summary']['total_gc_gc_meetings']}")
        print(f"Встреч внук-родитель: {results['summary']['total_gc_parent_meetings']}")
    
    return results
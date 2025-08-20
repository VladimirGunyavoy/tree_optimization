def optimize_grandchild_pair_distance(gc_i_idx, gc_j_idx, grandchildren, children, pendulum, 
                                     dt_bounds=(0.001, 0.1), root_position=None, show=False):
    """
    Оптимизирует dt для пары внуков с учетом их направлений времени.
    
    Args:
        gc_i_idx, gc_j_idx: int - индексы внуков в паре
        grandchildren: list - список всех внуков
        children: list - список всех родителей
        pendulum: PendulumSystem - объект маятника
        dt_bounds: tuple - границы поиска |dt| (всегда положительные)
        root_position: np.array - позиция корня для расчета distance_constraint
        show: bool - показать детали оптимизации
        
    Returns:
        dict: результаты оптимизации
    """
    import numpy as np
    from scipy.optimize import minimize
    
    gc_i = grandchildren[gc_i_idx]
    gc_j = grandchildren[gc_j_idx]
    
    # ВЫЧИСЛЯЕМ DISTANCE_CONSTRAINT: 1/10 от минимального расстояния корень-родители
    if root_position is not None:
        parent_distances = []
        for parent in children:
            distance = np.linalg.norm(parent['position'] - root_position)
            parent_distances.append(distance)
        min_parent_distance = min(parent_distances)
        distance_constraint = min_parent_distance / 10.0
        if show:
            print(f"    Distance constraint: {distance_constraint:.5f} (1/10 от мин. расст. корень-родители: {min_parent_distance:.5f})")
    else:
        distance_constraint = None  # Без ограничений если корень не передан
    
    # Позиции родителей (стартовые точки)
    parent_i_pos = children[gc_i['parent_idx']]['position']
    parent_j_pos = children[gc_j['parent_idx']]['position']
    
    # Определяем разрешенные диапазоны dt для каждого внука
    # ИСПРАВЛЕНО: внук может двигаться только в своем направлении времени
    original_dt_i = gc_i['dt']
    original_dt_j = gc_j['dt']
    
    if original_dt_i > 0:  # Forward внук - только положительные dt
        dt_i_bounds = dt_bounds  # (0.001, 0.1)
        direction_i = "forward"
    else:  # Backward внук - только отрицательные dt
        dt_i_bounds = (-dt_bounds[1], -dt_bounds[0])  # (-0.1, -0.001)
        direction_i = "backward"
    
    if original_dt_j > 0:  # Forward внук - только положительные dt
        dt_j_bounds = dt_bounds  # (0.001, 0.1)
        direction_j = "forward"
    else:  # Backward внук - только отрицательные dt
        dt_j_bounds = (-dt_bounds[1], -dt_bounds[0])  # (-0.1, -0.001)
        direction_j = "backward"
    
    if show:
        print(f"    Внук i (gc_{gc_i_idx}): original_dt={original_dt_i:+.5f} ({direction_i})")
        print(f"    Ограничения i: dt ∈ [{dt_i_bounds[0]:.3f}, {dt_i_bounds[1]:.3f}]")
        print(f"    Внук j (gc_{gc_j_idx}): original_dt={original_dt_j:+.5f} ({direction_j})")
        print(f"    Ограничения j: dt ∈ [{dt_j_bounds[0]:.3f}, {dt_j_bounds[1]:.3f}]")
    
    def distance_function(dt_params):
        """Функция расстояния между двумя движущимися внуками"""
        dt_i, dt_j = dt_params
        
        try:
            # Вычисляем финальные позиции обоих внуков
            pos_i = pendulum.step(parent_i_pos, gc_i['control'], dt_i)
            pos_j = pendulum.step(parent_j_pos, gc_j['control'], dt_j)
            
            # Расстояние между ними
            distance = np.linalg.norm(pos_i - pos_j)
            
            return distance
            
        except Exception as e:
            if show:
                print(f"    Ошибка в distance_function: {e}")
            return 1e6
    
    # Границы учитывают направление времени
    bounds = [dt_i_bounds, dt_j_bounds]
    
    # Начальное приближение в середине разрешенного диапазона
    x0_i = (dt_i_bounds[0] + dt_i_bounds[1]) / 2
    x0_j = (dt_j_bounds[0] + dt_j_bounds[1]) / 2
    x0 = [x0_i, x0_j]
    
    if show:
        print(f"    Начальное приближение: dt_i={x0_i:.3f}, dt_j={x0_j:.3f}")
    
    # Пробуем разные методы оптимизации
    methods = ['L-BFGS-B', 'Nelder-Mead']
    
    best_result = None
    best_distance = float('inf')
    
    for method in methods:
        try:
            if method == 'L-BFGS-B':
                result = minimize(
                    distance_function,
                    x0=x0,
                    bounds=bounds,
                    method=method,
                    options={'ftol': 1e-9, 'gtol': 1e-6}
                )
            else:  # Nelder-Mead
                result = minimize(
                    distance_function,
                    x0=x0,
                    method=method,
                    options={'xatol': 1e-9, 'fatol': 1e-9}  # Правильные опции для Nelder-Mead
                )
            
            if show:
                print(f"    Метод {method}: success={result.success}, fun={result.fun:.5f}")
                print(f"    Метод {method}: result.x={result.x}")
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: убеждаемся что результат в границах
            if result.success:
                dt_i_test, dt_j_test = result.x
                dt_i_in_bounds = dt_i_bounds[0] <= dt_i_test <= dt_i_bounds[1]
                dt_j_in_bounds = dt_j_bounds[0] <= dt_j_test <= dt_j_bounds[1]
                
                # Проверяем constraint на расстояние если есть
                distance_ok = True
                if distance_constraint is not None:
                    test_distance = distance_function(result.x)
                    distance_ok = test_distance <= distance_constraint
                    if show and not distance_ok:
                        print(f"    Метод {method}: расстояние {test_distance:.5f} > {distance_constraint:.5f}")
                
                if dt_i_in_bounds and dt_j_in_bounds and distance_ok and result.fun < best_distance:
                    best_result = result
                    best_distance = result.fun
                    if show:
                        print(f"    Метод {method}: принят")
                elif not dt_i_in_bounds or not dt_j_in_bounds:
                    if show:
                        print(f"    Метод {method}: отклонен (вне границ)")
                elif not distance_ok:
                    if show:
                        print(f"    Метод {method}: отклонен (нарушен constraint расстояния)")
                else:
                    if show:
                        print(f"    Метод {method}: отклонен (хуже distance)")
            else:
                if show:
                    print(f"    Метод {method}: отклонен (не success)")
                
        except Exception as e:
            if show:
                print(f"    Метод {method}: ошибка {e}")
            continue
    
    # Формируем результат
    if best_result is not None and best_distance < 1e5:
        optimal_dt_i, optimal_dt_j = best_result.x
        
        # ПРОВЕРКА: убеждаемся что оптимальные времена соблюдают ограничения направления
        dt_i_valid = dt_i_bounds[0] <= optimal_dt_i <= dt_i_bounds[1]
        dt_j_valid = dt_j_bounds[0] <= optimal_dt_j <= dt_j_bounds[1]
        
        if not dt_i_valid:
            if show:
                print(f"    ОШИБКА: optimal_dt_i={optimal_dt_i:.5f} вне границ {dt_i_bounds}")
            return {'success': False, 'error': 'dt_i violation'}
            
        if not dt_j_valid:
            if show:
                print(f"    ОШИБКА: optimal_dt_j={optimal_dt_j:.5f} вне границ {dt_j_bounds}")
            return {'success': False, 'error': 'dt_j violation'}
        
        # Дополнительная проверка знаков
        if (original_dt_i > 0 and optimal_dt_i <= 0) or (original_dt_i < 0 and optimal_dt_i >= 0):
            if show:
                print(f"    ОШИБКА: внук i изменил направление времени: {original_dt_i:+.5f} → {optimal_dt_i:+.5f}")
            return {'success': False, 'error': 'time direction violation i'}
            
        if (original_dt_j > 0 and optimal_dt_j <= 0) or (original_dt_j < 0 and optimal_dt_j >= 0):
            if show:
                print(f"    ОШИБКА: внук j изменил направление времени: {original_dt_j:+.5f} → {optimal_dt_j:+.5f}")
            return {'success': False, 'error': 'time direction violation j'}
        
        # Вычисляем финальные позиции
        final_pos_i = pendulum.step(parent_i_pos, gc_i['control'], optimal_dt_i)
        final_pos_j = pendulum.step(parent_j_pos, gc_j['control'], optimal_dt_j)
        
        return {
            'success': True,
            'min_distance': best_distance,
            'optimal_dt_i': optimal_dt_i,
            'optimal_dt_j': optimal_dt_j,
            'final_position_i': final_pos_i,
            'final_position_j': final_pos_j,
            'method_used': 'scipy_optimize',
            'distance_constraint': distance_constraint,
            'passes_constraint': distance_constraint is None or best_distance <= distance_constraint,
            'constraints': {
                'direction_i': direction_i,
                'direction_j': direction_j,
                'bounds_i': dt_i_bounds,
                'bounds_j': dt_j_bounds
            },
            'iterations': getattr(best_result, 'nit', 0)
        }
    else:
        return {
            'success': False,
            'min_distance': float('inf'),
            'method_used': 'failed',
            'distance_constraint': distance_constraint,
            'passes_constraint': False,
            'constraints': {
                'direction_i': direction_i,
                'direction_j': direction_j,
                'bounds_i': dt_i_bounds,
                'bounds_j': dt_j_bounds
            }
        }


def optimize_grandchild_parent_distance(gc_idx, parent_idx, grandchildren, children, pendulum,
                                       dt_bounds=(0.001, 0.1), show=False):
    """
    Оптимизирует dt для внука чтобы приблизиться к заданному родителю.
    
    Args:
        gc_idx: int - индекс внука
        parent_idx: int - индекс родителя 
        grandchildren: list - список всех внуков
        children: list - список всех родителей
        pendulum: PendulumSystem - объект маятника
        dt_bounds: tuple - границы поиска |dt|
        show: bool - показать детали оптимизации
        
    Returns:
        dict: результаты оптимизации
    """
    import numpy as np
    from scipy.optimize import minimize_scalar
    
    gc = grandchildren[gc_idx]
    parent = children[parent_idx]
    
    # Позиция родителя внука (стартовая точка)
    gc_parent_pos = children[gc['parent_idx']]['position']
    
    # Целевая позиция (позиция целевого родителя)
    target_parent_pos = parent['position']
    
    # Определяем разрешенный диапазон dt для внука
    # ИСПРАВЛЕНО: внук может двигаться только в своем направлении времени
    original_dt = gc['dt']
    
    if original_dt > 0:  # Forward внук - только положительные dt
        dt_bounds_signed = dt_bounds  # (0.001, 0.1)
        direction = "forward"
    else:  # Backward внук - только отрицательные dt
        dt_bounds_signed = (-dt_bounds[1], -dt_bounds[0])  # (-0.1, -0.001)
        direction = "backward"
    
    if show:
        print(f"    Внук gc_{gc_idx} ({direction}) к родителю {parent_idx}")
        print(f"    dt ∈ [{dt_bounds_signed[0]:.3f}, {dt_bounds_signed[1]:.3f}]")
    
    def distance_function(dt):
        """Функция расстояния от внука до целевого родителя"""
        try:
            # Вычисляем финальную позицию внука
            gc_final_pos = pendulum.step(gc_parent_pos, gc['control'], dt)
            
            # Расстояние до целевого родителя
            distance = np.linalg.norm(gc_final_pos - target_parent_pos)
            
            return distance
            
        except Exception as e:
            if show:
                print(f"    Ошибка в distance_function: {e}")
            return 1e6
    
    # Одномерная оптимизация
    try:
        result = minimize_scalar(
            distance_function,
            bounds=dt_bounds_signed,
            method='bounded',
            options={'xatol': 1e-9}
        )
        
        if show:
            print(f"    Результат: success={result.success}, min_distance={result.fun:.5f}")
        
        if result.success:
            optimal_dt = result.x
            
            # Вычисляем финальную позицию
            final_pos = pendulum.step(gc_parent_pos, gc['control'], optimal_dt)
            
            return {
                'success': True,
                'min_distance': result.fun,
                'optimal_dt': optimal_dt,
                'final_position': final_pos,
                'method_used': 'minimize_scalar',
                'constraints': {
                    'direction': direction,
                    'bounds': dt_bounds_signed
                },
                'iterations': getattr(result, 'nit', 0)
            }
        else:
            return {
                'success': False,
                'min_distance': float('inf'),
                'method_used': 'failed',
                'constraints': {
                    'direction': direction,
                    'bounds': dt_bounds_signed
                }
            }
            
    except Exception as e:
        if show:
            print(f"    Ошибка в оптимизации: {e}")
        return {
            'success': False,
            'min_distance': float('inf'),
            'method_used': 'failed',
            'error': str(e),
            'constraints': {
                'direction': direction,
                'bounds': dt_bounds_signed
            }
        }
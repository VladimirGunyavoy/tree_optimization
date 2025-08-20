def optimize_grandchild_pair_distance(gc_i_idx, gc_j_idx, grandchildren, children, pendulum, 
                                     dt_bounds=None, root_position=None, show=False):
    """
    Оптимизирует dt для пары внуков с учетом их направлений времени.
    РАСШИРЕННАЯ ВЕРСИЯ с детальным дебагом оптимизации.
    """
    import numpy as np
    from scipy.optimize import minimize
    
    gc_i = grandchildren[gc_i_idx]
    gc_j = grandchildren[gc_j_idx]
    
    # ВЫЧИСЛЯЕМ DISTANCE_CONSTRAINT
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
        distance_constraint = None
    
    # Позиции родителей (стартовые точки)
    parent_i_pos = children[gc_i['parent_idx']]['position']
    parent_j_pos = children[gc_j['parent_idx']]['position']
    
    # АДАПТИВНЫЕ ГРАНИЦЫ: от 0 до 2 * максимальное время родителей
    if dt_bounds is None:
        parent_times = [abs(child['dt']) for child in children]
        max_parent_time = max(parent_times)
        dt_max = 2 * max_parent_time
        dt_bounds = (0.001, dt_max)  # Минимум 0.001 чтобы избежать нуля
        
        if show:
            print(f"    📊 Времена родителей: {[f'{t:.5f}' for t in parent_times]}")
            print(f"    📊 Максимальное время родителя: {max_parent_time:.5f}")
            print(f"    📊 Адаптивные границы dt: (0.001, {dt_max:.5f})")
    else:
        if show:
            print(f"    📊 Фиксированные границы dt: {dt_bounds}")
    
    # Определяем разрешенные диапазоны dt для каждого внука
    original_dt_i = gc_i['dt']
    original_dt_j = gc_j['dt']
    
    if original_dt_i > 0:
        dt_i_bounds = dt_bounds
        direction_i = "forward"
    else:
        dt_i_bounds = (-dt_bounds[1], -dt_bounds[0])
        direction_i = "backward"
    
    if original_dt_j > 0:
        dt_j_bounds = dt_bounds
        direction_j = "forward"
    else:
        dt_j_bounds = (-dt_bounds[1], -dt_bounds[0])
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
        
        # Тестируем функцию в начальной точке
        initial_distance = distance_function(x0)
        print(f"    Начальное расстояние: {initial_distance:.6f}")
    
    # Пробуем разные методы оптимизации с ДЕТАЛЬНЫМ АНАЛИЗОМ
    methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']
    
    best_result = None
    best_distance = float('inf')
    all_results = {}
    
    for method in methods:
        try:
            if method == 'L-BFGS-B':
                result = minimize(
                    distance_function,
                    x0=x0,
                    bounds=bounds,
                    method=method,
                    options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 1000}
                )
            elif method == 'Nelder-Mead':
                result = minimize(
                    distance_function,
                    x0=x0,
                    method=method,
                    options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000}
                )
            elif method == 'Powell':
                result = minimize(
                    distance_function,
                    x0=x0,
                    method=method,
                    options={'ftol': 1e-12, 'xtol': 1e-12, 'maxiter': 1000}
                )
            
            # ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТА
            if show:
                print(f"    Метод {method}: success={result.success}, fun={result.fun:.8f}")
                print(f"    Метод {method}: result.x={result.x}")
                print(f"    Метод {method}: nit={getattr(result, 'nit', 'N/A')}")
                if hasattr(result, 'message'):
                    print(f"    Метод {method}: message='{result.message}'")
                if hasattr(result, 'nfev'):
                    print(f"    Метод {method}: nfev={result.nfev}")
            
            # Сохраняем ВСЕ результаты для анализа
            all_results[method] = {
                'result': result,
                'success': result.success,
                'fun': result.fun,
                'x': result.x.copy(),
                'message': getattr(result, 'message', 'N/A')
            }
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: проверяем границы ВРУЧНУЮ
            if len(result.x) >= 2:
                dt_i_test, dt_j_test = result.x[0], result.x[1]
                dt_i_in_bounds = dt_i_bounds[0] <= dt_i_test <= dt_i_bounds[1]
                dt_j_in_bounds = dt_j_bounds[0] <= dt_j_test <= dt_j_bounds[1]
                
                # Проверяем constraint на расстояние если есть
                distance_ok = True
                if distance_constraint is not None:
                    test_distance = distance_function(result.x)
                    distance_ok = test_distance <= distance_constraint
                else:
                    test_distance = result.fun
                
                if show:
                    print(f"    Метод {method}: dt_i в границах: {dt_i_in_bounds} ({dt_i_test:.6f})")
                    print(f"    Метод {method}: dt_j в границах: {dt_j_in_bounds} ({dt_j_test:.6f})")
                    print(f"    Метод {method}: расстояние: {test_distance:.8f}")
                    print(f"    Метод {method}: проходит constraint: {distance_ok}")
                
                # КРИТИЧЕСКИ ВАЖНО: принимаем результат если:
                # 1. Границы соблюдены 
                # 2. Constraint соблюден
                # 3. Расстояние лучше предыдущего
                # ИГНОРИРУЕМ success=False если результат хороший!
                
                bounds_ok = dt_i_in_bounds and dt_j_in_bounds
                
                if bounds_ok and distance_ok and test_distance < best_distance:
                    best_result = result
                    best_distance = test_distance
                    if show:
                        print(f"    Метод {method}: ✅ ПРИНЯТ как лучший (игнорируем success={result.success})")
                else:
                    if show:
                        reasons = []
                        if not bounds_ok:
                            reasons.append("вне границ")
                        if not distance_ok:
                            reasons.append("нарушен constraint")
                        if test_distance >= best_distance:
                            reasons.append(f"хуже distance ({test_distance:.6f} >= {best_distance:.6f})")
                        print(f"    Метод {method}: ❌ отклонен ({'; '.join(reasons)})")
            else:
                if show:
                    print(f"    Метод {method}: ❌ отклонен (некорректный result.x)")
                
        except Exception as e:
            if show:
                print(f"    Метод {method}: ❌ ошибка {e}")
            all_results[method] = {'error': str(e)}
            continue
    
    # ФИНАЛЬНЫЙ АНАЛИЗ
    if show:
        print(f"\n    🔍 СВОДКА ВСЕХ МЕТОДОВ:")
        for method, data in all_results.items():
            if 'error' in data:
                print(f"      {method}: ОШИБКА - {data['error']}")
            else:
                status = "✅ принят" if best_result == data['result'] else "❌ отклонен"
                print(f"      {method}: success={data['success']}, fun={data['fun']:.8f}, {status}")
    
    # Формируем результат
    if best_result is not None and best_distance < 1e5:
        optimal_dt_i, optimal_dt_j = best_result.x
        
        # Финальная проверка направлений времени
        dt_i_valid = dt_i_bounds[0] <= optimal_dt_i <= dt_i_bounds[1]
        dt_j_valid = dt_j_bounds[0] <= optimal_dt_j <= dt_j_bounds[1]
        
        if not dt_i_valid:
            if show:
                print(f"    🚨 КРИТИЧЕСКАЯ ОШИБКА: optimal_dt_i={optimal_dt_i:.5f} вне границ {dt_i_bounds}")
            return {'success': False, 'error': 'dt_i violation', 'all_results': all_results}
            
        if not dt_j_valid:
            if show:
                print(f"    🚨 КРИТИЧЕСКАЯ ОШИБКА: optimal_dt_j={optimal_dt_j:.5f} вне границ {dt_j_bounds}")
            return {'success': False, 'error': 'dt_j violation', 'all_results': all_results}
        
        # Проверка направлений времени
        if (original_dt_i > 0 and optimal_dt_i <= 0) or (original_dt_i < 0 and optimal_dt_i >= 0):
            if show:
                print(f"    🚨 ОШИБКА: внук i изменил направление времени: {original_dt_i:+.5f} → {optimal_dt_i:+.5f}")
            return {'success': False, 'error': 'time direction violation i', 'all_results': all_results}
            
        if (original_dt_j > 0 and optimal_dt_j <= 0) or (original_dt_j < 0 and optimal_dt_j >= 0):
            if show:
                print(f"    🚨 ОШИБКА: внук j изменил направление времени: {original_dt_j:+.5f} → {optimal_dt_j:+.5f}")
            return {'success': False, 'error': 'time direction violation j', 'all_results': all_results}
        
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
            'method_used': 'enhanced_multi_method',
            'distance_constraint': distance_constraint,
            'passes_constraint': distance_constraint is None or best_distance <= distance_constraint,
            'constraints': {
                'direction_i': direction_i,
                'direction_j': direction_j,
                'bounds_i': dt_i_bounds,
                'bounds_j': dt_j_bounds
            },
            'all_results': all_results,  # Полная информация о всех методах
            'best_method': next(method for method, data in all_results.items() 
                               if 'result' in data and data['result'] == best_result),
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
            },
            'all_results': all_results,  # Даже при неудаче показываем что пробовали
            'error': 'no_valid_solution'
        }


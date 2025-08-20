def optimize_grandchild_parent_distance(gc_idx, parent_idx, grandchildren, children, pendulum,
                                       dt_bounds=None, show=False):
    """
    Оптимизирует dt для внука чтобы приблизиться к заданному родителю.
    УЛУЧШЕННАЯ ВЕРСИЯ с адаптивными границами dt.
    """
    import numpy as np
    from scipy.optimize import minimize_scalar
    
    gc = grandchildren[gc_idx]
    parent = children[parent_idx]
    
    # Позиция родителя внука (стартовая точка)
    gc_parent_pos = children[gc['parent_idx']]['position']
    
    # Целевая позиция (позиция целевого родителя)
    target_parent_pos = parent['position']
    
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
    
    # Определяем разрешенный диапазон dt для внука
    original_dt = gc['dt']
    
    if original_dt > 0:  # Forward внук - только положительные dt
        dt_bounds_signed = dt_bounds  # (0.001, dt_max)
        direction = "forward"
    else:  # Backward внук - только отрицательные dt
        dt_bounds_signed = (-dt_bounds[1], -dt_bounds[0])  # (-dt_max, -0.001)
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
    
    # Тестируем функцию в начальной точке
    if show:
        mid_point = (dt_bounds_signed[0] + dt_bounds_signed[1]) / 2
        initial_distance = distance_function(mid_point)
        print(f"    Начальное расстояние (dt={mid_point:.5f}): {initial_distance:.6f}")
    
    # Одномерная оптимизация
    try:
        result = minimize_scalar(
            distance_function,
            bounds=dt_bounds_signed,
            method='bounded',
            options={'xatol': 1e-12, 'maxiter': 1000}
        )
        
        if show:
            print(f"    Результат: success={result.success}, min_distance={result.fun:.8f}")
            print(f"    Оптимальный dt: {result.x:.8f}")
            if hasattr(result, 'message'):
                print(f"    Message: '{result.message}'")
            if hasattr(result, 'nfev'):
                print(f"    Функция вызвана: {result.nfev} раз")
        
        if result.success:
            optimal_dt = result.x
            
            # Дополнительная проверка границ
            dt_in_bounds = dt_bounds_signed[0] <= optimal_dt <= dt_bounds_signed[1]
            
            if show:
                print(f"    dt в границах: {dt_in_bounds}")
            
            if dt_in_bounds:
                # Вычисляем финальную позицию
                final_pos = pendulum.step(gc_parent_pos, gc['control'], optimal_dt)
                
                return {
                    'success': True,
                    'min_distance': result.fun,
                    'optimal_dt': optimal_dt,
                    'final_position': final_pos,
                    'method_used': 'minimize_scalar_bounded',
                    'constraints': {
                        'direction': direction,
                        'bounds': dt_bounds_signed,
                        'adaptive_bounds': dt_bounds
                    },
                    'iterations': getattr(result, 'nit', 0),
                    'function_evaluations': getattr(result, 'nfev', 0)
                }
            else:
                if show:
                    print(f"    ❌ Результат вне границ!")
                return {
                    'success': False,
                    'min_distance': float('inf'),
                    'method_used': 'failed_bounds_check',
                    'error': 'result_out_of_bounds',
                    'constraints': {
                        'direction': direction,
                        'bounds': dt_bounds_signed,
                        'adaptive_bounds': dt_bounds
                    }
                }
        else:
            return {
                'success': False,
                'min_distance': float('inf'),
                'method_used': 'minimize_scalar_failed',
                'error': getattr(result, 'message', 'optimization_failed'),
                'constraints': {
                    'direction': direction,
                    'bounds': dt_bounds_signed,
                    'adaptive_bounds': dt_bounds
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
                'bounds': dt_bounds_signed,
                'adaptive_bounds': dt_bounds
            }
        }


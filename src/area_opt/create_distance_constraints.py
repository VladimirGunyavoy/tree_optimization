import numpy as np


def create_distance_constraints(pairs, tree, pendulum, constraint_distance=1e-5, show=False):
    """
    Создает список функций-констрейнтов для оптимизации площади.
    
    Для каждой пары внуков создает лямбда-функцию, которая:
    1. Принимает dt_vector [4 dt детей + 8 dt внуков]
    2. Вычисляет позиции обоих внуков в паре при этих временах
    3. Возвращает расстояние между ними
    
    Констрейнт считается выполненным когда расстояние <= constraint_distance.
    
    Args:
        pairs: список пар [(gc_i, gc_j, meeting_info), ...] от find_optimal_pairs()
        tree: исходное дерево SporeTree для получения структуры
        pendulum: объект маятника для вычисления step
        constraint_distance: float - максимально допустимое расстояние в парах
        show: bool - вывод отладочной информации
        
    Returns:
        list: список функций-констрейнтов для scipy.optimize.minimize
        dict: информация о констрейнтах для дебага
    """
    
    try:
        if not pairs:
            if show:
                print("Ошибка: Список пар пуст")
            return [], {}
            
        if show:
            print("СОЗДАНИЕ КОНСТРЕЙНТОВ РАССТОЯНИЙ")
            print("="*50)
            print(f"Создаем констрейнты для {len(pairs)} пар")
            print(f"Максимальное допустимое расстояние: {constraint_distance}")
        
        # Собираем информацию о структуре дерева
        root_position = tree.root['position']
        
        # Информация о детях: индекс -> (control, sign_dt)
        children_info = {}
        for i, child in enumerate(tree.children):
            children_info[i] = {
                'control': child['control'],
                'sign_dt': np.sign(child['dt'])  # +1 для forward, -1 для backward
            }
        
        # Информация о внуках: global_idx -> (parent_idx, control, sign_dt)
        grandchildren_info = {}
        for gc in tree.grandchildren:
            grandchildren_info[gc['global_idx']] = {
                'parent_idx': gc['parent_idx'],
                'control': gc['control'],
                'sign_dt': np.sign(gc['dt'])  # +1 для forward, -1 для backward
            }
        
        if show:
            print(f"\nИнформация о структуре:")
            print(f"  Корень: {root_position}")
            print(f"  Детей: {len(children_info)}")
            print(f"  Внуков: {len(grandchildren_info)}")
        
        # Создаем функции-констрейнты
        constraints = []
        constraint_info = {}
        
        for pair_idx, (gc_i, gc_j, meeting_info) in enumerate(pairs):
            
            # Получаем информацию о внуках в паре
            gc_i_info = grandchildren_info[gc_i]
            gc_j_info = grandchildren_info[gc_j]
            
            # Создаем замыкание для текущей пары
            def create_constraint_for_pair(gc_i_idx, gc_j_idx, gc_i_data, gc_j_data):
                """
                Создает функцию-констрейнт для конкретной пары.
                Использует замыкание для захвата параметров пары.
                """
                
                def constraint_function(dt_vector):
                    """
                    Функция-констрейнт для scipy.optimize.minimize.
                    
                    Args:
                        dt_vector: np.array из 12 элементов [4 dt детей + 8 dt внуков]
                        
                    Returns:
                        float: constraint_distance - расстояние_между_парой
                        Положительное значение = констрейнт выполнен
                        Отрицательное значение = констрейнт нарушен
                    """
                    try:
                        # Извлекаем времена из вектора
                        dt_children = dt_vector[0:4]
                        dt_grandchildren = dt_vector[4:12]
                        
                        # Вычисляем позицию первого внука
                        parent_i_idx = gc_i_data['parent_idx']
                        parent_i_control = children_info[parent_i_idx]['control']
                        parent_i_dt_signed = dt_children[parent_i_idx] * children_info[parent_i_idx]['sign_dt']
                        
                        # Позиция родителя gc_i
                        parent_i_pos = pendulum.step(root_position, parent_i_control, parent_i_dt_signed)
                        
                        # Позиция внука gc_i
                        gc_i_dt_signed = dt_grandchildren[gc_i_idx] * gc_i_data['sign_dt']
                        gc_i_pos = pendulum.step(parent_i_pos, gc_i_data['control'], gc_i_dt_signed)
                        
                        # Вычисляем позицию второго внука
                        parent_j_idx = gc_j_data['parent_idx']
                        parent_j_control = children_info[parent_j_idx]['control']
                        parent_j_dt_signed = dt_children[parent_j_idx] * children_info[parent_j_idx]['sign_dt']
                        
                        # Позиция родителя gc_j
                        parent_j_pos = pendulum.step(root_position, parent_j_control, parent_j_dt_signed)
                        
                        # Позиция внука gc_j
                        gc_j_dt_signed = dt_grandchildren[gc_j_idx] * gc_j_data['sign_dt']
                        gc_j_pos = pendulum.step(parent_j_pos, gc_j_data['control'], gc_j_dt_signed)
                        
                        # Расстояние между внуками
                        distance = np.linalg.norm(gc_i_pos - gc_j_pos)
                        
                        # Возвращаем constraint_distance - distance
                        # Положительное = констрейнт выполнен (расстояние меньше порога)
                        # Отрицательное = констрейнт нарушен (расстояние больше порога)
                        return constraint_distance - distance
                        
                    except Exception as e:
                        # При ошибке возвращаем большое отрицательное значение (нарушение)
                        return -1e6
                
                return constraint_function
            
            # Создаем функцию для текущей пары
            constraint_func = create_constraint_for_pair(gc_i, gc_j, gc_i_info, gc_j_info)
            constraints.append(constraint_func)
            
            # Сохраняем информацию о констрейнте
            constraint_info[pair_idx] = {
                'gc_i': gc_i,
                'gc_j': gc_j,
                'gc_i_parent': gc_i_info['parent_idx'],
                'gc_j_parent': gc_j_info['parent_idx'],
                'target_distance': constraint_distance,
                'original_distance': meeting_info['distance'],
                'meeting_time': meeting_info['meeting_time']
            }
            
            if show:
                gc_i_dir = "F" if gc_i_info['sign_dt'] > 0 else "B"
                gc_j_dir = "F" if gc_j_info['sign_dt'] > 0 else "B"
                print(f"  Констрейнт {pair_idx+1}: gc_{gc_i}({gc_i_dir}) ↔ gc_{gc_j}({gc_j_dir})")
                print(f"    Родители: {gc_i_info['parent_idx']} ↔ {gc_j_info['parent_idx']}")
                print(f"    Целевое расстояние: <= {constraint_distance}")
                print(f"    Исходное расстояние: {meeting_info['distance']:.6f}")
        
        if show:
            print(f"\nСоздано {len(constraints)} функций-констрейнтов")
            print(f"Формат dt_vector: [4 dt детей] + [8 dt внуков] = 12 элементов")
            print(f"Констрейнт выполнен когда функция возвращает >= 0")
            
            print(f"\nПример использования в scipy.optimize.minimize:")
            print(f"constraints = [{{'type': 'ineq', 'fun': func}} for func in constraint_functions]")
        
        return constraints, constraint_info
        
    except Exception as e:
        if show:
            print(f"Ошибка при создании констрейнтов: {e}")
        return [], {}


def test_constraints(constraint_functions, dt_vector, constraint_info, show=False):
    """
    Тестирует функции-констрейнты на заданном векторе dt.
    
    Args:
        constraint_functions: список функций от create_distance_constraints()
        dt_vector: np.array из 12 элементов для тестирования
        constraint_info: информация о констрейнтах
        show: bool - вывод результатов тестирования
        
    Returns:
        dict: результаты тестирования констрейнтов
    """
    
    try:
        if show:
            print("ТЕСТИРОВАНИЕ КОНСТРЕЙНТОВ")
            print("="*40)
            print(f"Тестируем {len(constraint_functions)} констрейнтов")
            print(f"dt_vector: {dt_vector}")
        
        results = {}
        all_satisfied = True
        
        for i, constraint_func in enumerate(constraint_functions):
            constraint_value = constraint_func(dt_vector)
            is_satisfied = constraint_value >= 0
            
            if not is_satisfied:
                all_satisfied = False
            
            results[i] = {
                'value': constraint_value,
                'satisfied': is_satisfied,
                'distance': constraint_info[i]['target_distance'] - constraint_value
            }
            
            if show:
                status = "✓" if is_satisfied else "✗"
                gc_i = constraint_info[i]['gc_i']
                gc_j = constraint_info[i]['gc_j']
                actual_distance = results[i]['distance']
                target_distance = constraint_info[i]['target_distance']
                
                print(f"  {status} Констрейнт {i+1} (gc_{gc_i} ↔ gc_{gc_j}): "
                      f"value={constraint_value:.6f}, расст={actual_distance:.6f}, "
                      f"цель=<={target_distance}")
        
        results['summary'] = {
            'total_constraints': len(constraint_functions),
            'satisfied_count': sum(1 for r in results.values() if isinstance(r, dict) and r.get('satisfied', False)),
            'all_satisfied': all_satisfied
        }
        
        if show:
            summary = results['summary']
            print(f"\nИТОГ: {summary['satisfied_count']}/{summary['total_constraints']} констрейнтов выполнено")
            print(f"Все констрейнты выполнены: {'Да' if summary['all_satisfied'] else 'Нет'}")
        
        return results
        
    except Exception as e:
        if show:
            print(f"Ошибка при тестировании констрейнтов: {e}")
        return {}
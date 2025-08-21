import numpy as np
from .time_profiler import StageProfiler

# Импорты всех необходимых функций из пайплайна
from src.pairs.compute_convergence_tables import compute_distance_derivative_table, compute_grandchild_parent_convergence_table
from src.pairs.find_converging_pairs import find_converging_grandchild_pairs, find_converging_grandchild_parent_pairs
from src.pairs.optimize_grandchild_pair_distance import optimize_grandchild_pair_distance
from src.pairs.optimize_grandchild_parent_distance import optimize_grandchild_parent_distance
from src.pairs.create_meeting_chronology import create_meeting_chronology
from src.pairs.extract_pairs_from_chronology import extract_pairs_from_chronology


def find_optimal_pairs_profiled(tree, show=False):
    """
    Профилированная версия find_optimal_pairs с замерами времени каждого этапа.
    
    Выполняет те же 6 этапов что и оригинальная функция, но с детальным
    профилированием времени выполнения каждого этапа.
    
    Args:
        tree: SporeTree объект с созданными детьми и внуками
        show: bool - вывод промежуточных результатов и профилирования
        
    Returns:
        tuple: (result, profiling_info)
            result: список пар [(gc_i, gc_j, meeting_info), ...] при успехе, None при неудаче
            profiling_info: dict с детальной информацией о временах выполнения
    """
    
    # Создаем профайлер
    profiler = StageProfiler(show=show)
    profiler.start_profiling()
    
    # ============================================================================
    # ПРОВЕРКИ ВХОДНЫХ ДАННЫХ  
    # ============================================================================
    
    try:
        if not hasattr(tree, '_children_created') or not tree._children_created:
            if show:
                print("Ошибка: В дереве не созданы дети. Вызовите tree.create_children()")
            return None, profiler.get_summary()
            
        if not hasattr(tree, '_grandchildren_created') or not tree._grandchildren_created:
            if show:
                print("Ошибка: В дереве не созданы внуки. Вызовите tree.create_grandchildren()")
            return None, profiler.get_summary()
            
    except Exception as e:
        if show:
            print(f"Ошибка проверки входных данных: {e}")
        return None, profiler.get_summary()
    
    # ============================================================================
    # ЭТАП 1: ВЫЧИСЛЕНИЕ ТАБЛИЦ КОНВЕРГЕНЦИИ
    # ============================================================================
    
    profiler.start_stage("Этап 1", "Вычисление таблиц конвергенции")
    
    try:
        convergence_gc_gc = compute_distance_derivative_table(tree.grandchildren, tree.pendulum, show=False)
        convergence_gc_parent = compute_grandchild_parent_convergence_table(tree.grandchildren, tree.children, tree.pendulum, show=False)
        
        if convergence_gc_gc is None or convergence_gc_parent is None:
            profiler.fail_stage("Этап 1", "Не удалось вычислить таблицы")
            return None, profiler.get_summary()
        
        # Подсчитываем статистику
        gc_gc_count = np.sum(~np.isnan(convergence_gc_gc))
        gc_parent_count = np.sum(~np.isnan(convergence_gc_parent))
        
        profiler.end_stage("Этап 1", f"внук-внук: {gc_gc_count}, внук-родитель: {gc_parent_count}")
        
    except Exception as e:
        profiler.fail_stage("Этап 1", f"Ошибка: {e}")
        return None, profiler.get_summary()
    
    # ============================================================================
    # ЭТАП 2: ПОИСК СБЛИЖАЮЩИХСЯ ПАР
    # ============================================================================
    
    profiler.start_stage("Этап 2", "Поиск сближающихся пар")
    
    try:
        converging_gc_pairs = find_converging_grandchild_pairs(convergence_gc_gc, show=False)
        converging_gc_parent_pairs = find_converging_grandchild_parent_pairs(convergence_gc_parent, show=False)
        
        if len(converging_gc_pairs) == 0 and len(converging_gc_parent_pairs) == 0:
            profiler.fail_stage("Этап 2", "Не найдено ни одной сближающейся пары")
            return None, profiler.get_summary()
        
        profiler.end_stage("Этап 2", f"{len(converging_gc_pairs)} внук-внук, {len(converging_gc_parent_pairs)} внук-родитель")
        
    except Exception as e:
        profiler.fail_stage("Этап 2", f"Ошибка: {e}")
        return None, profiler.get_summary()
    
    # ============================================================================
    # ЭТАП 3: ОПТИМИЗАЦИЯ ПАР (обычно самый медленный!)
    # ============================================================================
    
    profiler.start_stage("Этап 3", "Оптимизация пар")
    
    try:
        # Вычисляем параметры для оптимизации
        parent_distances = [np.linalg.norm(parent['position'] - tree.root['position']) for parent in tree.children]
        min_parent_distance = min(parent_distances)
        distance_constraint = min_parent_distance / 10.0
        
        # Адаптивные границы dt
        parent_times = [abs(child['dt']) for child in tree.children]
        max_parent_time = max(parent_times)
        adaptive_dt_max = 2 * max_parent_time
        
        # Оптимизация внук-внук
        gc_gc_optimization_results = {}
        for i, pair in enumerate(converging_gc_pairs):
            gc_i_idx = pair['gc_i']
            gc_j_idx = pair['gc_j']
            pair_name = pair['pair_name']
            
            result = optimize_grandchild_pair_distance(
                gc_i_idx, gc_j_idx,
                tree.grandchildren, tree.children, tree.pendulum,
                dt_bounds=None,  # Адаптивные границы
                root_position=tree.root['position'],
                show=False
            )
            
            gc_gc_optimization_results[pair_name] = result
        
        # Оптимизация внук-родитель
        gc_parent_optimization_results = {}
        for i, pair in enumerate(converging_gc_parent_pairs):
            gc_idx = pair['gc_idx']
            parent_idx = pair['parent_idx']
            pair_name = pair['pair_name']
            
            result = optimize_grandchild_parent_distance(
                gc_idx, parent_idx,
                tree.grandchildren, tree.children, tree.pendulum,
                dt_bounds=None,  # Адаптивные границы
                show=False
            )
            
            gc_parent_optimization_results[pair_name] = result
        
        # Статистика оптимизации
        gc_gc_success = sum(1 for r in gc_gc_optimization_results.values() if r['success'])
        gc_gc_constraint_pass = sum(1 for r in gc_gc_optimization_results.values() 
                                   if r['success'] and r.get('passes_constraint', True))
        gc_parent_success = sum(1 for r in gc_parent_optimization_results.values() if r['success'])
        
        if gc_gc_constraint_pass == 0 and gc_parent_success == 0:
            profiler.fail_stage("Этап 3", "Ни одна оптимизация не прошла constraint")
            return None, profiler.get_summary()
        
        profiler.end_stage("Этап 3", f"{gc_gc_constraint_pass}/{len(converging_gc_pairs)} внук-внук, {gc_parent_success}/{len(converging_gc_parent_pairs)} внук-родитель успешно")
        
    except Exception as e:
        profiler.fail_stage("Этап 3", f"Ошибка: {e}")
        return None, profiler.get_summary()
    
    # ============================================================================
    # ЭТАП 4: ПОСТРОЕНИЕ ИТОГОВЫХ ТАБЛИЦ
    # ============================================================================
    
    profiler.start_stage("Этап 4", "Построение итоговых таблиц")
    
    try:
        n_gc = len(tree.grandchildren)
        n_parents = len(tree.children)
        
        # Инициализируем таблицы
        gc_gc_distance_table = np.full((n_gc, n_gc), np.nan)
        gc_gc_time_i_table = np.full((n_gc, n_gc), np.nan)
        gc_gc_time_j_table = np.full((n_gc, n_gc), np.nan)
        gc_parent_distance_table = np.full((n_gc, n_parents), np.nan)
        gc_parent_time_table = np.full((n_gc, n_parents), np.nan)
        
        # Заполняем таблицы внук-внук
        filled_gc_gc = 0
        for pair_name, result in gc_gc_optimization_results.items():
            if result['success'] and result.get('passes_constraint', True):
                # Извлекаем индексы из имени пары
                parts = pair_name.split('-')
                gc_i_idx = int(parts[0].split('_')[1])
                gc_j_idx = int(parts[1].split('_')[1])
                
                # Заполняем таблицы
                gc_gc_distance_table[gc_i_idx, gc_j_idx] = result['min_distance']
                gc_gc_distance_table[gc_j_idx, gc_i_idx] = result['min_distance']
                
                gc_gc_time_i_table[gc_i_idx, gc_j_idx] = result['optimal_dt_i']
                gc_gc_time_j_table[gc_i_idx, gc_j_idx] = result['optimal_dt_j']
                gc_gc_time_i_table[gc_j_idx, gc_i_idx] = result['optimal_dt_j']
                gc_gc_time_j_table[gc_j_idx, gc_i_idx] = result['optimal_dt_i']
                
                filled_gc_gc += 1
        
        # Заполняем таблицы внук-родитель
        filled_gc_parent = 0
        for pair_name, result in gc_parent_optimization_results.items():
            if result['success']:
                # Извлекаем индексы из имени пары
                parts = pair_name.split('-')
                gc_idx = int(parts[0].split('_')[1])
                parent_idx = int(parts[1].split('_')[1])
                
                # Заполняем таблицы
                gc_parent_distance_table[gc_idx, parent_idx] = result['min_distance']
                gc_parent_time_table[gc_idx, parent_idx] = result['optimal_dt']
                
                filled_gc_parent += 1
        
        profiler.end_stage("Этап 4", f"{filled_gc_gc} ячеек внук-внук, {filled_gc_parent} ячеек внук-родитель")
        
    except Exception as e:
        profiler.fail_stage("Этап 4", f"Ошибка: {e}")
        return None, profiler.get_summary()
    
    # ============================================================================
    # ЭТАП 5: СОЗДАНИЕ ХРОНОЛОГИИ ВСТРЕЧ
    # ============================================================================
    
    profiler.start_stage("Этап 5", "Создание хронологии встреч")
    
    try:
        # Подготавливаем таблицы для хронологии
        gc_gc_tables = {
            'distance_table': gc_gc_distance_table,
            'time_i_table': gc_gc_time_i_table,
            'time_j_table': gc_gc_time_j_table
        }
        
        gc_parent_tables = {
            'distance_table': gc_parent_distance_table,
            'time_table': gc_parent_time_table
        }
        
        chronology_result = create_meeting_chronology(
            gc_gc_tables, gc_parent_tables, 
            tree.grandchildren, tree.children,
            show=False
        )
        
        if chronology_result is None:
            profiler.fail_stage("Этап 5", "Не удалось создать хронологию")
            return None, profiler.get_summary()
        
        # Подсчитываем статистику встреч
        total_meetings = sum(len(meetings) for meetings in chronology_result['chronology'].values())
        unique_gc_meetings = sum(len([m for m in meetings if m['type'] == 'grandchild']) 
                                for meetings in chronology_result['chronology'].values()) // 2
        total_parent_meetings = sum(len([m for m in meetings if m['type'] == 'parent']) 
                                   for meetings in chronology_result['chronology'].values())
        
        profiler.end_stage("Этап 5", f"{unique_gc_meetings} встреч внук-внук, {total_parent_meetings} встреч внук-родитель")
        
    except Exception as e:
        profiler.fail_stage("Этап 5", f"Ошибка: {e}")
        return None, profiler.get_summary()
    
    # ============================================================================
    # ЭТАП 6: ИЗВЛЕЧЕНИЕ ФИНАЛЬНЫХ ПАР
    # ============================================================================
    
    profiler.start_stage("Этап 6", "Извлечение финальных пар")
    
    try:
        final_pairs = extract_pairs_from_chronology(chronology_result, show=False)
        
        if final_pairs is None or len(final_pairs) == 0:
            profiler.fail_stage("Этап 6", "Не удалось извлечь финальные пары")
            return None, profiler.get_summary()
        
        profiler.end_stage("Этап 6", f"{len(final_pairs)} финальных пар найдено")
        
    except Exception as e:
        profiler.fail_stage("Этап 6", f"Ошибка: {e}")
        return None, profiler.get_summary()
    
    # ============================================================================
    # ФИНАЛЬНАЯ СВОДКА
    # ============================================================================
    
    profiler.print_summary()
    
    # Возвращаем результат и информацию о профилировании
    return final_pairs, profiler.get_summary()
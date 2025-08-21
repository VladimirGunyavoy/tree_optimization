import numpy as np
import pandas as pd

# Импорты всех необходимых функций из пайплайна
from .compute_convergence_tables import compute_distance_derivative_table, compute_grandchild_parent_convergence_table
from .find_converging_pairs import find_converging_grandchild_pairs, find_converging_grandchild_parent_pairs
from .optimize_grandchild_pair_distance import optimize_grandchild_pair_distance
from .optimize_grandchild_parent_distance import optimize_grandchild_parent_distance
from .extract_pairs_from_chronology import extract_pairs_from_chronology


def find_optimal_pairs(tree, show=False):
    """
    Находит оптимальные пары внуков в дереве спор через полный пайплайн оптимизации.
    
    Выполняет 6 этапов:
    1. Вычисление скоростей сближения внук-внук и внук-родитель
    2. Поиск сближающихся пар
    3. Оптимизация пар с адаптивными границами dt и distance constraint
    4. Построение итоговых таблиц расстояний и времен
    5. Создание хронологии встреч по времени
    6. Извлечение финальных пар из хронологии
    
    Args:
        tree: SporeTree объект с созданными детьми и внуками
        show: bool - вывод промежуточных результатов (False = тишина + скорость)
        
    Returns:
        list: список пар [(gc_i, gc_j, meeting_info), ...] при успехе
        None: при неудаче (проблемы на любом из этапов)
    """
    
    # ============================================================================
    # ПРОВЕРКИ ВХОДНЫХ ДАННЫХ
    # ============================================================================
    
    try:
        if not hasattr(tree, '_children_created') or not tree._children_created:
            if show:
                print("Ошибка: В дереве не созданы дети. Вызовите tree.create_children()")
            return None
            
        if not hasattr(tree, '_grandchildren_created') or not tree._grandchildren_created:
            if show:
                print("Ошибка: В дереве не созданы внуки. Вызовите tree.create_grandchildren()")
            return None
            
        if len(tree.grandchildren) == 0:
            if show:
                print("Ошибка: В дереве нет внуков")
            return None
            
        pendulum = tree.pendulum
        
        if show:
            print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ПОИСКА ОПТИМАЛЬНЫХ ПАР...")
            print("="*60)
            
    except Exception as e:
        if show:
            print(f"Ошибка проверки входных данных: {e}")
        return None
    
    # ============================================================================
    # ЭТАП 1: ВЫЧИСЛЕНИЕ СКОРОСТЕЙ СБЛИЖЕНИЯ
    # ============================================================================
    
    try:
        if show:
            print("1️⃣ Вычисление скоростей сближения...", end=" ")
        
        # Скорости сближения внук-внук
        convergence_gc_gc = compute_distance_derivative_table(
            tree.grandchildren, pendulum, show=show and False  # Детальный дебаг только при необходимости
        )
        
        # Скорости сближения внук-родитель
        convergence_gc_parent = compute_grandchild_parent_convergence_table(
            tree.grandchildren, tree.children, pendulum, show=show and False
        )
        
        # Быстрая статистика для проверки
        gc_gc_values = convergence_gc_gc.values
        upper_triangle = np.triu(gc_gc_values, k=1)
        valid_values = upper_triangle[upper_triangle != 0]
        gc_gc_converging_count = (valid_values < -1e-6).sum()
        
        gc_parent_values = convergence_gc_parent.values[~np.isnan(convergence_gc_parent.values)]
        gc_parent_converging_count = (gc_parent_values < -1e-6).sum()
        
        if show:
            print(f"✅ ({gc_gc_converging_count} пар внук-внук, {gc_parent_converging_count} пар внук-родитель)")
            
    except Exception as e:
        if show:
            print(f"❌ Ошибка на этапе 1: {e}")
        return None
    
    # ============================================================================
    # ЭТАП 2: ПОИСК СБЛИЖАЮЩИХСЯ ПАР
    # ============================================================================
    
    try:
        if show:
            print("2️⃣ Поиск сближающихся пар...", end=" ")
        
        converging_gc_pairs = find_converging_grandchild_pairs(convergence_gc_gc, show=show and False)
        converging_gc_parent_pairs = find_converging_grandchild_parent_pairs(convergence_gc_parent, show=show and False)
        
        if len(converging_gc_pairs) == 0 and len(converging_gc_parent_pairs) == 0:
            if show:
                print("❌ Не найдено ни одной сближающейся пары")
            return None
        
        if show:
            print(f"✅ ({len(converging_gc_pairs)} внук-внук, {len(converging_gc_parent_pairs)} внук-родитель)")
            
    except Exception as e:
        if show:
            print(f"❌ Ошибка на этапе 2: {e}")
        return None
    
    # ============================================================================
    # ЭТАП 3: ОПТИМИЗАЦИЯ ПАР (с адаптивными границами)
    # ============================================================================
    
    try:
        if show:
            print("3️⃣ Оптимизация пар...", end=" ")
        
        # Вычисляем distance constraint
        parent_distances = [np.linalg.norm(parent['position'] - tree.root['position']) for parent in tree.children]
        min_parent_distance = min(parent_distances)
        distance_constraint = min_parent_distance / 10.0
        
        # Адаптивные границы dt
        parent_times = [abs(child['dt']) for child in tree.children]
        max_parent_time = max(parent_times)
        adaptive_dt_max = 2 * max_parent_time
        
        if show:
            print(f"\n    📏 Distance constraint: {distance_constraint:.5f}")
            print(f"    📊 Адаптивные границы dt: (0.001, {adaptive_dt_max:.5f})")
        
        # Оптимизация внук-внук
        gc_gc_optimization_results = {}
        for pair in converging_gc_pairs:
            gc_i_idx = pair['gc_i']
            gc_j_idx = pair['gc_j']
            pair_name = pair['pair_name']
            
            if show:
                print(f"    🔧 Оптимизация {pair_name}...")
            
            result = optimize_grandchild_pair_distance(
                gc_i_idx, gc_j_idx, 
                tree.grandchildren, tree.children, pendulum,
                dt_bounds=None,  # Адаптивные границы
                root_position=tree.root['position'],
                show=show and False  # Детальный дебаг только при необходимости
            )
            
            gc_gc_optimization_results[pair_name] = result
        
        # Оптимизация внук-родитель
        gc_parent_optimization_results = {}
        for pair in converging_gc_parent_pairs:
            gc_idx = pair['gc_idx']
            parent_idx = pair['parent_idx']
            pair_name = pair['pair_name']
            
            if show:
                print(f"    🔧 Оптимизация {pair_name}...")
            
            result = optimize_grandchild_parent_distance(
                gc_idx, parent_idx,
                tree.grandchildren, tree.children, pendulum,
                dt_bounds=None,  # Адаптивные границы
                show=show and False
            )
            
            gc_parent_optimization_results[pair_name] = result
        
        # Статистика оптимизации
        gc_gc_success = sum(1 for r in gc_gc_optimization_results.values() if r['success'])
        gc_gc_constraint_pass = sum(1 for r in gc_gc_optimization_results.values() 
                                   if r['success'] and r.get('passes_constraint', True))
        gc_parent_success = sum(1 for r in gc_parent_optimization_results.values() if r['success'])
        
        if gc_gc_constraint_pass == 0 and gc_parent_success == 0:
            if show:
                print("❌ Ни одна оптимизация не прошла constraint или не удалась")
            return None
        
        if show:
            print(f"    ✅ ({gc_gc_constraint_pass}/{len(converging_gc_pairs)} внук-внук успешно, {gc_parent_success}/{len(converging_gc_parent_pairs)} внук-родитель успешно)")
            
    except Exception as e:
        if show:
            print(f"❌ Ошибка на этапе 3: {e}")
        return None
    
    # ============================================================================
    # ЭТАП 4: ПОСТРОЕНИЕ ТАБЛИЦ
    # ============================================================================
    
    try:
        if show:
            print("4️⃣ Построение итоговых таблиц...", end=" ")
        
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
                
                gc_parent_distance_table[gc_idx, parent_idx] = result['min_distance']
                gc_parent_time_table[gc_idx, parent_idx] = result['optimal_dt']
                filled_gc_parent += 1
        
        # Создаем DataFrame
        row_names_gc = [f"gc_{i}" for i in range(n_gc)]
        col_names_gc = [f"gc_{i}" for i in range(n_gc)]
        col_names_parent = [f"parent_{i}" for i in range(n_parents)]
        
        distance_gc_gc_df = pd.DataFrame(gc_gc_distance_table, index=row_names_gc, columns=col_names_gc)
        time_i_gc_gc_df = pd.DataFrame(gc_gc_time_i_table, index=row_names_gc, columns=col_names_gc)
        time_j_gc_gc_df = pd.DataFrame(gc_gc_time_j_table, index=row_names_gc, columns=col_names_gc)
        distance_gc_parent_df = pd.DataFrame(gc_parent_distance_table, index=row_names_gc, columns=col_names_parent)
        time_gc_parent_df = pd.DataFrame(gc_parent_time_table, index=row_names_gc, columns=col_names_parent)
        
        if filled_gc_gc == 0 and filled_gc_parent == 0:
            if show:
                print("❌ Ни одна ячейка таблиц не заполнена")
            return None
        
        if show:
            print(f"✅ ({filled_gc_gc} ячеек внук-внук, {filled_gc_parent} ячеек внук-родитель)")
            
    except Exception as e:
        if show:
            print(f"❌ Ошибка на этапе 4: {e}")
        return None
    
    # ============================================================================
    # ЭТАП 5: СОЗДАНИЕ ХРОНОЛОГИИ
    # ============================================================================
    
    try:
        if show:
            print("5️⃣ Создание хронологии встреч...", end=" ")
        
        # Создаем хронологию из таблиц
        chronology = {}
        
        for gc_idx in range(len(tree.grandchildren)):
            meetings = []
            
            # Собираем встречи с другими внуками
            for other_gc_idx in range(len(tree.grandchildren)):
                if gc_idx == other_gc_idx:
                    continue
                    
                distance = distance_gc_gc_df.iloc[gc_idx, other_gc_idx]
                if not np.isnan(distance):
                    time_i = time_i_gc_gc_df.iloc[gc_idx, other_gc_idx]
                    time_j = time_j_gc_gc_df.iloc[gc_idx, other_gc_idx]
                    
                    # Время встречи = максимум из двух времен
                    meeting_time = max(abs(time_i), abs(time_j))
                    
                    meeting = {
                        'type': 'grandchild',
                        'partner': f"gc_{other_gc_idx}",
                        'partner_idx': other_gc_idx,
                        'distance': distance,
                        'time_gc': time_i,
                        'time_partner': time_j,
                        'meeting_time': meeting_time,
                        'who_waits': 'gc' if abs(time_i) > abs(time_j) else 'partner'
                    }
                    meetings.append(meeting)
            
            # Собираем встречи с чужими родителями
            for parent_idx in range(len(tree.children)):
                if parent_idx == tree.grandchildren[gc_idx]['parent_idx']:  # Пропускаем своего родителя
                    continue
                    
                distance = distance_gc_parent_df.iloc[gc_idx, parent_idx]
                if not np.isnan(distance):
                    time_gc = time_gc_parent_df.iloc[gc_idx, parent_idx]
                    
                    meeting = {
                        'type': 'parent',
                        'partner': f"parent_{parent_idx}",
                        'partner_idx': parent_idx,
                        'distance': distance,
                        'time_gc': time_gc,
                        'time_partner': None,
                        'meeting_time': abs(time_gc),
                        'who_waits': None
                    }
                    meetings.append(meeting)
            
            # Сортируем встречи по времени встречи (ХРОНОЛОГИЯ!)
            meetings.sort(key=lambda x: x['meeting_time'])
            chronology[gc_idx] = meetings
        
        # Статистика хронологии
        total_meetings = sum(len(meetings) for meetings in chronology.values())
        unique_gc_meetings = sum(len([m for m in meetings if m['type'] == 'grandchild']) 
                                for meetings in chronology.values()) // 2
        total_parent_meetings = sum(len([m for m in meetings if m['type'] == 'parent']) 
                                   for meetings in chronology.values())
        
        if unique_gc_meetings == 0 and total_parent_meetings == 0:
            if show:
                print("❌ Хронология пуста - нет встреч")
            return None
        
        if show:
            print(f"✅ ({unique_gc_meetings} встреч внук-внук, {total_parent_meetings} встреч внук-родитель)")
            
    except Exception as e:
        if show:
            print(f"❌ Ошибка на этапе 5: {e}")
        return None
    
    # ============================================================================
    # ЭТАП 6: ИЗВЛЕЧЕНИЕ ФИНАЛЬНЫХ ПАР
    # ============================================================================
    
    try:
        if show:
            print("6️⃣ Извлечение пар из хронологии...", end=" ")
        
        # Извлекаем пары
        final_pairs = extract_pairs_from_chronology(chronology, show=show and False)
        
        if not final_pairs:
            if show:
                print("❌ Не удалось извлечь финальные пары")
            return None
        
        if show:
            print(f"✅ ({len(final_pairs)} финальных пар)")
            
    except Exception as e:
        if show:
            print(f"❌ Ошибка на этапе 6: {e}")
        return None
    
    # ============================================================================
    # ИТОГОВАЯ СТАТИСТИКА И РЕЗУЛЬТАТ
    # ============================================================================
    
    if show:
        print("\n" + "="*60)
        print("🏁 ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        print("="*60)
        
        print(f"📊 Адаптивные границы dt: (0.001, {adaptive_dt_max:.5f})")
        print(f"📏 Distance constraint: {distance_constraint:.5f}")
        
        print(f"\n📈 Результаты по этапам:")
        print(f"  1️⃣ Сближающихся пар найдено: {len(converging_gc_pairs)} внук-внук + {len(converging_gc_parent_pairs)} внук-родитель")
        print(f"  2️⃣ Оптимизация успешна: {gc_gc_constraint_pass}/{len(converging_gc_pairs)} внук-внук + {gc_parent_success}/{len(converging_gc_parent_pairs)} внук-родитель")
        print(f"  3️⃣ Таблицы заполнены: {filled_gc_gc} ячеек внук-внук + {filled_gc_parent} ячеек внук-родитель")
        print(f"  4️⃣ Хронология создана: {unique_gc_meetings} уникальных встреч внук-внук + {total_parent_meetings} встреч внук-родитель")
        print(f"  5️⃣ Финальных пар извлечено: {len(final_pairs)}")
        
        # Анализ качества пар
        if final_pairs:
            distances = [meeting['distance'] for _, _, meeting in final_pairs]
            times = [meeting['meeting_time'] for _, _, meeting in final_pairs]
            
            print(f"\n🎯 Качество финальных пар:")
            print(f"  Среднее расстояние: {np.mean(distances):.6f}")
            print(f"  Минимальное расстояние: {np.min(distances):.6f}")
            print(f"  Максимальное расстояние: {np.max(distances):.6f}")
            print(f"  Среднее время встречи: {np.mean(times):.6f}с")
            
            # Качество сближения
            ultra_close = sum(1 for d in distances if d < 1e-6)
            very_close = sum(1 for d in distances if d < 1e-5)
            close = sum(1 for d in distances if d < 1e-4)
            
            print(f"  Ультра-близкие (< 1e-6): {ultra_close}/{len(final_pairs)}")
            print(f"  Очень близкие (< 1e-5): {very_close}/{len(final_pairs)}")
            print(f"  Близкие (< 1e-4): {close}/{len(final_pairs)}")
        
        # Показываем сами пары
        print(f"\n👥 ФИНАЛЬНЫЕ ПАРЫ:")
        for i, (gc_i, gc_j, meeting_info) in enumerate(final_pairs):
            gc_i_info = tree.grandchildren[gc_i]
            gc_j_info = tree.grandchildren[gc_j]
            direction_i = "F" if gc_i_info['dt'] > 0 else "B"
            direction_j = "F" if gc_j_info['dt'] > 0 else "B"
            
            print(f"  {i+1}. gc_{gc_i}({direction_i}) ↔ gc_{gc_j}({direction_j}): "
                  f"расст={meeting_info['distance']:.6f}, t={meeting_info['meeting_time']:.6f}с")
        
        print("="*60)
    
    return final_pairs
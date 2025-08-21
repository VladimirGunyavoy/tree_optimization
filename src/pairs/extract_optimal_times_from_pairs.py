import numpy as np


def extract_optimal_times_from_pairs(pairs, tree, show=False):
    """
    Извлекает оптимальные времена для детей и внуков из найденных пар.
    
    Берет исходные времена из дерева и обновляет только времена спаренных внуков
    оптимальными значениями из пар. Времена детей остаются исходными.
    
    Args:
        pairs: список пар [(gc_i, gc_j, meeting_info), ...] от find_optimal_pairs()
        tree: исходное дерево SporeTree
        show: bool - вывод детальной информации
        
    Returns:
        dict: {
            'dt_children': np.array - времена детей (исходные),
            'dt_grandchildren': np.array - времена внуков (оптимизированные),
            'pair_mapping': dict - информация о парах,
            'unpaired_grandchildren': list - индексы неспаренных внуков,
            'stats': dict - статистика изменений
        }
        None при ошибке
    """
    
    try:
        if not pairs:
            if show:
                print("Ошибка: Список пар пуст")
            return None
            
        if not hasattr(tree, 'children') or not hasattr(tree, 'grandchildren'):
            if show:
                print("Ошибка: Дерево не содержит детей или внуков")
            return None
        
        if show:
            print("ИЗВЛЕЧЕНИЕ ОПТИМАЛЬНЫХ ВРЕМЕН ИЗ ПАР")
            print("="*50)
        
        # ================================================================
        # ИСХОДНЫЕ ВРЕМЕНА
        # ================================================================
        
        original_dt_children = np.array([child['dt'] for child in tree.children])
        original_dt_grandchildren = np.array([gc['dt'] for gc in tree.grandchildren])
        
        if show:
            print(f"Исходные времена:")
            print(f"  Дети: {[f'{dt:+.5f}' for dt in original_dt_children]}")
            print(f"  Внуки: {[f'{dt:+.5f}' for dt in original_dt_grandchildren]}")
        
        # Инициализируем оптимальные времена исходными значениями
        optimal_dt_children = original_dt_children.copy()
        optimal_dt_grandchildren = original_dt_grandchildren.copy()
        
        # ================================================================
        # ОБРАБОТКА ПАР
        # ================================================================
        
        pair_mapping = {}
        paired_grandchildren = set()
        
        if show:
            print(f"\nОбработка {len(pairs)} пар:")
        
        for pair_idx, (gc_i, gc_j, meeting_info) in enumerate(pairs):
            # Извлекаем оптимальные времена из встречи
            optimal_dt_i = meeting_info['time_gc']      # время для gc_i
            optimal_dt_j = meeting_info['time_partner'] # время для gc_j
            
            # Обновляем dt для внуков в паре
            optimal_dt_grandchildren[gc_i] = optimal_dt_i
            optimal_dt_grandchildren[gc_j] = optimal_dt_j
            
            # Сохраняем маппинг для отчета
            pair_mapping[gc_i] = {
                'pair_idx': pair_idx,
                'partner': gc_j,
                'optimal_dt': optimal_dt_i,
                'original_dt': original_dt_grandchildren[gc_i],
                'meeting_distance': meeting_info['distance'],
                'meeting_time': meeting_info['meeting_time']
            }
            pair_mapping[gc_j] = {
                'pair_idx': pair_idx,
                'partner': gc_i,
                'optimal_dt': optimal_dt_j,
                'original_dt': original_dt_grandchildren[gc_j],
                'meeting_distance': meeting_info['distance'],
                'meeting_time': meeting_info['meeting_time']
            }
            
            # Отмечаем как спаренных
            paired_grandchildren.add(gc_i)
            paired_grandchildren.add(gc_j)
            
            if show:
                # Показываем изменения
                change_i = abs(optimal_dt_i - original_dt_grandchildren[gc_i]) > 1e-10
                change_j = abs(optimal_dt_j - original_dt_grandchildren[gc_j]) > 1e-10
                
                print(f"  Пара {pair_idx+1}: gc_{gc_i} ↔ gc_{gc_j}")
                print(f"    gc_{gc_i}: {original_dt_grandchildren[gc_i]:+.6f} → {optimal_dt_i:+.6f} {'(изменен)' if change_i else '(исходный)'}")
                print(f"    gc_{gc_j}: {original_dt_grandchildren[gc_j]:+.6f} → {optimal_dt_j:+.6f} {'(изменен)' if change_j else '(исходный)'}")
                print(f"    Встреча: расст={meeting_info['distance']:.6f}, время={meeting_info['meeting_time']:.6f}с")
        
        # ================================================================
        # СТАТИСТИКА И АНАЛИЗ
        # ================================================================
        
        # Находим неспаренных внуков
        unpaired_grandchildren = [i for i in range(len(tree.grandchildren)) if i not in paired_grandchildren]
        
        # Анализ изменений
        changed_count = sum(1 for i in range(len(tree.grandchildren)) 
                           if abs(optimal_dt_grandchildren[i] - original_dt_grandchildren[i]) > 1e-10)
        
        # Проверяем направления времени
        direction_violations = 0
        for i in range(len(tree.grandchildren)):
            original = original_dt_grandchildren[i]
            optimal = optimal_dt_grandchildren[i]
            
            # Проверяем что знак не поменялся
            if (original > 0 and optimal <= 0) or (original < 0 and optimal >= 0):
                direction_violations += 1
                if show:
                    print(f"ВНИМАНИЕ: gc_{i} изменил направление времени {original:+.6f} → {optimal:+.6f}")
        
        # Вычисляем статистику изменений
        change_ratios = []
        if changed_count > 0:
            for i in range(len(tree.grandchildren)):
                if abs(optimal_dt_grandchildren[i] - original_dt_grandchildren[i]) > 1e-10:
                    if abs(original_dt_grandchildren[i]) > 1e-10:  # Избегаем деления на ноль
                        change_ratio = abs(optimal_dt_grandchildren[i]) / abs(original_dt_grandchildren[i])
                        change_ratios.append(change_ratio)
        
        stats = {
            'total_grandchildren': len(tree.grandchildren),
            'paired_count': len(paired_grandchildren),
            'unpaired_count': len(unpaired_grandchildren),
            'changed_count': changed_count,
            'direction_violations': direction_violations,
            'change_ratios': change_ratios,
            'avg_change_ratio': np.mean(change_ratios) if change_ratios else 1.0,
            'min_change_ratio': np.min(change_ratios) if change_ratios else 1.0,
            'max_change_ratio': np.max(change_ratios) if change_ratios else 1.0
        }
        
        if show:
            print(f"\nИТОГОВАЯ СТАТИСТИКА:")
            print(f"  Спарено внуков: {stats['paired_count']}")
            print(f"  Неспаренных внуков: {stats['unpaired_count']}")
            if unpaired_grandchildren:
                print(f"  Индексы неспаренных: {unpaired_grandchildren}")
            
            print(f"\nСРАВНЕНИЕ ВРЕМЕН (все {len(tree.grandchildren)} внуков):")
            print("  Индекс | Исходное    | Оптимальное | Изменение")
            print("  -------|-------------|-------------|----------")
            for i in range(len(tree.grandchildren)):
                original = original_dt_grandchildren[i]
                optimal = optimal_dt_grandchildren[i]
                changed = abs(optimal - original) > 1e-10
                status = "ИЗМЕНЕН" if changed else "исходное"
                
                print(f"  gc_{i:2d}   | {original:+10.6f} | {optimal:+10.6f} | {status}")
            
            print(f"\nАНАЛИЗ ИЗМЕНЕНИЙ:")
            print(f"  Изменено dt внуков: {stats['changed_count']}/{stats['total_grandchildren']}")
            print(f"  Dt детей остались: исходными (не оптимизировались)")
            
            print(f"\nПРОВЕРКА НАПРАВЛЕНИЙ ВРЕМЕНИ:")
            if stats['direction_violations'] == 0:
                print(f"  Все направления времени сохранены")
            else:
                print(f"  НАРУШЕНИЙ направления времени: {stats['direction_violations']}")
            
            if stats['changed_count'] > 0 and change_ratios:
                print(f"\nСТАТИСТИКА ИЗМЕНЕНИЙ:")
                print(f"  Среднее отношение |новый|/|старый|: {stats['avg_change_ratio']:.3f}")
                print(f"  Минимальное отношение: {stats['min_change_ratio']:.3f}")
                print(f"  Максимальное отношение: {stats['max_change_ratio']:.3f}")
            
            # Форматированный вывод для копирования
            print(f"\n" + "="*50)
            print(f"ГОТОВЫЕ ОПТИМАЛЬНЫЕ ВРЕМЕНА:")
            print(f"="*50)
            
            print(f"dt_children = np.array([{', '.join(f'{dt:.6f}' for dt in optimal_dt_children)}])")
            print(f"dt_grandchildren = np.array([{', '.join(f'{dt:.6f}' for dt in optimal_dt_grandchildren)}])")
            
            print(f"\nПРИМЕР ИСПОЛЬЗОВАНИЯ:")
            print(f"optimized_tree = SporeTree(pendulum, config, dt_children=dt_children, dt_grandchildren=dt_grandchildren)")
        
        return {
            'dt_children': optimal_dt_children,
            'dt_grandchildren': optimal_dt_grandchildren,
            'pair_mapping': pair_mapping,
            'unpaired_grandchildren': unpaired_grandchildren,
            'stats': stats
        }
        
    except Exception as e:
        if show:
            print(f"Ошибка при извлечении времен: {e}")
        return None
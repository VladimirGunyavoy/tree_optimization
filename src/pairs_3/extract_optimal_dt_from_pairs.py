def extract_optimal_dt_from_pairs(pairs, tree, show=False):
    """
    Извлекает оптимальные dt для внуков из найденных пар.
    
    Args:
        pairs: список пар от extract_pairs_from_chronology() 
               [(gc_i, gc_j, meeting_info), ...]
        tree: исходное дерево SporeTree для получения исходных dt
        show: bool - показать процесс извлечения
        
    Returns:
        dict: {
            'dt_grandchildren': np.array из 8 элементов - оптимальные dt для внуков,
            'dt_children': np.array из 4 элементов - исходные dt детей,
            'pair_mapping': dict - маппинг какой внук в какой паре,
            'unpaired_grandchildren': list - индексы внуков без пар
        }
    """
    import numpy as np
    
    if show:
        print("ИЗВЛЕЧЕНИЕ ОПТИМАЛЬНЫХ DT ИЗ ПАР")
        print("=" * 50)
    
    # Получаем исходные dt
    original_dt_children = np.array([child['dt'] for child in tree.children])
    original_dt_grandchildren = np.array([gc['dt'] for gc in tree.grandchildren])
    
    # Инициализируем новые dt исходными значениями
    optimal_dt_grandchildren = original_dt_grandchildren.copy()
    
    # Маппинг пар
    pair_mapping = {}
    paired_grandchildren = set()
    
    if show:
        print(f"Исходные dt внуков: {original_dt_grandchildren}")
        print(f"Обрабатываем {len(pairs)} пар:")
    
    # Обрабатываем каждую пару
    for pair_idx, (gc_i, gc_j, meeting_info) in enumerate(pairs):
        # Извлекаем оптимальные времена из встречи
        optimal_dt_i = meeting_info['time_gc']      # время для gc_i
        optimal_dt_j = meeting_info['time_partner'] # время для gc_j
        
        # Обновляем dt для внуков в паре
        optimal_dt_grandchildren[gc_i] = optimal_dt_i
        optimal_dt_grandchildren[gc_j] = optimal_dt_j
        
        # Сохраняем маппинг
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
            print(f"  Пара {pair_idx+1}: gc_{gc_i} ↔ gc_{gc_j}")
            print(f"    gc_{gc_i}: {original_dt_grandchildren[gc_i]:+.6f} → {optimal_dt_i:+.6f}")
            print(f"    gc_{gc_j}: {original_dt_grandchildren[gc_j]:+.6f} → {optimal_dt_j:+.6f}")
            print(f"    Расстояние: {meeting_info['distance']:.6f}, Время встречи: {meeting_info['meeting_time']:.6f}с")
    
    # Находим неспаренных внуков
    unpaired_grandchildren = [i for i in range(len(tree.grandchildren)) if i not in paired_grandchildren]
    
    if show:
        print(f"\nИТОГ:")
        print(f"  Спарено внуков: {len(paired_grandchildren)}")
        print(f"  Неспаренных внуков: {len(unpaired_grandchildren)}")
        if unpaired_grandchildren:
            print(f"  Индексы неспаренных: {unpaired_grandchildren}")
        
        print(f"\nИТОГОВЫЕ DT ВНУКОВ:")
        for i, (original_dt, optimal_dt) in enumerate(zip(original_dt_grandchildren, optimal_dt_grandchildren)):
            status = "ИЗМЕНЕН" if abs(original_dt - optimal_dt) > 1e-10 else "исходный"
            print(f"    gc_{i}: {original_dt:+.6f} → {optimal_dt:+.6f} ({status})")
    
    return {
        'dt_grandchildren': optimal_dt_grandchildren,
        'dt_children': original_dt_children,
        'pair_mapping': pair_mapping,
        'unpaired_grandchildren': unpaired_grandchildren,
        'pairs_count': len(pairs),
        'paired_count': len(paired_grandchildren)
    }


def create_optimized_tree_from_pairs(pairs, original_tree, pendulum, show=False):
    """
    Создает новое дерево с оптимальными dt из найденных пар.
    
    Args:
        pairs: список пар от extract_pairs_from_chronology()
        original_tree: исходное дерево SporeTree
        pendulum: объект маятника
        show: bool - показать процесс создания
        
    Returns:
        SporeTree: новое дерево с оптимальными dt
    """
    if show:
        print("СОЗДАНИЕ ОПТИМИЗИРОВАННОГО ДЕРЕВА ИЗ ПАР")
        print("=" * 50)
    
    # Извлекаем оптимальные dt
    dt_info = extract_optimal_dt_from_pairs(pairs, original_tree, show=show)
    
    # Создаем новое дерево с оптимальными dt
    optimized_tree = SporeTree(
        pendulum,
        original_tree.config,
        dt_children=dt_info['dt_children'],
        dt_grandchildren=dt_info['dt_grandchildren'],
        show=show
    )
    
    if show:
        print(f"\n✅ Создано оптимизированное дерево:")
        print(f"   Использовано {dt_info['pairs_count']} пар")
        print(f"   Оптимизировано {dt_info['paired_count']}/8 внуков")
        print(f"   Неспаренных: {len(dt_info['unpaired_grandchildren'])}")
    
    return optimized_tree, dt_info


def compare_trees_distances(original_tree, optimized_tree, show=False):
    """
    Сравнивает расстояния между парами в исходном и оптимизированном дереве.
    
    Args:
        original_tree: исходное дерево
        optimized_tree: оптимизированное дерево
        show: bool - показать сравнение
        
    Returns:
        dict: статистика сравнения
    """
    if show:
        print("СРАВНЕНИЕ РАССТОЯНИЙ МЕЖДУ ДЕРЕВЬЯМИ")
        print("=" * 50)
    
    # Вычисляем расстояния в исходном дереве
    original_distances = []
    for i in range(0, len(original_tree.grandchildren), 2):
        if i+1 < len(original_tree.grandchildren):
            pos1 = original_tree.grandchildren[i]['position']
            pos2 = original_tree.grandchildren[i+1]['position']
            distance = np.linalg.norm(pos1 - pos2)
            original_distances.append(distance)
    
    # Вычисляем расстояния в оптимизированном дереве
    optimized_distances = []
    for i in range(0, len(optimized_tree.grandchildren), 2):
        if i+1 < len(optimized_tree.grandchildren):
            pos1 = optimized_tree.grandchildren[i]['position']
            pos2 = optimized_tree.grandchildren[i+1]['position']
            distance = np.linalg.norm(pos1 - pos2)
            optimized_distances.append(distance)
    
    import numpy as np
    
    stats = {
        'original_distances': original_distances,
        'optimized_distances': optimized_distances,
        'original_avg': np.mean(original_distances) if original_distances else 0,
        'optimized_avg': np.mean(optimized_distances) if optimized_distances else 0,
        'original_min': np.min(original_distances) if original_distances else 0,
        'optimized_min': np.min(optimized_distances) if optimized_distances else 0,
        'improvement_ratio': 0
    }
    
    if stats['original_avg'] > 0:
        stats['improvement_ratio'] = stats['optimized_avg'] / stats['original_avg']
    
    if show:
        print(f"Исходное дерево:")
        print(f"  Средн. расстояние: {stats['original_avg']:.6f}")
        print(f"  Мин. расстояние: {stats['original_min']:.6f}")
        print(f"Оптимизированное дерево:")
        print(f"  Средн. расстояние: {stats['optimized_avg']:.6f}")
        print(f"  Мин. расстояние: {stats['optimized_min']:.6f}")
        print(f"Улучшение: {stats['improvement_ratio']:.2f}x")
        
        if len(original_distances) == len(optimized_distances):
            print(f"\nДетальное сравнение пар:")
            for i, (orig, opt) in enumerate(zip(original_distances, optimized_distances)):
                improvement = opt / orig if orig > 0 else 1.0
                print(f"  Пара {i}: {orig:.6f} → {opt:.6f} ({improvement:.2f}x)")
    
    return stats
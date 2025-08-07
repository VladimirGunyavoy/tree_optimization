"""
Создание и работа с топологией дерева спор.
Разделяет медленное создание структуры и быстрый пересчет позиций.
"""
import numpy as np
from pendulum import OptimizationPendulum


def create_tree_topology(initial_position, pendulum, config):
    """
    Создает топологию дерева один раз для быстрого пересчета.
    
    Args:
        initial_position: np.array([theta, theta_dot])
        pendulum: OptimizationPendulum
        config: dict - конфигурация
    
    Returns:
        dict - топология с инструкциями для быстрого пересчета
    """
    show = config["debug"]["show_topology_creation"]
    
    if show:
        print(f"🌱 Создание топологии дерева из позиции {initial_position}")
    
    u_min, u_max = pendulum.get_control_bounds()
    
    # Определяем 4 конфигурации детей (как в оригинале)
    child_configs = [
        {'control': u_max, 'dt_sign': 1, 'name': 'forward_max', 'color': 'blue'},
        {'control': u_max, 'dt_sign': -1, 'name': 'backward_max', 'color': 'cyan'}, 
        {'control': u_min, 'dt_sign': 1, 'name': 'forward_min', 'color': 'green'},
        {'control': u_min, 'dt_sign': -1, 'name': 'backward_min', 'color': 'orange'}
    ]
    
    # Инструкции для построения внуков (обращенное управление)
    grandchild_configs = []
    for parent_idx in range(4):
        parent_config = child_configs[parent_idx]
        reversed_control = -parent_config['control']
        
        # 2 внука от каждого родителя: +dt и -dt
        for local_idx in range(2):
            dt_sign = 1 if local_idx == 0 else -1
            name = f"gc_{parent_idx}_{['forward', 'backward'][local_idx]}"
            
            grandchild_configs.append({
                'parent_idx': parent_idx,
                'local_idx': local_idx,
                'control': reversed_control,
                'dt_sign': dt_sign,
                'name': name,
                'color': 'lightblue' if local_idx == 0 else 'lightcoral'
            })
    
    if show:
        print(f"📊 Создана топология:")
        print(f"  🍄 Детей: {len(child_configs)}")
        print(f"  👶 Внуков: {len(grandchild_configs)}")
    
    topology = {
        'initial_position': initial_position.copy(),
        'child_configs': child_configs,
        'grandchild_configs': grandchild_configs,
        'u_min': u_min,
        'u_max': u_max,
        'config_snapshot': config.copy()
    }
    
    if show:
        print("✅ Топология создана")
    
    return topology





def calculate_grandchildren_positions(topology, dt_vector, pendulum, config):
    """
    Быстро пересчитывает позиции 8 внуков с ПРАВИЛЬНОЙ сортировкой.
    
    Args:
        topology: топология от create_tree_topology()
        dt_vector: np.array(12) - [4 dt детей + 8 dt внуков]
        pendulum: OptimizationPendulum  
        config: dict конфигурация
    
    Returns:
        np.array((8, 2)) - позиции всех 8 внуков В ПРАВИЛЬНОМ ПОРЯДКЕ
    """
    show = config["debug"]["show_calculations"]
    
    if show:
        print(f"🌱 Пересчет позиций внуков с правильной сортировкой")
    
    dt_children = dt_vector[0:4]
    dt_grandchildren = dt_vector[4:12]
    initial_pos = topology['initial_position']
    
    # Шаг 1: Вычисляем позиции 4 детей
    children_with_positions = []
    for i, child_config in enumerate(topology['child_configs']):
        control = child_config['control']
        dt_signed = dt_children[i] * child_config['dt_sign']
        
        child_pos = pendulum.step(initial_pos, control, dt_signed)
        
        # Создаем структуру как в оригинале
        child = {
            'position': child_pos,
            'id': f"child_{i}",
            'name': child_config['name'],
            'color': child_config['color'],
            'control': control,
            'dt': dt_signed,
            'dt_abs': abs(dt_signed)
        }
        children_with_positions.append(child)
        
        if show:
            print(f"  🍄 Ребенок {i}: {child_config['name']}, u={control:+.1f}, dt={dt_signed:+.3f} → {child_pos}")
    
    # Шаг 2: Сортируем детей по углу (как в оригинале)
    def get_angle_child(child):
        dx = child['position'][0] - initial_pos[0] 
        dy = child['position'][1] - initial_pos[1]
        return np.arctan2(dy, dx)
    
    children_sorted = sorted(children_with_positions, key=get_angle_child)
    
    # Переназначаем ID по порядку
    for i, child in enumerate(children_sorted):
        child['id'] = f"child_{i}"
    
    if show:
        print("\n🔄 Дети после сортировки по углу:")
        for i, child in enumerate(children_sorted):
            angle = get_angle_child(child) * 180 / np.pi
            print(f"  {i}: {child['name']} под углом {angle:.1f}°")
    
    # Шаг 3: Создаем внуков
    grandchildren_list = []
    gc_idx = 0
    
    for parent_idx, parent in enumerate(children_sorted):
        reversed_control = -parent['control']
        
        if show:
            print(f"\n👶 От родителя {parent_idx} ({parent['name']}):")
        
        # 2 внука от каждого родителя
        for local_idx in range(2):
            dt_sign = 1 if local_idx == 0 else -1
            dt_signed = dt_grandchildren[gc_idx] * dt_sign
            
            grandchild_pos = pendulum.step(parent['position'], reversed_control, dt_signed)
            
            # Структура для сортировки
            grandchild = {
                'position': grandchild_pos,
                'parent_idx': parent_idx,
                'local_idx': local_idx,
                'global_idx': gc_idx,
                'name': f"gc_{parent_idx}_{['forward', 'backward'][local_idx]}",
                'control': reversed_control,
                'dt': dt_signed
            }
            grandchildren_list.append(grandchild)
            
            if show:
                direction = "forward" if dt_sign > 0 else "backward"
                print(f"    🌱 {local_idx}: u={reversed_control:+.1f}, dt={dt_signed:+.4f} ({direction}) → {grandchild_pos}")
            
            gc_idx += 1
    
    # Шаг 4: Сортируем внуков по углу от корня
    def get_angle_from_root(gc):
        dx = gc['position'][0] - initial_pos[0]
        dy = gc['position'][1] - initial_pos[1] 
        return np.arctan2(dy, dx)
    
    # Сортируем по углу (против часовой стрелки)
    sorted_gc = sorted(grandchildren_list, key=get_angle_from_root, reverse=True)
    
    if show:
        print("\n🔍 Внуки после сортировки по углу:")
        for i, gc in enumerate(sorted_gc):
            angle_deg = get_angle_from_root(gc) * 180 / np.pi
            print(f"  {i}: {gc['name']} (родитель {gc['parent_idx']}) под углом {angle_deg:.1f}°")
    
    # Шаг 5: КРИТИЧЕСКИЙ АЛГОРИТМ - гарантируем что пары от разных родителей
    # Проверяем первые два внука
    if len(sorted_gc) >= 2:
        first_parent = sorted_gc[0]['parent_idx']
        second_parent = sorted_gc[1]['parent_idx']
        
        if show:
            print(f"\n🎯 Проверка первой пары:")
            print(f"  Внук 0: родитель {first_parent}")
            print(f"  Внук 1: родитель {second_parent}")
        
        if first_parent == second_parent:
            # Внуки 0 и 1 от одного родителя - делаем roll на 1
            sorted_gc = np.roll(sorted_gc, 1).tolist()
            if show:
                print("🔄 ПРИМЕНЕН ROLL +1 - первые два внука были от одного родителя")
                print(f"  Новая первая пара: внук 0 (родитель {sorted_gc[0]['parent_idx']}) и внук 1 (родитель {sorted_gc[1]['parent_idx']})")
        else:
            if show:
                print("✅ Первые два внука уже от разных родителей - roll не нужен")
    
    # Шаг 6: Извлекаем позиции в правильном порядке
    sorted_positions = np.array([gc['position'] for gc in sorted_gc])
    
    if show:
        print(f"\n✅ ФИНАЛЬНЫЙ ПОРЯДОК ВНУКОВ:")
        for i, gc in enumerate(sorted_gc):
            print(f"  {i}: {gc['name']} от родителя {gc['parent_idx']}")
        
        print(f"\n📋 ПРОВЕРКА ПАР:")
        for pair_idx in range(4):
            idx1, idx2 = pair_idx * 2, pair_idx * 2 + 1
            parent1 = sorted_gc[idx1]['parent_idx']
            parent2 = sorted_gc[idx2]['parent_idx']
            different = parent1 != parent2
            print(f"  Пара {pair_idx} (внуки {idx1}-{idx2}): родители {parent1}-{parent2} {'✅' if different else '❌'}")
    
    return sorted_positions


def calculate_metrics(grandchild_positions, config):
    """
    Быстро вычисляет метрики: расстояния между парами и площадь.
    
    Args:
        grandchild_positions: np.array((8, 2)) - позиции внуков
        config: dict - конфигурация
    
    Returns:
        dict с расстояниями, средними точками и площадью
    """
    show = config["debug"]["show_calculations"]
    
    if show:
        print("🔍 Вычисляем метрики:")
    
    # Пары внуков: (0,1), (2,3), (4,5), (6,7)
    pair_distances = np.zeros(4)
    mean_points = np.zeros((4, 2))
    
    for pair_idx in range(4):
        idx1 = pair_idx * 2
        idx2 = pair_idx * 2 + 1
        
        pos1 = grandchild_positions[idx1]
        pos2 = grandchild_positions[idx2]
        
        # Расстояние между парой
        distance = np.linalg.norm(pos1 - pos2)
        pair_distances[pair_idx] = distance
        
        # Средняя точка пары
        mean_points[pair_idx] = (pos1 + pos2) / 2
        
        if show:
            print(f"  📏 Пара {pair_idx}: расстояние = {distance:.6f}, средняя = {mean_points[pair_idx]}")
    
    # Площадь четырехугольника (формула Шнура)
    x = mean_points[:, 0]
    y = mean_points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    if show:
        print(f"  📊 Площадь четырехугольника: {area:.6f}")
    
    return {
        'pair_distances': pair_distances,
        'mean_points': mean_points,
        'area': area
    }
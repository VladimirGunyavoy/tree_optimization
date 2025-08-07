"""
Визуализация дерева спор для новой архитектуры.
Адаптация оригинальной функции под topology + positions структуру.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from src.tree_topology import calculate_grandchildren_positions, calculate_metrics


def visualize_optimized_tree(topology, grandchild_positions, metrics, config, dt_vector, pendulum):
    """
    Полная визуализация дерева с детьми, внуками и стрелочками.
    
    Args:
        topology: топология от create_tree_topology()
        grandchild_positions: позиции внуков от calculate_grandchildren_positions()
        metrics: метрики от calculate_metrics()
        config: конфигурация
        dt_vector: np.array(12) - используемые dt
        pendulum: OptimizationPendulum для вычисления позиций детей
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    initial_pos = topology['initial_position']
    dt_children = dt_vector[0:4]
    dt_grandchildren = dt_vector[4:12]
    
    # 1. Корневая спора
    ax.scatter(initial_pos[0], initial_pos[1], 
              c='red', s=100, alpha=0.8, 
              label='Root', edgecolors='black', linewidth=2)
    
    # 2. Вычисляем и рисуем детей (БЕЗ стрелочек)
    child_positions = []
    for i, child_config in enumerate(topology['child_configs']):
        control = child_config['control']
        dt_signed = dt_children[i] * child_config['dt_sign']
        
        # Вычисляем позицию ребенка
        child_pos = pendulum.step(initial_pos, control, dt_signed)
        child_positions.append(child_pos)
        
        # Рисуем ребенка
        ax.scatter(child_pos[0], child_pos[1],
                  c=child_config['color'], s=60, alpha=0.7,
                  edgecolors='black', linewidth=1,
                  label=child_config['name'] if i < 4 else "")
        
        # Номер ребенка
        ax.text(child_pos[0], child_pos[1], str(i), 
               fontsize=12, fontweight='bold', 
               color='white', ha='center', va='center')
    
    # 3. Рисуем внуков и связи с родителями
    for i, pos in enumerate(grandchild_positions):
        gc_config = topology['grandchild_configs'][i]
        parent_idx = gc_config['parent_idx']
        parent_pos = child_positions[parent_idx]
        
        # Точка внука
        ax.scatter(pos[0], pos[1], 
                  c=gc_config['color'], s=40, alpha=0.6,
                  edgecolors='gray', linewidth=1)
        
        # Номер внука
        ax.text(pos[0], pos[1], str(i), 
               fontsize=10, fontweight='bold', 
               color='white', ha='center', va='center',
               bbox=dict(boxstyle="circle,pad=0.15", facecolor='purple', alpha=0.8))
        
        # Стрелочка от родителя к внуку
        arrow = FancyArrowPatch(
            (parent_pos[0], parent_pos[1]),
            (pos[0], pos[1]),
            arrowstyle='->', mutation_scale=10,
            color=gc_config['color'], alpha=0.5, linewidth=1.5
        )
        ax.add_patch(arrow)
    
    # 4. Средние точки и четырехугольник
    mean_points = metrics['mean_points']
    
    # Рисуем средние точки
    for i, mean_pos in enumerate(mean_points):
        ax.scatter(mean_pos[0], mean_pos[1], 
                  c='gold', s=80, alpha=0.9, marker='s',
                  edgecolors='black', linewidth=2,
                  label='Mean Points' if i == 0 else "")
        
        ax.text(mean_pos[0], mean_pos[1], f"P{i}", 
               fontsize=9, fontweight='bold', ha='center', va='center')
    
    # Рисуем четырехугольник
    for i in range(4):
        start_point = mean_points[i]
        end_point = mean_points[(i + 1) % 4]
        
        ax.plot([start_point[0], end_point[0]], 
               [start_point[1], end_point[1]], 
               'gold', linewidth=3, alpha=0.8)
    
    # 5. Настройки графика
    ax.set_xlabel('θ (радианы)', fontsize=12)
    ax.set_ylabel('θ̇ (рад/с)', fontsize=12)
    
    title = f"Дерево спор (правильная сортировка пар)\n"
    title += f"Площадь: {metrics['area']:.6f} | "
    title += f"Расстояния пар: {[f'{d:.4f}' for d in metrics['pair_distances']]}"
    
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_complete_tree(topology, dt_vector, pendulum, config):
    """
    Удобная функция - вычисляет позиции И рисует дерево одним вызовом.
    
    Args:
        topology: топология дерева
        dt_vector: np.array(12) параметры dt
        pendulum: система маятника
        config: конфигурация
        
    Returns:
        tuple: (grandchild_positions, metrics) для дальнейшего анализа
    """
    # Вычисляем позиции
    grandchild_positions = calculate_grandchildren_positions(topology, dt_vector, pendulum, config)
    
    # Вычисляем метрики  
    metrics = calculate_metrics(grandchild_positions, config)
    
    # Рисуем полное дерево
    visualize_optimized_tree(topology, grandchild_positions, metrics, config, dt_vector, pendulum)
    
    return grandchild_positions, metrics
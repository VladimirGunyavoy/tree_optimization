import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

def visualize_spore_tree(tree, title="Дерево спор"):
    """
    Упрощенная визуализация: только точки спор + линии четырехугольника.
    
    Args:
        tree: объект SporeTree
        title: заголовок графика
    """
    fig, ax = plt.subplots(1, 1, figsize=tree.config.figure_size)
    
    # === ТОЛЬКО ТОЧКИ ===
    
    # Корень
    ax.scatter(tree.root['position'][0], tree.root['position'][1], 
              c='#2C3E50', s=80, alpha=0.8, zorder=5)
    
    # Дети + стрелки
    if tree._children_created:
        child_colors = ['#5DADE2', '#A569BD', '#58D68D', '#F4D03F']
        for i, child in enumerate(tree.children):
            ax.scatter(child['position'][0], child['position'][1],
                      c=child_colors[i], s=60, alpha=1, zorder=4)
            
            # НАПРАВЛЕНИЕ стрелки зависит от знака dt
            if child['dt'] > 0:  # forward: от корня к ребенку
                arrow_start = tree.root['position']
                arrow_end = child['position']
            else:  # backward: от ребенка к корню
                arrow_start = child['position']
                arrow_end = tree.root['position']
            
            # ЦВЕТ стрелки зависит от знака управления
            if child['control'] > 0:  # u_max 
                arrow_color = '#FF6B6B'  # коралловый для u_max
            else:  # u_min
                arrow_color = '#1ABC9C'  # бирюзовый для u_min
            
            # Стрелка
            arrow = FancyArrowPatch(
                arrow_start, arrow_end,
                arrowstyle='->', 
                mutation_scale=20,
                color=arrow_color,
                alpha=0.7,
                linewidth=3
            )
            ax.add_patch(arrow)
    
    # Внуки + стрелки
    if tree._grandchildren_created:
        grandchild_colors = ['#D6EAF8', '#E8DAEF', '#D5F4E6', '#FCF3CF']
        grandchildren_to_show = tree.sorted_grandchildren if tree._grandchildren_sorted else tree.grandchildren
        
        for gc in grandchildren_to_show:
            gc_color = grandchild_colors[gc['parent_idx']]
            ax.scatter(gc['position'][0], gc['position'][1],
                      c=gc_color, s=40, alpha=1, zorder=3)
            
            # Стрелка от/к родителю
            parent = tree.children[gc['parent_idx']]
            
            # НАПРАВЛЕНИЕ стрелки зависит от знака dt внука
            if gc['dt'] > 0:  # forward: от родителя к внуку
                arrow_start = parent['position']
                arrow_end = gc['position']
            else:  # backward: от внука к родителю
                arrow_start = gc['position']
                arrow_end = parent['position']
            
            # ЦВЕТ стрелки зависит от знака управления внука
            if gc['control'] > 0:  # u_max
                arrow_color = '#FF6B6B'  # коралловый для u_max
            else:  # u_min  
                arrow_color = '#1ABC9C'  # бирюзовый для u_min
            
            # Стрелка
            arrow = FancyArrowPatch(
                arrow_start, arrow_end,
                arrowstyle='->', 
                mutation_scale=15,
                color=arrow_color,
                alpha=0.5,
                linewidth=2
            )
            ax.add_patch(arrow)
    
    # === СРЕДНИЕ ТОЧКИ И ЛИНИИ ===
    
    if hasattr(tree, 'mean_points') and tree.mean_points is not None:
        mean_points = tree.mean_points
        
        # Средние точки
        ax.scatter(mean_points[:, 0], mean_points[:, 1], 
                  c='#27AE60', s=70, alpha=0.9, zorder=10)
        
        # Линии четырехугольника
        for i in range(4):
            start_point = mean_points[i]
            end_point = mean_points[(i + 1) % 4]
            ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    color='#566573', linewidth=2, alpha=0.2, zorder=9, linestyle='--')
    
    # === НАСТРОЙКИ ГРАФИКА ===
    
    ax.set_xlabel('θ (радианы)')
    ax.set_ylabel('θ̇ (рад/с)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Ручная настройка границ: плотно по X, отступы по Y
    all_x_coords = []
    all_y_coords = []
    
    # Собираем ВСЕ координаты
    all_x_coords.append(tree.root['position'][0])
    all_y_coords.append(tree.root['position'][1])
    
    if tree._children_created:
        for child in tree.children:
            all_x_coords.append(child['position'][0])
            all_y_coords.append(child['position'][1])
    
    if tree._grandchildren_created:
        grandchildren_to_show = tree.sorted_grandchildren if tree._grandchildren_sorted else tree.grandchildren
        for gc in grandchildren_to_show:
            all_x_coords.append(gc['position'][0])
            all_y_coords.append(gc['position'][1])
    
    if hasattr(tree, 'mean_points') and tree.mean_points is not None:
        for mp in tree.mean_points:
            all_x_coords.append(mp[0])
            all_y_coords.append(mp[1])
    
    # Границы: плотно по X, отступы по Y
    x_min, x_max = min(all_x_coords), max(all_x_coords)
    y_min, y_max = min(all_y_coords), max(all_y_coords)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # По X: очень малый отступ (только чтобы точки не касались краев)
    x_margin = max(x_range * 0.02, 1e-6)  # 2% или минимум
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    
    # По Y: нормальные отступы
    y_margin = max(y_range * 0.1, 0.001)  # 10% отступ
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.show()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

def visualize_spore_tree(tree, title="Дерево спор", figsize=None):
    """
    Упрощенная визуализация: только точки спор + линии четырехугольника.
    
    Args:
        tree: объект SporeTree
        title: заголовок графика
        figsize: размер полотна (ширина, высота). Если None, используется из config
    """
    if figsize is None:
        figsize = tree.config.figure_size
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
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
        # Яркие и разнообразные цвета для внуков (по 2 цвета для каждого родителя)
        grandchild_colors = [
            '#FF6B9D', '#C44569',  # От родителя 0: розовый, малиновый
            '#4834DF', '#686DE0',  # От родителя 1: синий, светло-синий
            '#00D2D3', '#01A3A4',  # От родителя 2: бирюзовый, тёмно-бирюзовый
            '#FFA726', '#FF7043'   # От родителя 3: оранжевый, красно-оранжевый
        ]
        grandchildren_to_show = tree.sorted_grandchildren if tree._grandchildren_sorted else tree.grandchildren
        
        for gc in grandchildren_to_show:
            # Используем global_idx для выбора уникального цвета
            gc_color = grandchild_colors[gc['global_idx']]
            ax.scatter(gc['position'][0], gc['position'][1],
                      c=gc_color, s=40, alpha=1, zorder=3, 
                      label=f'Внук {gc["global_idx"]}' if gc['global_idx'] < 8 else None)
            
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
    
    # === СРЕДНИЕ ТОЧКИ И ЛИНИИ УДАЛЕНЫ ===
    # Средние точки пар больше не отображаются для упрощения визуализации
    
    # === НАСТРОЙКИ ГРАФИКА ===
    
    ax.set_xlabel('θ (угол маятника, радианы)')
    ax.set_ylabel('θ̇ (угловая скорость, рад/с)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Настройка легенды
    if tree._grandchildren_created:
        # Создаём легенду только для внуков
        handles, labels = ax.get_legend_handles_labels()
        # Фильтруем только внуков (начинаются с "Внук")
        grandchild_handles = [h for h, l in zip(handles, labels) if l and l.startswith('Внук')]
        grandchild_labels = [l for l in labels if l and l.startswith('Внук')]
        
        if grandchild_handles:
            legend = ax.legend(grandchild_handles, grandchild_labels, 
                             bbox_to_anchor=(1.05, 1), loc='upper left',
                             title='Внуки (индексы)', title_fontsize=10,
                             fontsize=9, framealpha=0.9)
            legend.get_title().set_fontweight('bold')
    
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
    
    # Убираем сбор координат средних точек, так как они больше не отображаются
    
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
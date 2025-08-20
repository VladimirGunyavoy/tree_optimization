import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

def visualize_spore_tree(tree_data, title="Дерево спор", ax=None, figsize=None, show_legend=True):
    """
    Упрощенная визуализация: только точки спор + линии четырехугольника.
    
    Args:
        tree_data: dict с данными дерева ('root', 'children', 'grandchildren')
                   или объект SporeTree
        title: заголовок графика
        ax: объект осей matplotlib для рисования. Если None, создается новый.
        figsize: размер полотна (ширина, высота).
        show_legend: показывать ли легенду.
    """
    if ax is None:
        if figsize is None:
            figsize = (12, 9)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Определяем, работаем мы с объектом tree или со словарем
    if isinstance(tree_data, dict):
        root = tree_data['root']
        children = tree_data['children']
        grandchildren = tree_data['grandchildren']
        _children_created = bool(children)
        _grandchildren_created = bool(grandchildren)
        # В словаре нет информации о сортировке, предполагаем, что она не нужна
        grandchildren_to_show = grandchildren
    else: # SporeTree object
        root = tree_data.root
        children = tree_data.children
        grandchildren = tree_data.grandchildren
        _children_created = tree_data._children_created
        _grandchildren_created = tree_data._grandchildren_created
        grandchildren_to_show = tree_data.sorted_grandchildren if tree_data._grandchildren_sorted else tree_data.grandchildren


    # === ТОЧКИ ===
    
    # Корень
    ax.scatter(root['position'][0], root['position'][1], 
               c='#2C3E50', s=300, alpha=0.9, zorder=5) # Увеличен размер
    
    # Дети + стрелки
    if _children_created:
        child_colors = ['#5DADE2', '#A569BD', '#58D68D', '#F4D03F']
        for i, child in enumerate(children):
            ax.scatter(child['position'][0], child['position'][1],
                      c=child_colors[i], s=300, alpha=1, zorder=4, label=f'{i}') # Увеличен размер
            
            # НАПРАВЛЕНИЕ стрелки зависит от знака dt
            if child['dt'] > 0:  # forward: от корня к ребенку
                arrow_start = root['position']
                arrow_end = child['position']
            else:  # backward: от ребенка к корню
                arrow_start = child['position']
                arrow_end = root['position']
            
            # ЦВЕТ стрелки зависит от знака управления
            if child['control'] > 0:  # u_max 
                arrow_color = '#FF6B6B'
            else:  # u_min
                arrow_color = '#1ABC9C'
            
            arrow = FancyArrowPatch(
                arrow_start, arrow_end, arrowstyle='->', 
                mutation_scale=20, color=arrow_color, alpha=0.7, linewidth=3
            )
            ax.add_patch(arrow)
    
    # Внуки + стрелки
    if _grandchildren_created:
        grandchild_colors = [
            '#FF1744', '#9C27B0', '#2196F3', '#4CAF50',
            '#FF9800', '#795548', '#E91E63', '#607D8B'
        ]
        
        for gc in grandchildren_to_show:
            gc_color = grandchild_colors[gc['global_idx']]
            ax.scatter(gc['position'][0], gc['position'][1],
                      c=gc_color, s=400, alpha=1, zorder=3) # Еще крупнее
            
            # Добавляем номер внука прямо на точку
            ax.text(gc['position'][0], gc['position'][1], str(gc['global_idx']),
                    color='white', ha='center', va='center',
                    fontweight='bold', fontsize=12)

            parent = children[gc['parent_idx']]
            
            if gc['dt'] > 0:
                arrow_start, arrow_end = parent['position'], gc['position']
            else:
                arrow_start, arrow_end = gc['position'], parent['position']
            
            arrow_color = '#FF6B6B' if gc['control'] > 0 else '#1ABC9C'
            
            arrow = FancyArrowPatch(
                arrow_start, arrow_end, arrowstyle='->', 
                mutation_scale=25, color=arrow_color, alpha=0.6, linewidth=3) # Стрелки жирнее
            ax.add_patch(arrow)
    
    # === НАСТРОЙКИ ГРАФИКА ===
    
    ax.set_xlabel('θ (угол, рад)')
    ax.set_ylabel('θ̇ (скорость, рад/с)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Легенда больше не нужна
    # if show_legend and _grandchildren_created:
    #     handles, labels = ax.get_legend_handles_labels()
    #     grandchild_handles = [h for h, l in zip(handles, labels) if l and l.startswith('Внук')]
    #     grandchild_labels = [l for l in labels if l and l.startswith('Внук')]
        
    #     if grandchild_handles:
    #         legend = ax.legend(grandchild_handles, grandchild_labels, 
    #                          bbox_to_anchor=(1.05, 1), loc='upper left',
    #                          title='Внуки (индексы)', title_fontsize=20,
    #                          fontsize=16)
    #         legend.get_title().set_fontweight('bold')
    
    all_x = [root['position'][0]]
    all_y = [root['position'][1]]
    if _children_created:
        for child in children:
            all_x.append(child['position'][0])
            all_y.append(child['position'][1])
    if _grandchildren_created:
        for gc in grandchildren_to_show:
            all_x.append(gc['position'][0])
            all_y.append(gc['position'][1])
            
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_margin = max(x_range * 0.05, 1e-6)
    y_margin = max(y_range * 0.1, 0.001)
    # ax.set_xlim(x_min - x_margin, x_max + x_margin)
    # ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # if ax is None:
    #     plt.tight_layout(rect=[0, 0, 0.85, 1])
    #     plt.show()
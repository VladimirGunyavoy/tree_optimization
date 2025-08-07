import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

def visualize_spore_tree(tree, title="Дерево спор"):
    """
    Визуализирует дерево спор с правильными цветами стрелок и направлениями.
    
    Args:
        tree: объект SporeTree
        title: заголовок графика
    """
    fig, ax = plt.subplots(1, 1, figsize=tree.config.figure_size)
    
    print("🎨 ИСПРАВЛЕННАЯ версия visualize_spore_tree загружена!")
    
    # Рисуем корневую спору
    ax.scatter(tree.root['position'][0], tree.root['position'][1], 
              c=tree.root['color'], s=tree.root['size'], alpha=0.8, 
              label='Root', edgecolors='black', linewidth=2)
    
    # Добавляем подпись к корню
    ax.text(tree.root['position'][0], tree.root['position'][1] + 0.02, 
            'ROOT', fontsize=10, fontweight='bold', 
            ha='center', va='bottom')
    
    # Рисуем детей и стрелки
    if tree._children_created:
        for i, child in enumerate(tree.children):
            # Точка ребенка
            ax.scatter(child['position'][0], child['position'][1],
                      c=child['color'], s=child['size'], alpha=0.8,
                      edgecolors='black', linewidth=1, 
                      label=f"{child['name']} (dt={child['dt']:+.5f})")
            
            # Номер ребенка
            ax.text(child['position'][0], child['position'][1] + 0.01, 
                    str(i), fontsize=9, fontweight='bold', 
                    color='white', ha='center', va='center',
                    bbox=dict(boxstyle="circle,pad=0.1", facecolor='black', alpha=0.8))
            
            # НАПРАВЛЕНИЕ стрелки зависит от знака dt
            if child['dt'] > 0:  # forward: от корня к ребенку
                arrow_start = tree.root['position']
                arrow_end = child['position']
            else:  # backward: от ребенка к корню (ОБРАТНОЕ!)
                arrow_start = child['position']
                arrow_end = tree.root['position']
            
            # ЦВЕТ стрелки зависит от знака управления
            if child['control'] > 0:  # u_max 
                arrow_color = '#FF6B6B'  # коралловый для u_max
            else:  # u_min
                arrow_color = '#1ABC9C'  # бирюзовый для u_min
            
            # Стрелка БЕЗ аннотаций!
            arrow = FancyArrowPatch(
                arrow_start, arrow_end,
                arrowstyle='->', 
                mutation_scale=20,
                color=arrow_color,  # ← ПРАВИЛЬНЫЙ цвет!
                alpha=0.7,
                linewidth=3
            )
            ax.add_patch(arrow)
    
    # Рисуем внуков и стрелки
    if tree._grandchildren_created:
        # Цвета для внуков - светлые версии родительских цветов
        grandchild_colors = {
            '#FF6B6B': '#FFB3BA',  # коралловый → светло-розовый
            '#9B59B6': '#D7BDE2',  # фиолетовый → светло-фиолетовый
            '#1ABC9C': '#A3E4D7',  # бирюзовый → светло-бирюзовый  
            '#F39C12': '#F8C471'   # оранжевый → светло-оранжевый
        }
        
        for gc in tree.grandchildren:
            parent = tree.children[gc['parent_idx']]
            gc_color = grandchild_colors.get(parent['color'], 'gray')
            
            # Точка внука
            ax.scatter(gc['position'][0], gc['position'][1],
                    c=gc_color, s=gc['size'], alpha=0.8,
                    edgecolors='gray', linewidth=1, 
                    label=f"{gc['name']} (dt={gc['dt']:+.5f})")
            
            # Глобальный индекс внука
            ax.text(gc['position'][0], gc['position'][1] + 0.01, 
                    str(gc['global_idx']), fontsize=7, fontweight='bold', 
                    color='white', ha='center', va='center',
                    bbox=dict(boxstyle="circle,pad=0.5", facecolor='darkgray', alpha=0.9))
            
            # НАПРАВЛЕНИЕ стрелки зависит от знака dt внука
            if gc['dt'] > 0:  # forward: от родителя к внуку
                arrow_start = parent['position']
                arrow_end = gc['position']
            else:  # backward: от внука к родителю (ОБРАТНОЕ!)
                arrow_start = gc['position']
                arrow_end = parent['position']
            
            # ЦВЕТ стрелки зависит от знака управления внука
            if gc['control'] > 0:  # u_max
                arrow_color = '#FF6B6B'  # коралловый для u_max
            else:  # u_min  
                arrow_color = '#1ABC9C'  # бирюзовый для u_min
            
            # Стрелка БЕЗ аннотаций!
            arrow = FancyArrowPatch(
                arrow_start, arrow_end,
                arrowstyle='->', 
                mutation_scale=15,
                color=arrow_color,  # ← ПРАВИЛЬНЫЙ цвет!
                alpha=0.6,
                linewidth=2
            )
            ax.add_patch(arrow)
    
    # Настройка графика
    ax.set_xlabel('θ (радианы)')
    ax.set_ylabel('θ̇ (рад/с)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Автоматическое масштабирование
    all_x = [tree.root['position'][0]]
    all_y = [tree.root['position'][1]]
    
    if tree._children_created:
        all_x.extend([child['position'][0] for child in tree.children])
        all_y.extend([child['position'][1] for child in tree.children])
    
    if tree._grandchildren_created:
        all_x.extend([gc['position'][0] for gc in tree.grandchildren])
        all_y.extend([gc['position'][1] for gc in tree.grandchildren])
    
    # Добавляем отступы
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    x_margin = max(x_range * 0.1, 0.05)
    y_margin = max(y_range * 0.1, 0.05)
    
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    
    # Легенда справа
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    plt.show()
import numpy as np
from numba import jit

# Мы будем использовать ту же JIT-компилированную функцию, что и раньше.
# Ее определение должно быть выполнено один раз в вашем коде.
# Если вы используете Jupyter Notebook или аналогичную среду,
# достаточно выполнить эту ячейку один раз.
@jit(nopython=True)
def _calculate_total_area_numba(root_pos, children_pos, grandchildren_pos, parent_indices):
    """
    JIT-компилированная функция для быстрого расчета общей площади.
    Предполагается, что на вход подаются NumPy-массивы.
    """
    total_area = 0.0
    num_grandchildren = grandchildren_pos.shape[0]

    for i in range(num_grandchildren):
        p1 = root_pos
        p2 = children_pos[parent_indices[i]]
        p3 = grandchildren_pos[i]

        area = 0.5 * np.abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        total_area += area
        
    return total_area

# --- ГЛАВНАЯ ФУНКЦИЯ ---
def get_tree_area(tree):
    """
    Принимает объект дерева и возвращает общую площадь.
    Эта функция подготавливает данные для Numba и вызывает JIT-компилированный код.
    """
    if not tree._children_created or not tree._grandchildren_created:
        print("Ошибка: Дерево должно содержать и детей, и внуков.")
        return 0.0

    # 1. Извлечение данных из дерева и преобразование в NumPy-массивы
    root_pos = np.array(tree.root['position'])
    
    children_positions = np.array([child['position'] for child in tree.children])
    
    grandchildren_positions = np.array([gc['position'] for gc in tree.grandchildren])

    # 2. Создание индекса для связи внуков с их родителями
    # Это работает, если внуки упорядочены по родителям:
    # gc[0], gc[1] -> children[0]
    # gc[2], gc[3] -> children[1] и т.д.
    parent_indices = np.repeat(np.arange(len(tree.children)), 2)
    
    # 3. Вызов оптимизированной Numba-функции
    total_area = _calculate_total_area_numba(
        root_pos, 
        children_positions, 
        grandchildren_positions, 
        parent_indices
    )
    
    return total_area

# --- Пример использования (предполагая, что optimized_tree существует) ---
# Например, вы можете вызвать ее так:
# tree_area = get_tree_area(optimized_tree)
# print(f"Общая площадь дерева: {tree_area:.4f}")
import numpy as np
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def _calculate_total_area_numba(root_pos, children_pos, grandchildren_pos, parent_indices):
    """
    JIT-компилированная функция для быстрого расчета общей площади дерева.
    
    Вычисляет сумму площадей треугольников root-child-grandchild для всех внуков.
    
    Args:
        root_pos: np.array([theta, theta_dot]) - позиция корня
        children_pos: np.array((4, 2)) - позиции детей  
        grandchildren_pos: np.array((8, 2)) - позиции внуков
        parent_indices: np.array([0,0,1,1,2,2,3,3]) - индексы родителей для каждого внука
        
    Returns:
        float: общая площадь дерева
    """
    total_area = 0.0
    num_grandchildren = grandchildren_pos.shape[0]

    for i in range(num_grandchildren):
        # Треугольник: корень -> родитель -> внук
        p1 = root_pos
        p2 = children_pos[parent_indices[i]]
        p3 = grandchildren_pos[i]

        # Площадь треугольника через векторное произведение
        area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                        p2[0] * (p3[1] - p1[1]) + 
                        p3[0] * (p1[1] - p2[1]))
        total_area += area
        
    return total_area


class TreeAreaEvaluator:
    """
    Быстрый оценщик площади дерева для оптимизации.
    
    Минимальная реализация без легаси - только то что нужно для площади:
    - JIT-оптимизированные вычисления
    - Быстрое обновление позиций
    - Вычисление общей площади дерева
    """
    
    def __init__(self, tree, show=False):
        """
        Инициализирует оценщик с базовой структурой дерева.
        
        Args:
            tree: SporeTree объект с созданными детьми и внуками
            show: вывод отладочной информации
        """
        
        if show:
            print("Создание TreeAreaEvaluator...")
        
        # Проверки
        if not hasattr(tree, '_children_created') or not tree._children_created:
            raise ValueError("Дерево должно иметь созданных детей")
        if not hasattr(tree, '_grandchildren_created') or not tree._grandchildren_created:
            raise ValueError("Дерево должно иметь созданных внуков")
        
        # Сохраняем ссылки на основные объекты
        self.pendulum = tree.pendulum
        self.root_position = tree.root['position'].copy()
        
        # Извлекаем структурную информацию (не меняется при оптимизации)
        self.children_info = []
        for child in tree.children:
            self.children_info.append({
                'control': child['control'],
                'dt_sign': np.sign(child['dt'])  # +1 для forward, -1 для backward
            })
        
        self.grandchildren_info = []
        for gc in tree.grandchildren:
            self.grandchildren_info.append({
                'parent_idx': gc['parent_idx'],
                'control': gc['control'], 
                'dt_sign': np.sign(gc['dt'])  # +1 для forward, -1 для backward
            })
        
        # Маппинг внук -> родитель для numba (константный массив)
        self.parent_indices = np.array([gc['parent_idx'] for gc in tree.grandchildren], dtype=np.int32)
        
        # Кэш для позиций (переиспользуем массивы)
        self.children_positions = np.zeros((len(self.children_info), 2))
        self.grandchildren_positions = np.zeros((len(self.grandchildren_info), 2))
        
        if show:
            print(f"TreeAreaEvaluator создан:")
            print(f"  Детей: {len(self.children_info)}")
            print(f"  Внуков: {len(self.grandchildren_info)}")
    
    def area(self, dt_vector, show=False):
        """
        Вычисляет общую площадь дерева при заданных временах.
        
        Args:
            dt_vector: np.array из 12 элементов [4 dt детей + 8 dt внуков]
            show: вывод отладочной информации
            
        Returns:
            float: общая площадь дерева
        """
        try:
            dt_vector = np.asarray(dt_vector).ravel()
            
            if len(dt_vector) != 12:
                raise ValueError(f"dt_vector должен содержать 12 элементов, получено {len(dt_vector)}")
            
            # Извлекаем времена (всегда положительные)
            dt_children = np.abs(dt_vector[0:4])
            dt_grandchildren = np.abs(dt_vector[4:12])
            
            if show:
                print(f"Вычисление площади для dt_vector: {dt_vector}")
            
            # Обновляем позиции детей
            for i, child_info in enumerate(self.children_info):
                dt_signed = dt_children[i] * child_info['dt_sign']
                self.children_positions[i] = self.pendulum.step(
                    self.root_position, 
                    child_info['control'], 
                    dt_signed
                )
            
            # Обновляем позиции внуков
            for i, gc_info in enumerate(self.grandchildren_info):
                parent_pos = self.children_positions[gc_info['parent_idx']]
                dt_signed = dt_grandchildren[i] * gc_info['dt_sign']
                self.grandchildren_positions[i] = self.pendulum.step(
                    parent_pos,
                    gc_info['control'],
                    dt_signed
                )
            
            # Вычисляем общую площадь через JIT
            total_area = _calculate_total_area_numba(
                self.root_position,
                self.children_positions,
                self.grandchildren_positions,
                self.parent_indices
            )
            
            if show:
                print(f"Вычисленная площадь: {total_area:.6f}")
            
            return total_area
            
        except Exception as e:
            if show:
                print(f"Ошибка вычисления площади: {e}")
            # При ошибке возвращаем 0 (плохая площадь)
            return 0.0
    
    def test_area_calculation(self, tree, show=False):
        """
        Тестирует правильность вычисления площади по сравнению с исходным деревом.
        
        Args:
            tree: исходное дерево для сравнения
            show: вывод результатов теста
            
        Returns:
            dict: результаты теста
        """
        try:
            # Получаем исходный dt_vector из дерева
            dt_children = np.abs([child['dt'] for child in tree.children])
            dt_grandchildren = np.abs([gc['dt'] for gc in tree.grandchildren])
            original_dt_vector = np.hstack([dt_children, dt_grandchildren])
            
            # Вычисляем площадь через evaluator
            evaluator_area = self.area(original_dt_vector, show=False)
            
            # Вычисляем площадь через get_tree_area для сравнения
            try:
                from .get_tree_area import get_tree_area
                
                reference_area = get_tree_area(tree)
                
                # Сравниваем результаты
                difference = abs(evaluator_area - reference_area)
                relative_error = difference / reference_area if reference_area > 0 else float('inf')
                
                test_result = {
                    'success': difference < 1e-10,
                    'evaluator_area': evaluator_area,
                    'reference_area': reference_area,
                    'difference': difference,
                    'relative_error': relative_error
                }
                
                if show:
                    print(f"ТЕСТ ВЫЧИСЛЕНИЯ ПЛОЩАДИ:")
                    print(f"  Evaluator area: {evaluator_area:.10f}")
                    print(f"  Reference area: {reference_area:.10f}")
                    print(f"  Разность: {difference:.2e}")
                    print(f"  Относительная ошибка: {relative_error:.2e}")
                    print(f"  Тест прошел: {'ДА' if test_result['success'] else 'НЕТ'}")
                
                return test_result
                
            except ImportError:
                if show:
                    print("get_tree_area недоступна для сравнения")
                return {'success': True, 'evaluator_area': evaluator_area}
            
        except Exception as e:
            if show:
                print(f"Ошибка теста: {e}")
            return {'success': False, 'error': str(e)}
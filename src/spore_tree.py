import numpy as np
from typing import List, Dict, Any, Optional

# Импорт конфигурации (должен быть в том же пакете или добавлен в путь)
from spore_tree_config import SporeTreeConfig

class SporeTree:
    """
    Класс для работы с деревом спор маятника.
    """
    
    def __init__(self, pendulum, config: SporeTreeConfig):
        """
        Инициализация дерева спор.
        
        Args:
            pendulum: объект маятника (PendulumSystem)
            config: конфигурация SporeTreeConfig
        """
        self.pendulum = pendulum
        self.config = config
        
        # Валидируем конфиг
        self.config.validate()
        
        # Корневая спора
        self.root = {
            'position': self.config.initial_position.copy(),
            'id': 'root',
            'color': 'red',
            'size': self.config.root_size
        }
        
        # Контейнеры для потомков
        self.children = []
        self.grandchildren = []
        self.sorted_grandchildren = []
        
        # Флаги состояния
        self._children_created = False
        self._grandchildren_created = False
        self._grandchildren_sorted = False
        
        if self.config.show_debug:
            print(f"🌱 SporeTree создан с позицией {self.config.initial_position}")
    
    def create_children(self, dt_children: Optional[np.ndarray] = None, show: bool = None) -> List[Dict[str, Any]]:
        """
        Создает 4 детей с разными управлениями.
        
        Args:
            dt_children: массив из 4 значений dt для детей.
                        Если None, использует config.dt_base для всех.
            show: включать ли отладочную информацию. Если None, использует config.show_debug
        
        Returns:
            List детей
        """
        if show is None:
            show = self.config.show_debug
            
        if self._children_created:
            if show:
                print("⚠️ Дети уже созданы, пересоздаем...")
        
        # Получаем границы управления
        u_min, u_max = self.pendulum.get_control_bounds()
        
        # Настраиваем dt для детей
        if dt_children is None:
            dt_children = np.ones(4) * self.config.dt_base
        else:
            assert len(dt_children) == 4, "dt_children должен содержать ровно 4 элемента"
        
        # Управления и направления: [forw_max, back_max, forw_min, back_min]
        controls = [u_max, u_max, u_min, u_min]
        dt_signs = [1, -1, 1, -1]  # forw: +dt, back: -dt
        colors = ['#FF6B6B', '#9B59B6', '#1ABC9C', '#F39C12']  # Коралловый, фиолетовый, бирюзовый, оранжевый
        names = ['forw_max', 'back_max', 'forw_min', 'back_min']
        
        self.children = []
        
        for i in range(4):
            # Используем dt с нужным знаком
            signed_dt = dt_children[i] * dt_signs[i]
            
            # Вычисляем новую позицию через scipy_rk45_step
            new_position = self.pendulum.scipy_rk45_step(
                state=self.root['position'],
                control=controls[i],
                dt=signed_dt
            )
            
            child = {
                'position': new_position,
                'id': f'child_{i}',
                'name': f'{names[i]}',
                'parent_idx': None,  # корень не имеет индекса
                'control': controls[i],
                'dt': signed_dt,  # храним подписанный dt (+ для forw, - для back)
                'color': colors[i],  # УНИКАЛЬНЫЙ цвет для каждого ребенка
                'size': self.config.child_size,
                'child_idx': i
            }
            
            self.children.append(child)
        
        self._children_created = True
        
        if show:
            print(f"👶 Создано {len(self.children)} детей:")
            for i, child in enumerate(self.children):
                print(f"  {i}: {child['name']} с dt={child['dt']:.4f}, цвет={child['color']}")
        
        return self.children
    
    def create_grandchildren(self, dt_grandchildren: Optional[np.ndarray] = None, show: bool = None) -> List[Dict[str, Any]]:
        """
        Создает внуков (по 2 для каждого ребенка = 8 всего).
        
        Args:
            dt_grandchildren: массив из 8 значений dt для внуков.
                            Если None, использует parent_dt * config.dt_grandchildren_factor
            show: включать ли отладочную информацию. Если None, использует config.show_debug
        
        Returns:
            List внуков
        """
        if show is None:
            show = self.config.show_debug
            
        if not self._children_created:
            raise RuntimeError("Сначала нужно создать детей через create_children()")
            
        if self._grandchildren_created:
            if show:
                print("⚠️ Внуки уже созданы, пересоздаем...")
        
        # Получаем границы управления
        u_min, u_max = self.pendulum.get_control_bounds()
        
        # Настраиваем dt для внуков
        if dt_grandchildren is None:
            # Автоматический режим: dt_внука = dt_родителя * factor
            dt_grandchildren = []
            for child in self.children:
                parent_dt = child['dt']
                grandchild_dt = parent_dt * self.config.dt_grandchildren_factor  # Знак сохраняется!
                dt_grandchildren.extend([grandchild_dt, grandchild_dt])  # по 2 на ребенка
            dt_grandchildren = np.array(dt_grandchildren)
        else:
            assert len(dt_grandchildren) == 8, "dt_grandchildren должен содержать ровно 8 элементов"
        
        self.grandchildren = []
        grandchild_global_idx = 0
        
        for parent_idx, parent in enumerate(self.children):
            # Создаем 2 внуков для текущего родителя
            controls = [u_max, u_min]
            control_names = ['max', 'min']
            
            for local_idx in range(2):
                # dt для текущего внука
                grandchild_dt = dt_grandchildren[grandchild_global_idx]
                
                # Вычисляем позицию внука от позиции родителя
                new_position = self.pendulum.scipy_rk45_step(
                    state=parent['position'],
                    control=controls[local_idx], 
                    dt=grandchild_dt
                )
                
                grandchild = {
                    'position': new_position,
                    'id': f'grandchild_{parent_idx}_{local_idx}',
                    'name': f'gc_{parent_idx}_{local_idx}_{control_names[local_idx]}',
                    'parent_idx': parent_idx,  # индекс родителя (0-3)
                    'local_idx': local_idx,    # локальный индекс у родителя (0-1)
                    'global_idx': grandchild_global_idx,  # глобальный индекс (0-7)
                    'control': controls[local_idx],
                    'dt': grandchild_dt,
                    'color': parent['color'],  # наследуем цвет родителя
                    'size': self.config.grandchild_size
                }
                
                self.grandchildren.append(grandchild)
                grandchild_global_idx += 1
        
        self._grandchildren_created = True
        
        if show:
            print(f"👶 Создано {len(self.grandchildren)} внуков:")
            for gc in self.grandchildren:
                print(f"  {gc['global_idx']}: {gc['name']} от родителя {gc['parent_idx']}, dt={gc['dt']:.6f}")
        
        return self.grandchildren
    
    def get_default_dt_vector(self) -> np.ndarray:
        """
        Возвращает дефолтный вектор времен для оптимизации.
        
        Returns:
            np.array из 12 элементов: [4 dt для детей] + [8 dt для внуков]
        """
        return self.config.get_default_dt_vector()
    
    def reset(self):
        """Сбрасывает дерево к начальному состоянию."""
        self.children = []
        self.grandchildren = []
        self.sorted_grandchildren = []
        self._children_created = False
        self._grandchildren_created = False
        self._grandchildren_sorted = False
        
        if self.config.show_debug:
            print("🔄 Дерево сброшено к начальному состоянию")
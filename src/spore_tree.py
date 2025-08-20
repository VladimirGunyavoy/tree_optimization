import numpy as np
from typing import List, Dict, Any, Optional

# Импорт конфигурации (должен быть в том же пакете или добавлен в путь)
from .spore_tree_config import SporeTreeConfig

class SporeTree:
    """
    Класс для работы с деревом спор маятника.
    """
    
    
    def __init__(self, pendulum, config: SporeTreeConfig, 
                 dt_children: Optional[np.ndarray] = None, 
                 dt_grandchildren: Optional[np.ndarray] = None,
                 auto_create: bool = False,
                 show: bool = None):
        """
        Инициализация дерева спор.
        
        Args:
            pendulum: объект маятника (PendulumSystem)
            config: конфигурация SporeTreeConfig
            dt_children: np.array из 4 элементов - dt для детей (опционально)
            dt_grandchildren: np.array из 8 элементов - dt для внуков (опционально)
            auto_create: bool - создать дерево автоматически по config.dt_base и config.dt_grandchildren_factor
            show: включать ли отладочную информацию. Если None, использует config.show_debug
        """
        if show is None:
            show = config.show_debug
            
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
        self.pairing_candidate_map: Dict[int, List[int]] = {}
        
        # Флаги состояния
        self._children_created = False
        self._grandchildren_created = False
        self._grandchildren_sorted = False
        
        # Кэш для средних точек
        self.mean_points = None
        
        if show:
            print(f"🌱 SporeTree создан с позицией {self.config.initial_position}")
        
        # АВТОМАТИЧЕСКОЕ СОЗДАНИЕ ДЕРЕВА
        if dt_children is not None and dt_grandchildren is not None:
            # Сценарий 1: Заданы конкретные dt массивы
            if show:
                print(f"📊 Создаем дерево с заданными dt:")
                print(f"   dt_children: {dt_children}")
                print(f"   dt_grandchildren: {dt_grandchildren}")
            
            self._create_tree_from_dt_arrays(dt_children, dt_grandchildren, show)
            
        elif auto_create:
            # Сценарий 2: Автоматическое создание по базовым параметрам config
            if show:
                print(f"🤖 Создаем дерево автоматически:")
                print(f"   dt_base: {self.config.dt_base}")
                print(f"   dt_grandchildren_factor: {self.config.dt_grandchildren_factor}")
            
            self._create_tree_auto(show)
            
        elif show:
            print("⚠️ Дерево создано пустым. Используйте create_children() и create_grandchildren()")
    
    def _create_tree_from_dt_arrays(self, dt_children: np.ndarray, dt_grandchildren: np.ndarray, show: bool):
        """Создает дерево из заданных dt массивов."""
        # Проверяем размеры
        assert len(dt_children) == 4, f"dt_children должен содержать 4 элемента, получено {len(dt_children)}"
        assert len(dt_grandchildren) == 8, f"dt_grandchildren должен содержать 8 элементов, получено {len(dt_grandchildren)}"
        
        # Создаем детей
        self.create_children(dt_children=dt_children, show=show)
        
        # Создаем внуков
        self.create_grandchildren(dt_grandchildren=dt_grandchildren, show=show)
        
        if show:
            print(f"✅ Дерево создано: {len(self.children)} детей + {len(self.grandchildren)} внуков")
    
    def _create_tree_auto(self, show: bool):
        """Создает дерево автоматически по базовым параметрам config."""
        # Все дети с базовым dt
        dt_children = np.ones(4) * self.config.dt_base
        
        # Все внуки с уменьшенным dt
        dt_grandchildren = np.ones(8) * self.config.dt_base * self.config.dt_grandchildren_factor
        
        # Создаем детей
        self.create_children(dt_children=dt_children, show=show)
        
        # Создаем внуков  
        self.create_grandchildren(dt_grandchildren=dt_grandchildren, show=show)
        
        if show:
            print(f"✅ Автоматическое дерево создано: {len(self.children)} детей + {len(self.grandchildren)} внуков")



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
            
            # Вычисляем новую позицию через step
            new_position = self.pendulum.step(
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
        Создает 8 внуков (по 2 от каждого родителя) с ОБРАТНЫМ управлением.
        
        ИСПРАВЛЕННАЯ ЛОГИКА:
        - Внук берет ОБРАТНОЕ управление родителя (parent_control → -parent_control)
        - Один внук эволюционирует ВПЕРЕД (+dt), другой НАЗАД (-dt)
        - dt передается положительное, но второй внук его инвертирует
        
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
        
        # Настраиваем dt для внуков
        if dt_grandchildren is None:
            # Автоматический режим: dt_внука = |dt_родителя| * factor
            dt_grandchildren = []
            for child in self.children:
                parent_dt_abs = abs(child['dt'])  # всегда положительное!
                grandchild_dt = parent_dt_abs * self.config.dt_grandchildren_factor
                dt_grandchildren.extend([grandchild_dt, grandchild_dt])  # по 2 на ребенка
            dt_grandchildren = np.array(dt_grandchildren)
        else:
            assert len(dt_grandchildren) == 8, "dt_grandchildren должен содержать ровно 8 элементов"
        
        self.grandchildren = []
        grandchild_global_idx = 0
        
        if show:
            print(f"👶 Создание внуков с ОБРАТНЫМ управлением:")
        
        for parent_idx, parent in enumerate(self.children):
            # ОБРАТНОЕ управление родителя
            reversed_control = -parent['control']
            
            if show:
                print(f"\n  От родителя {parent_idx} ({parent['name']}, u={parent['control']:+.1f}):")
                print(f"    └─ Внуки будут использовать u={reversed_control:+.1f} (обратное)")
            
            # Создаем 2 внуков: один вперед (+dt), другой назад (-dt)
            for local_idx in range(2):
                # dt для текущего внука (всегда передается положительное)
                dt_positive = dt_grandchildren[grandchild_global_idx]
                
                # Первый внук: +dt (вперед), второй внук: -dt (назад)
                if local_idx == 0:
                    final_dt = dt_positive  # вперед во времени
                    direction = "forward"
                else:
                    final_dt = -dt_positive  # назад во времени  
                    direction = "backward"
                
                # Вычисляем позицию внука от позиции родителя
                new_position = self.pendulum.step(
                    state=parent['position'],
                    control=reversed_control,  # ОБРАТНОЕ управление!
                    dt=final_dt
                )
                
                grandchild = {
                    'position': new_position,
                    'id': f'grandchild_{parent_idx}_{local_idx}',
                    'name': f'gc_{parent_idx}_{local_idx}_{direction}',
                    'parent_idx': parent_idx,  # индекс родителя (0-3)
                    'local_idx': local_idx,    # локальный индекс у родителя (0-1)
                    'global_idx': grandchild_global_idx,  # глобальный индекс (0-7)
                    'control': reversed_control,  # ОБРАТНОЕ управление родителя
                    'dt': final_dt,            # финальный dt (может быть отрицательным)
                    'dt_abs': dt_positive,     # абсолютное значение dt  
                    'color': parent['color'],  # наследуем цвет родителя
                    'size': self.config.grandchild_size
                }
                
                self.grandchildren.append(grandchild)
                
                if show:
                    print(f"    🌱 Внук {local_idx}: u={reversed_control:+.1f}, dt={final_dt:+.6f} ({direction}) → {new_position}")
                
                grandchild_global_idx += 1
        
        self._grandchildren_created = True
        
        # Создаем карту кандидатов после того, как все внуки созданы
        self._create_pairing_candidate_map(show=show)
        
        if show:
            print(f"\n✅ Создано {len(self.grandchildren)} внуков с ОБРАТНЫМ управлением")
            print(f"   Структура: от каждого родителя по 2 внука (forward/backward)")
        
        return self.grandchildren

    
    def _create_pairing_candidate_map(self, show: bool = None):
        """
        Создает и кеширует карту кандидатов для спаривания.
        Ключ - global_idx внука, значение - список global_idx всех внуков от других родителей.
        Вызывается автоматически после создания внуков.
        """
        if show is None:
            show = self.config.show_debug
            
        if show:
            print("🗺️  Создание карты кандидатов для спаривания...")

        self.pairing_candidate_map = {}
        
        # Для небольшого количества внуков (8) прямые циклы достаточно эффективны
        for current_grandchild in self.grandchildren:
            current_id = current_grandchild['global_idx']
            current_parent_id = current_grandchild['parent_idx']
            
            candidates = []
            for other_grandchild in self.grandchildren:
                if other_grandchild['parent_idx'] != current_parent_id:
                    candidates.append(other_grandchild['global_idx'])
            
            self.pairing_candidate_map[current_id] = sorted(candidates)

        if show:
            print(f"✅ Карта кандидатов создана. Количество ключей: {len(self.pairing_candidate_map)}")

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

    def sort_and_pair_grandchildren(self, show: bool = None) -> List[Dict[str, Any]]:
        """
        Сортирует 8 внуков по углу от корня и группирует в пары.
        
        КРИТИЧЕСКИЙ МЕТОД с жестким ассертом!
        Проверяет что в каждой паре (0,1), (2,3), (4,5), (6,7) внуки от разных родителей.
        Если проверка не прошла - останавливает программу с четким сообщением об ошибке.
        
        Args:
            show: включать ли отладочную информацию. Если None, использует config.show_debug
            
        Returns:
            List отсортированных внуков
            
        Raises:
            RuntimeError: если внуки не созданы
            AssertionError: если пары содержат внуков от одинаковых родителей
        """
        if show is None:
            show = self.config.show_debug
            
        if not self._grandchildren_created:
            raise RuntimeError("Сначала нужно создать внуков через create_grandchildren()")
        
        if show:
            print(f"🔄 Сортировка {len(self.grandchildren)} внуков по углу от корня...")
        
        def get_angle_from_root(gc):
            """Вычисляет угол от корня до внука."""
            dx = gc['position'][0] - self.root['position'][0]
            dy = gc['position'][1] - self.root['position'][1] 
            return np.arctan2(dy, dx)
        
        # 1. Сортируем по углу (против часовой стрелки)
        sorted_gc = sorted(self.grandchildren, key=get_angle_from_root, reverse=True)
        
        if show:
            print("🔍 Углы внуков после первичной сортировки:")
            for i, gc in enumerate(sorted_gc):
                angle_deg = get_angle_from_root(gc) * 180 / np.pi
                print(f"  {i}: {gc['name']} (родитель {gc['parent_idx']}) под углом {angle_deg:.1f}°")
        
        # 2. Находим первого внука от родителя 0
        roll_offset = 0
        for i, gc in enumerate(sorted_gc):
            if gc['parent_idx'] == 0:
                roll_offset = i
                if show:
                    print(f"🎯 Найден внук родителя 0 на позиции {i}, roll_offset = {roll_offset}")
                break
        
        # 3. Делаем roll чтобы внук родителя 0 стал первым
        sorted_gc = np.roll(sorted_gc, -roll_offset).tolist()
        if show:
            print(f"🔄 Применен roll на {-roll_offset}")
        
        # 4. Проверяем критерий: 1-й внук от другого родителя?
        if len(sorted_gc) >= 2 and sorted_gc[1]['parent_idx'] == 0:
            # Если 1-й тоже от родителя 0, сдвигаем на 1
            sorted_gc = np.roll(sorted_gc, 1).tolist()
            if show:
                print("🔄 Применен дополнительный roll +1")
        
        # 5. ⚠️ КРИТИЧЕСКАЯ ПРОВЕРКА ВСЕХ ПАР - ЖЕСТКИЙ АССЕРТ!
        if show:
            print(f"\n🧐 КРИТИЧЕСКАЯ ПРОВЕРКА ПАР:")
        
        for pair_idx in range(4):
            idx1 = pair_idx * 2      # 0, 2, 4, 6
            idx2 = pair_idx * 2 + 1  # 1, 3, 5, 7
            
            if idx1 < len(sorted_gc) and idx2 < len(sorted_gc):
                parent1 = sorted_gc[idx1]['parent_idx']
                parent2 = sorted_gc[idx2]['parent_idx']
                
                if show:
                    different = parent1 != parent2
                    status = "✅" if different else "❌"
                    print(f"  Пара {pair_idx} (внуки {idx1}-{idx2}): родители {parent1}-{parent2} {status}")
                
                # 🚨 ЖЕСТКИЙ АССЕРТ - остановка программы!
                assert parent1 != parent2, (
                    f"\n❌ КРИТИЧЕСКАЯ ОШИБКА АЛГОРИТМА СОРТИРОВКИ!\n"
                    f"Пара {pair_idx} содержит внуков от одинакового родителя {parent1}!\n"
                    f"Внук {idx1}: {sorted_gc[idx1]['name']} (родитель {parent1})\n"
                    f"Внук {idx2}: {sorted_gc[idx2]['name']} (родитель {parent2})\n"
                    f"Алгоритм сортировки требует исправления!"
                )
            else:
                # Недостаточно внуков - тоже критическая ошибка
                assert False, (
                    f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: Недостаточно внуков для пары {pair_idx}!\n"
                    f"Требуются индексы {idx1} и {idx2}, но есть только {len(sorted_gc)} внуков."
                )
        
        # 6. Если все проверки прошли - сохраняем результат
        self.sorted_grandchildren = sorted_gc
        self._grandchildren_sorted = True
        
        if show:
            print(f"\n✅ ВСЕ ПАРЫ КОРРЕКТНЫ! Сортировка завершена.")
            print(f"   📋 Итоговый порядок внуков:")
            for i, gc in enumerate(sorted_gc):
                print(f"     {i}: {gc['name']} от родителя {gc['parent_idx']}")
        
        return sorted_gc
    

    def calculate_mean_points(self, show: bool = None) -> np.ndarray:
        """
        Вычисляет средние точки для 4 пар отсортированных внуков.
        
        Пары: (0,1), (2,3), (4,5), (6,7) по индексам в sorted_grandchildren.
        Требует предварительного вызова sort_and_pair_grandchildren().
        
        Args:
            show: включать ли отладочную информацию. Если None, использует config.show_debug
            
        Returns:
            np.array размера (4, 2) со средними точками 4 пар
            
        Raises:
            RuntimeError: если внуки не отсортированы
        """
        if show is None:
            show = self.config.show_debug
            
        if not self._grandchildren_sorted:
            raise RuntimeError("Сначала нужно отсортировать внуков через sort_and_pair_grandchildren()")
        
        if show:
            print(f"📊 Вычисление средних точек для {len(self.sorted_grandchildren)} отсортированных внуков...")
        
        # Проверяем что у нас ровно 8 внуков
        assert len(self.sorted_grandchildren) == 8, (
            f"Ожидается 8 внуков, получено {len(self.sorted_grandchildren)}"
        )
        
        means = np.zeros((4, 2))
        
        for pair_idx in range(4):
            # Прямые индексы пары: (0,1), (2,3), (4,5), (6,7)
            idx1 = pair_idx * 2
            idx2 = pair_idx * 2 + 1
            
            # Берем внуков из отсортированного списка
            gc1 = self.sorted_grandchildren[idx1]
            gc2 = self.sorted_grandchildren[idx2]
            
            # Вычисляем среднюю точку пары
            pos1 = gc1['position']
            pos2 = gc2['position']
            mean_point = (pos1 + pos2) / 2
            means[pair_idx] = mean_point
            
            if show:
                distance = np.linalg.norm(pos1 - pos2)
                print(f"  📏 Пара {pair_idx} (внуки {idx1}-{idx2}):")
                print(f"     {gc1['name']} (родитель {gc1['parent_idx']}) → {pos1}")
                print(f"     {gc2['name']} (родитель {gc2['parent_idx']}) → {pos2}")
                print(f"     Расстояние: {distance:.6f}, Средняя точка: {mean_point}")
        
        # Сохраняем результат в объекте
        self.mean_points = means
        
        if show:
            print(f"\n✅ Средние точки вычислены и сохранены в tree.mean_points")
            print(f"   🎯 Размерность: {means.shape}")
        
        return means




    # ─── добавьте в класс SporeTree ─────────────────────────────────────
    # def update_positions(self,
    #                     dt_children: np.ndarray,
    #                     dt_grandchildren: np.ndarray,
    #                     recompute_means: bool = True):
    #     """
    #     Пересчитывает координаты детей и внуков, сохраняя
    #     ИСХОДНЫЙ порядок self.children и self.sorted_grandchildren.
    #     dt_children        – 4 положительных числа
    #     dt_grandchildren   – 8 положительных чисел
    #     """
    #     assert self._grandchildren_sorted, (
    #         "Сперва создайте дерево (children+grandchildren) и вызовите "
    #         "sort_and_pair_grandchildren(), чтобы зафиксировать порядок."
    #     )

    #     # 1. дети
    #     for i, child in enumerate(self.children):
    #         signed_dt = np.sign(child['dt']) * dt_children[i]
    #         child['dt'] = signed_dt
    #         child['position'] = self.pendulum.step(
    #             state=self.root['position'],
    #             control=child['control'],
    #             dt=signed_dt
    #         )

    #     # 2. внуки (используем global_idx, sign dt остаётся как было)
    #     for gc in self.grandchildren:
    #         j = gc['global_idx']                 # 0‥7
    #         signed_dt = np.sign(gc['dt']) * dt_grandchildren[j]
    #         gc['dt'] = signed_dt
    #         gc['dt_abs'] = abs(signed_dt)

    #         parent = self.children[gc['parent_idx']]
    #         gc['position'] = self.pendulum.step(
    #             state=parent['position'],
    #             control=gc['control'],
    #             dt=signed_dt
    #         )

    #     # 3. пересчитаем средние точки
    #     if recompute_means:
    #         self.calculate_mean_points(show=False)
    # ────────────────────────────────────────────────────────────────────

    def update_positions(self, dt_children: np.ndarray, dt_grandchildren: np.ndarray, 
                                        recompute_means: bool = True, show: bool = False):
        """
        🚀 ОПТИМИЗИРОВАННАЯ JIT версия update_positions() 
        
        Основана на результатах бенчмарка: JIT одиночные вызовы быстрее batch в 2x!
        Убираем все лишние операции и проверки.
        """
        # МИНИМАЛЬНЫЕ проверки (только критические)
        assert self._grandchildren_sorted, "Дерево должно быть отсортировано"

        # ═══════════════════════════════════════════════════════════════════
        # ЭТАП 1: 🔥 БЫСТРОЕ ОБНОВЛЕНИЕ ДЕТЕЙ (4 JIT вызова)
        # ═══════════════════════════════════════════════════════════════════
        
        root_pos = self.root['position']  # Кешируем обращение
        
        # Обновляем детей напрямую без промежуточных массивов
        for i in range(4):  # Развернутый цикл быстрее enumerate
            child = self.children[i]
            dt_sign = 1 if child['dt'] > 0 else -1  # Быстрее np.sign()
            signed_dt = dt_children[i] * dt_sign
            
            # Прямое обновление без копирований
            child['dt'] = signed_dt
            child['position'] = self.pendulum.step(root_pos, child['control'], signed_dt)

        # ═══════════════════════════════════════════════════════════════════
        # ЭТАП 2: 🔥 БЫСТРОЕ ОБНОВЛЕНИЕ ВНУКОВ (8 JIT вызовов)
        # ═══════════════════════════════════════════════════════════════════
        
        # Предвычисляем позиции детей для быстрого доступа
        child_positions = [child['position'] for child in self.children]
        
        # Обновляем внуков напрямую по global_idx
        for gc in self.grandchildren:
            j = gc['global_idx']  # 0-7
            parent_pos = child_positions[gc['parent_idx']]  # Быстрый доступ
            
            dt_sign = 1 if gc['dt'] > 0 else -1  # Быстрее np.sign()
            signed_dt = dt_grandchildren[j] * dt_sign
            
            # Прямое обновление
            gc['dt'] = signed_dt
            gc['dt_abs'] = abs(signed_dt)  # Inline abs быстрее np.abs
            gc['position'] = self.pendulum.step(parent_pos, gc['control'], signed_dt)

        # ═══════════════════════════════════════════════════════════════════
        # ЭТАП 3: БЫСТРЫЙ ПЕРЕСЧЕТ СРЕДНИХ ТОЧЕК (если нужно)
        # ═══════════════════════════════════════════════════════════════════
        
        if recompute_means:
            # Inline вычисление вместо вызова метода (убираем overhead)
            self.mean_points = np.zeros((4, 2))
            
            sorted_gc = self.sorted_grandchildren
            for pair_idx in range(4):
                idx1, idx2 = pair_idx * 2, pair_idx * 2 + 1
                pos1 = sorted_gc[idx1]['position']
                pos2 = sorted_gc[idx2]['position']
                self.mean_points[pair_idx] = (pos1 + pos2) * 0.5  # * 0.5 быстрее / 2
            
        if show:
            print("🔄 JIT update: 4 детей + 8 внуков за 12 оптимизированных вызовов")


    def mean_points(self, show: bool = None) -> np.ndarray:
        """
        🚀 Быстрая версия calculate_mean_points без лишних проверок.
        """
        if self.mean_points is None:
            self.mean_points = np.zeros((4, 2))
        
        sorted_gc = self.sorted_grandchildren  # Один доступ к атрибуту
        
        # Развернутый цикл для максимальной скорости
        self.mean_points[0] = (sorted_gc[0]['position'] + sorted_gc[1]['position']) * 0.5
        self.mean_points[1] = (sorted_gc[2]['position'] + sorted_gc[3]['position']) * 0.5
        self.mean_points[2] = (sorted_gc[4]['position'] + sorted_gc[5]['position']) * 0.5
        self.mean_points[3] = (sorted_gc[6]['position'] + sorted_gc[7]['position']) * 0.5
        
        return self.mean_points


    # Также можно добавить к классу:
    def reset_for_optimization(self):
        """Быстрый сброс перед оптимизацией - убираем только необходимое."""
        # НЕ пересоздаем массивы, только обнуляем флаги
        self._children_created = False
        self._grandchildren_created = False
        self._grandchildren_sorted = False
        # mean_points оставляем - переиспользуем массив